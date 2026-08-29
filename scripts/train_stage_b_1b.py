from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.binary_mix_stream import BinaryMixStream
from src.checkpoint import (
    load_checkpoint,
    move_optimizer_to_device,
    save_checkpoint,
)
from src.model import V5Config, V5Model
from src.train_utils import build_optimizer


BASE = "checkpoints/v5_6b/step_011097.pt"
DATA = "data/stage_b_1b_packed"
OUT = Path("checkpoints/stage_b_7b")

MICRO_BATCH = 12
GRAD_ACCUM = 22
LOCAL_STEPS = 1850

START_LR = 4e-5
END_LR = 4e-7

SEED = 1337
SAVE_EVERY = 100
LOG_EVERY = 10

RESUME_STAGE_B = None
# Example:
# RESUME_STAGE_B = "checkpoints/stage_b_7b/latest.pt"


device = torch.device("cuda")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(stream, batch_size):
    return torch.stack(
        [next(stream) for _ in range(batch_size)]
    ).to(device, non_blocking=True)


seed_all(SEED)

torch.set_float32_matmul_precision("high")

config = V5Config()

print("=" * 72)
print("V5 STAGE B — 6B -> 7B")
print("=" * 72)
print("GPU:", torch.cuda.get_device_name(0))

model = V5Model(config)
model.gradient_checkpointing = True
model.to(device)
model.train()

optimizer = build_optimizer(
    model,
    learning_rate=START_LR,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    fused=True,
)

stream = BinaryMixStream(
    DATA,
    block_size=config.block_size,
    seed=SEED,
)

base_step = 11097
base_tokens = 5_999_837_184
local_start = 0


if RESUME_STAGE_B is None:
    print("\nLoading untouched 6B checkpoint...")

    ckpt = torch.load(
        BASE,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    assert int(ckpt["step"]) == base_step
    assert int(ckpt["tokens_seen"]) == base_tokens

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    # Preserve Stage-A AdamW moments.
    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    move_optimizer_to_device(
        optimizer,
        device,
    )

    # Stage B begins from Stage A's final LR.
    for group in optimizer.param_groups:
        group["lr"] = START_LR
        group["initial_lr"] = START_LR

    del ckpt
    gc.collect()

else:
    print(
        "\nResuming Stage B from:",
        RESUME_STAGE_B,
    )

    info = load_checkpoint(
        RESUME_STAGE_B,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        stream=stream,
        device=device,
    )

    global_step = int(info["step"])
    base_tokens_from_ckpt = int(info["tokens_seen"])

    local_start = global_step - base_step

    if not 0 <= local_start < LOCAL_STEPS:
        raise RuntimeError(
            f"Invalid Stage-B local step: {local_start}"
        )

    expected_tokens = (
        base_tokens
        + local_start
        * MICRO_BATCH
        * GRAD_ACCUM
        * config.block_size
    )

    if base_tokens_from_ckpt != expected_tokens:
        raise RuntimeError(
            "Stage-B token count mismatch."
        )


tokens_per_step = (
    MICRO_BATCH
    * GRAD_ACCUM
    * config.block_size
)

assert tokens_per_step == 540_672
assert tokens_per_step * LOCAL_STEPS == 1_000_243_200
assert stream.total_blocks == 488_400


def lr_for_step(local_step: int) -> float:
    progress = local_step / LOCAL_STEPS

    cosine = 0.5 * (
        1.0 + np.cos(np.pi * progress)
    )

    return (
        END_LR
        + (START_LR - END_LR) * cosine
    )


# Restore correct LR if resuming.
resume_lr = lr_for_step(local_start)

for group in optimizer.param_groups:
    group["lr"] = resume_lr


print()
print("micro batch       :", MICRO_BATCH)
print("grad accumulation :", GRAD_ACCUM)
print("tokens / step     :", f"{tokens_per_step:,}")
print("Stage-B steps     :", LOCAL_STEPS)
print("Stage-B tokens    :", f"{tokens_per_step * LOCAL_STEPS:,}")
print("final total       :", f"{base_tokens + tokens_per_step * LOCAL_STEPS:,}")
print("start LR          :", f"{START_LR:.2e}")
print("end LR            :", f"{END_LR:.2e}")
print("resume local step :", local_start)
print("stream position   :", f"{stream.position:,}/{stream.total_blocks:,}")
print()

OUT.mkdir(parents=True, exist_ok=True)

print("torch.compile: ON")
model = torch.compile(model)

tokens_seen = (
    base_tokens
    + local_start * tokens_per_step
)

last_time = time.perf_counter()
last_tokens = tokens_seen


for local_step in range(
    local_start + 1,
    LOCAL_STEPS + 1,
):
    global_step = base_step + local_step

    # Set LR for this update.
    lr = lr_for_step(local_step - 1)

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    for _ in range(GRAD_ACCUM):
        batch = make_batch(
            stream,
            MICRO_BATCH,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            _, loss = model(
                batch,
                batch,
            )

            scaled_loss = (
                loss / GRAD_ACCUM
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss "
                f"at global step {global_step}"
            )

        total_loss += (
            loss.detach().float().item()
        )

        scaled_loss.backward()

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
    )

    if not torch.isfinite(grad_norm):
        raise RuntimeError(
            f"non-finite gradient "
            f"at global step {global_step}"
        )

    optimizer.step()

    tokens_seen += tokens_per_step

    if (
        local_step == 1
        or local_step % LOG_EVERY == 0
    ):
        torch.cuda.synchronize()

        now = time.perf_counter()

        elapsed = now - last_time
        token_delta = tokens_seen - last_tokens

        tok_s = token_delta / elapsed

        avg_loss = (
            total_loss / GRAD_ACCUM
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"stage {local_step:04d}/{LOCAL_STEPS} | "
            f"global {global_step:05d} | "
            f"loss {avg_loss:.4f} | "
            f"lr {lr:.3e} | "
            f"grad {grad_norm.item():.2f} | "
            f"tokens {tokens_seen:,} | "
            f"{tok_s:,.0f} tok/s | "
            f"peak {peak_vram:.2f} GB"
        )

        last_time = now
        last_tokens = tokens_seen

    if local_step % SAVE_EVERY == 0:
        latest = OUT / "latest.pt"

        print("saving:", latest)

        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            stream=stream,
            step=global_step,
            tokens_seen=tokens_seen,
            config=config,
        )

        print("checkpoint saved")


final_path = OUT / "step_012947_7b.pt"

save_checkpoint(
    final_path,
    model=model,
    optimizer=optimizer,
    scheduler=None,
    stream=stream,
    step=base_step + LOCAL_STEPS,
    tokens_seen=tokens_seen,
    config=config,
)

print()
print("=" * 72)
print("STAGE B COMPLETE")
print("step        :", base_step + LOCAL_STEPS)
print("tokens      :", f"{tokens_seen:,}")
print("stream pos  :", f"{stream.position:,}/{stream.total_blocks:,}")
print("checkpoint  :", final_path)
print("=" * 72)

assert tokens_seen == 7_000_080_384
assert stream.position == 488_400
