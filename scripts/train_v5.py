from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.data_stream import FineWebPackedStream
from src.model import (
    V5Config,
    V5Model,
    count_parameters,
)
from src.train_utils import (
    build_cosine_scheduler,
    build_optimizer,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--micro-batch",
        type=int,
        default=1,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-steps",
        type=int,
        required=True,
    )

    p.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=4e-4,
    )

    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    p.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/v5_pretrain",
    )

    p.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    p.add_argument(
        "--compile",
        action="store_true",
    )

    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )

    p.add_argument(
        "--fused-adamw",
        action="store_true",
    )

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(
    stream,
    batch_size,
    device,
):
    rows = [
        next(stream)
        for _ in range(batch_size)
    ]

    batch = torch.stack(
        rows,
        dim=0,
    )

    return batch.to(
        device,
        non_blocking=True,
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "V5 pretraining requires GPU."
        )

    device = torch.device("cuda")
    seed = 1337

    seed_everything(seed)

    torch.set_float32_matmul_precision(
        "high"
    )

    config = V5Config()

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "parameters:",
        f"{count_parameters(V5Model(config)):,}",
    )

    model = V5Model(config)

    model.gradient_checkpointing = (
        args.gradient_checkpointing
    )

    model.to(device)
    model.train()

    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        fused=args.fused_adamw,
    )

    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    stream = FineWebPackedStream(
        tokenizer_path="tokenizer",
        block_size=config.block_size,
        shuffle_buffer=args.shuffle_buffer,
        seed=seed,
    )

    start_step = 0
    tokens_seen = 0

    if args.resume is not None:
        info = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stream=stream,
            device=device,
        )

        start_step = info["step"]
        tokens_seen = info["tokens_seen"]

        print(
            f"resumed from step {start_step:,}"
        )

        print(
            f"resumed tokens {tokens_seen:,}"
        )

    if args.compile:
        print("torch.compile: ON")
        model = torch.compile(model)
    else:
        print("torch.compile: OFF")

    tokens_per_micro = (
        args.micro_batch
        * config.block_size
    )

    tokens_per_step = (
        tokens_per_micro
        * args.grad_accum
    )

    print()
    print(
        "micro batch:",
        args.micro_batch,
    )

    print(
        "gradient accumulation:",
        args.grad_accum,
    )

    print(
        "tokens / optimizer step:",
        f"{tokens_per_step:,}",
    )

    print(
        "planned tokens:",
        f"{args.max_steps * tokens_per_step:,}",
    )

    print(
        "gradient checkpointing:",
        args.gradient_checkpointing,
    )

    print()

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_log_time = time.perf_counter()
    last_log_tokens = tokens_seen

    for step in range(
        start_step + 1,
        args.max_steps + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0

        for _ in range(args.grad_accum):
            batch = make_batch(
                stream,
                args.micro_batch,
                device,
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
                    loss
                    / args.grad_accum
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss "
                    f"at step {step}"
                )

            total_loss += (
                loss.detach().float().item()
            )

            scaled_loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite gradient norm "
                f"at step {step}"
            )

        optimizer.step()
        scheduler.step()

        tokens_seen += tokens_per_step

        if (
            step == 1
            or step % args.log_every == 0
        ):
            torch.cuda.synchronize()

            now = time.perf_counter()

            elapsed = (
                now - last_log_time
            )

            token_delta = (
                tokens_seen
                - last_log_tokens
            )

            tok_s = (
                token_delta / elapsed
            )

            avg_loss = (
                total_loss
                / args.grad_accum
            )

            lr = scheduler.get_last_lr()[0]

            peak_vram = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"step {step:06d} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.3e} | "
                f"grad {grad_norm.item():.2f} | "
                f"tokens {tokens_seen:,} | "
                f"{tok_s:,.0f} tok/s | "
                f"peak {peak_vram:.2f} GB"
            )

            last_log_time = now
            last_log_tokens = tokens_seen

        if (
            args.save_every > 0
            and step % args.save_every == 0
        ):
            path = (
                checkpoint_dir
                / f"step_{step:06d}.pt"
            )

            print(
                f"saving {path} ..."
            )

            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=stream,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
            )

            print("checkpoint saved")

    final_path = (
        checkpoint_dir
        / "final.pt"
    )

    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stream=stream,
        step=args.max_steps,
        tokens_seen=tokens_seen,
        config=config,
    )

    print()
    print(
        "TRAINING COMPLETE"
    )

    print(
        "tokens:",
        f"{tokens_seen:,}",
    )

    print(
        "checkpoint:",
        final_path,
    )


if __name__ == "__main__":
    main()
