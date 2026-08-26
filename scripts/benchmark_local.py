import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.model import V5Config, V5Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()

    device = torch.device("cuda")

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    config = V5Config(
        block_size=args.context,
    )

    model = V5Model(config).to(device)
    model.gradient_checkpointing = args.checkpointing
    model.train()

    if args.compile:
        print("torch.compile: ON")
        model = torch.compile(model)
    else:
        print("torch.compile: OFF")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )

    B = args.batch_size
    T = args.context

    print("GPU:", torch.cuda.get_device_name(0))
    print("batch:", B)
    print("context:", T)
    print("gradient checkpointing:", args.checkpointing)
    print("tokens/step:", B * T)
    print()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    throughputs = []

    for step in range(args.steps):
        x = torch.randint(
            0,
            config.vocab_size,
            (B, T),
            device=device,
        )

        y = torch.randint(
            0,
            config.vocab_size,
            (B, T),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            _, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - start

        tokens = B * T
        tok_s = tokens / dt

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"step {step:02d} | "
            f"{dt:.3f}s | "
            f"{tok_s:,.0f} tok/s | "
            f"peak {peak:.2f} GB"
        )

        # Ignore first two warmup iterations.
        if step >= 2:
            throughputs.append(tok_s)

    avg = sum(throughputs) / len(throughputs)

    print()
    print("=" * 45)
    print(f"AVERAGE: {avg:,.0f} tok/s")
    print(
        "PEAK VRAM:",
        f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
    )
    print("=" * 45)


if __name__ == "__main__":
    main()
