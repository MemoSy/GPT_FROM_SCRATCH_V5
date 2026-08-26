import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.model import V5Config, V5Model


def main():
    assert torch.cuda.is_available()

    device = torch.device("cuda")
    torch.manual_seed(1337)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("HIP:", torch.version.hip)

    config = V5Config()

    model = V5Model(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )

    B = 1
    T = 512
    steps = 20

    torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        # Predictable synthetic sequence:
        # 0,1,2,...,255,0,1,2,...
        offset = step % 256

        tokens = (
            torch.arange(T + 1, device=device)
            + offset
        ) % 256

        x = tokens[:-1].unsqueeze(0).long()
        y = tokens[1:].unsqueeze(0).long()

        optimizer.zero_grad(set_to_none=True)

        start = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()

        bad_grads = 0

        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    bad_grads += 1

        if bad_grads:
            raise RuntimeError(
                f"{bad_grads} non-finite gradient tensors "
                f"at step {step}"
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite grad norm at step {step}"
            )

        optimizer.step()

        torch.cuda.synchronize()

        dt = time.perf_counter() - start

        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"step {step:02d} | "
            f"loss {loss.detach().item():.4f} | "
            f"grad {grad_norm.detach().item():.2f} | "
            f"{dt:.2f}s | "
            f"VRAM {allocated:.2f}G | "
            f"peak {peak:.2f}G"
        )

    print()
    print("BF16 forward:  PASS")
    print("BF16 backward: PASS")
    print("finite grads:  PASS")
    print("optimizer:     PASS")
    print()
    print("V5 BF16 STABILITY TEST: PASS")


if __name__ == "__main__":
    main()
