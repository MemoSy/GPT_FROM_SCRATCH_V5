import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.model import (
    V5Config,
    V5Model,
    count_parameters,
)


EXPECTED_PARAMS = 134_515_008


def main():
    config = V5Config()

    model = V5Model(config)

    params = count_parameters(model)

    print(f"parameters: {params:,}")

    assert params == EXPECTED_PARAMS, (
        f"Expected {EXPECTED_PARAMS:,}, "
        f"got {params:,}"
    )

    assert (
        model.lm_head.weight.data_ptr()
        == model.token_embedding.weight.data_ptr()
    )

    print("tied embeddings: PASS")

    B = 2
    T = 32

    x = torch.randint(
        0,
        config.vocab_size,
        (B, T),
    )

    labels = torch.randint(
        0,
        config.vocab_size,
        (B, T),
    )

    logits, loss = model(
        x,
        labels,
    )

    print("input shape: ", tuple(x.shape))
    print("logits shape:", tuple(logits.shape))
    print("loss:", float(loss.detach()))

    assert logits.shape == (
        B,
        T,
        config.vocab_size,
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()

    nonfinite_grads = 0

    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nonfinite_grads += 1

    print(
        "non-finite gradient tensors:",
        nonfinite_grads,
    )

    assert nonfinite_grads == 0

    print()
    print("V5 MODEL TEST: PASS")


if __name__ == "__main__":
    main()
