import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.model import (
    V5Config,
    GQAAttention,
    RotaryEmbedding,
)


def test_gqa(device):
    config = V5Config()
    attn = GQAAttention(config).to(device)
    attn.eval()

    assert attn.num_heads == 9
    assert attn.num_kv_heads == 3
    assert attn.num_kv_groups == 3

    assert attn.q_proj.weight.shape == (576, 576)
    assert attn.k_proj.weight.shape == (192, 576)
    assert attn.v_proj.weight.shape == (192, 576)
    assert attn.o_proj.weight.shape == (576, 576)

    x = torch.randn(2, 32, 576, device=device)

    with torch.no_grad():
        y = attn(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    print("GQA shapes: PASS")


def test_rope(device):
    rope = RotaryEmbedding(
        head_dim=64,
        max_seq_len=128,
        base=10000.0,
    ).to(device)

    q = torch.randn(2, 9, 32, 64, device=device)
    k = torch.randn(2, 3, 32, 64, device=device)

    q_norm_before = q.float().pow(2).sum(dim=-1)
    k_norm_before = k.float().pow(2).sum(dim=-1)

    q2, k2 = rope(q, k)

    q_norm_after = q2.float().pow(2).sum(dim=-1)
    k_norm_after = k2.float().pow(2).sum(dim=-1)

    q_diff = (q_norm_before - q_norm_after).abs().max().item()
    k_diff = (k_norm_before - k_norm_after).abs().max().item()

    print(f"RoPE Q norm max diff: {q_diff:.8f}")
    print(f"RoPE K norm max diff: {k_diff:.8f}")

    assert torch.allclose(
        q_norm_before,
        q_norm_after,
        atol=1e-4,
        rtol=1e-4,
    )

    assert torch.allclose(
        k_norm_before,
        k_norm_after,
        atol=1e-4,
        rtol=1e-4,
    )

    print("RoPE rotation: PASS")


def test_causality(device):
    torch.manual_seed(1337)

    config = V5Config()
    attn = GQAAttention(config).to(device)
    attn.eval()

    T = 24
    pivot = 10

    x1 = torch.randn(
        1,
        T,
        config.hidden_size,
        device=device,
    )

    x2 = x1.clone()

    # Completely change everything AFTER the pivot.
    x2[:, pivot + 1:] = torch.randn_like(
        x2[:, pivot + 1:]
    ) * 10.0

    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)

    # Positions 0..pivot must be completely unaffected
    # by tokens in the future.
    prefix_diff = (
        y1[:, :pivot + 1]
        - y2[:, :pivot + 1]
    ).abs().max().item()

    # Later positions SHOULD usually change.
    future_diff = (
        y1[:, pivot + 1:]
        - y2[:, pivot + 1:]
    ).abs().max().item()

    print(f"Causal prefix max diff: {prefix_diff:.10f}")
    print(f"Future max diff:        {future_diff:.10f}")

    assert prefix_diff < 1e-5
    assert future_diff > 1e-5

    print("Causal attention: PASS")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_gqa(device)
    test_rope(device)
    test_causality(device)

    print()
    print("V5 ATTENTION TESTS: PASS")


if __name__ == "__main__":
    main()
