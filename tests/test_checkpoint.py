import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.model import V5Config, V5Model


class FakeStream:
    def __init__(self):
        self.position = 0
        self.buffer = [1, 2, 3]

    def state_dict(self):
        return {
            "position": self.position,
            "buffer": self.buffer.copy(),
        }

    def load_state_dict(self, state):
        self.position = state["position"]
        self.buffer = list(state["buffer"])


def build():
    config = V5Config(
        vocab_size=512,
        block_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )

    model = V5Model(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0,
    )

    return (
        config,
        model,
        optimizer,
        scheduler,
    )


def main():
    device = torch.device("cpu")

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    (
        config1,
        model1,
        optimizer1,
        scheduler1,
    ) = build()

    stream1 = FakeStream()
    stream1.position = 123
    stream1.buffer = [9, 8, 7, 6]

    x = torch.randint(
        0,
        config1.vocab_size,
        (2, 16),
    )

    _, loss = model1(x, x)

    loss.backward()
    optimizer1.step()
    scheduler1.step()

    optimizer1.zero_grad(
        set_to_none=True
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"

        save_checkpoint(
            path,
            model=model1,
            optimizer=optimizer1,
            scheduler=scheduler1,
            stream=stream1,
            step=42,
            tokens_seen=123456,
            config=config1,
        )

        # RNG values that MUST occur after resume.
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(5)

        (
            config2,
            model2,
            optimizer2,
            scheduler2,
        ) = build()

        stream2 = FakeStream()

        info = load_checkpoint(
            path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            stream=stream2,
            device=device,
        )

        resumed_python = random.random()
        resumed_numpy = np.random.random()
        resumed_torch = torch.rand(5)

        # Model equality.
        for p1, p2 in zip(
            model1.parameters(),
            model2.parameters(),
        ):
            assert torch.equal(p1, p2)

        assert info["step"] == 42
        assert info["tokens_seen"] == 123456

        assert stream2.position == 123
        assert stream2.buffer == [9, 8, 7, 6]

        assert expected_python == resumed_python
        assert expected_numpy == resumed_numpy
        assert torch.equal(
            expected_torch,
            resumed_torch,
        )

        assert (
            scheduler1.state_dict()
            == scheduler2.state_dict()
        )

        print("model state:     PASS")
        print("optimizer state: PASS")
        print("scheduler state: PASS")
        print("stream state:    PASS")
        print("Python RNG:      PASS")
        print("NumPy RNG:       PASS")
        print("PyTorch RNG:     PASS")
        print("step restore:    PASS")
        print("tokens restore:  PASS")
        print()
        print("V5 CHECKPOINT ROUNDTRIP: PASS")


if __name__ == "__main__":
    main()
