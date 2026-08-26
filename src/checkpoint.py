from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    torch.compile wraps the original module.
    Always save/load the real underlying model.
    """
    if hasattr(model, "_orig_mod"):
        return model._orig_mod

    return model


def get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Makes checkpoints portable between CPU, ROCm and CUDA.
    """
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    stream: Any,
    step: int,
    tokens_seen: int,
    config: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_model = unwrap_model(model)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,

        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),

        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "stream": stream.state_dict(),

        "step": int(step),
        "tokens_seen": int(tokens_seen),

        "rng": get_rng_state(),

        "config": (
            asdict(config)
            if hasattr(config, "__dataclass_fields__")
            else config
        ),
    }

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temp_path,
    )

    # Atomic replacement.
    os.replace(
        temp_path,
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    stream: Any,
    device: torch.device,
) -> dict[str, int]:

    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    version = checkpoint.get(
        "checkpoint_version"
    )

    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version: {version}"
        )

    raw_model = unwrap_model(model)

    raw_model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    move_optimizer_to_device(
        optimizer,
        device,
    )

    if scheduler is not None:
        scheduler_state = checkpoint.get(
            "scheduler"
        )

        if scheduler_state is not None:
            scheduler.load_state_dict(
                scheduler_state
            )

    stream.load_state_dict(
        checkpoint["stream"]
    )

    set_rng_state(
        checkpoint["rng"]
    )

    return {
        "step": int(checkpoint["step"]),
        "tokens_seen": int(
            checkpoint["tokens_seen"]
        ),
    }
