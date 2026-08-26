from __future__ import annotations

import math

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    fused: bool = False,
):
    decay = []
    no_decay = []

    seen = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # Tied embeddings can otherwise appear twice.
        if id(parameter) in seen:
            continue

        seen.add(id(parameter))

        # Matrix weights use WD.
        # Embedding and normalization parameters do not.
        if (
            parameter.ndim >= 2
            and "token_embedding" not in name
        ):
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    kwargs = dict(
        lr=learning_rate,
        betas=(beta1, beta2),
    )

    if fused:
        kwargs["fused"] = True

    try:
        optimizer = torch.optim.AdamW(
            groups,
            **kwargs,
        )
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)

        optimizer = torch.optim.AdamW(
            groups,
            **kwargs,
        )

    return optimizer


def build_cosine_scheduler(
    optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
):
    warmup_steps = max(
        1,
        warmup_steps,
    )

    max_steps = max(
        warmup_steps + 1,
        max_steps,
    )

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (
                float(step + 1)
                / float(warmup_steps)
            )

        progress = (
            float(step - warmup_steps)
            / float(max_steps - warmup_steps)
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio)
            * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )
