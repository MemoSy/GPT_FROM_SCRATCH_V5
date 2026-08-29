from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class BinaryMixStream:
    def __init__(
        self,
        data_dir: str,
        block_size: int,
        seed: int = 1337,
    ):
        self.data_dir = Path(data_dir)
        self.block_size = int(block_size)
        self.seed = int(seed)

        self.paths = sorted(
            self.data_dir.glob("*.bin")
        )

        if not self.paths:
            raise RuntimeError(
                f"No .bin files found in {self.data_dir}"
            )

        self.arrays = [
            np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
            )
            for path in self.paths
        ]

        pairs = []

        for source_idx, arr in enumerate(self.arrays):
            blocks = len(arr) // self.block_size

            for block_idx in range(blocks):
                pairs.append(
                    (source_idx, block_idx)
                )

        self.schedule = np.asarray(
            pairs,
            dtype=np.int32,
        )

        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.schedule)

        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.schedule):
            raise StopIteration

        source_idx, block_idx = self.schedule[
            self.position
        ]

        self.position += 1

        start = int(block_idx) * self.block_size
        end = start + self.block_size

        # Copy to int64 because torch embedding indices use long.
        block = np.array(
            self.arrays[int(source_idx)][start:end],
            dtype=np.int64,
            copy=True,
        )

        return torch.from_numpy(block)

    @property
    def total_blocks(self):
        return len(self.schedule)

    @property
    def total_tokens(self):
        return (
            self.total_blocks
            * self.block_size
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "binary_mix_v1",
            "data_dir": str(self.data_dir),
            "block_size": self.block_size,
            "seed": self.seed,
            "position": self.position,
            "files": [
                p.name for p in self.paths
            ],
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state["type"] != "binary_mix_v1":
            raise ValueError("stream type mismatch")

        if int(state["block_size"]) != self.block_size:
            raise ValueError("block_size mismatch")

        if int(state["seed"]) != self.seed:
            raise ValueError("seed mismatch")

        current_files = [
            p.name for p in self.paths
        ]

        if state["files"] != current_files:
            raise ValueError("binary file list mismatch")

        position = int(state["position"])

        if not 0 <= position <= len(self.schedule):
            raise ValueError("invalid stream position")

        self.position = position
