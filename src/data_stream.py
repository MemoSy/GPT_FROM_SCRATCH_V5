from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class FineWebPackedStream:
    def __init__(
        self,
        *,
        tokenizer_path: str = "tokenizer",
        block_size: int = 2048,
        min_chars: int = 300,
        max_chars: int = 50_000,
        min_score: float = 3.0,
        min_int_score: int = 3,
        shuffle_buffer: int = 0,
        seed: int = 1337,
    ):
        self.block_size = block_size
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_score = min_score
        self.min_int_score = min_int_score
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )

        if len(self.tokenizer) != 49152:
            raise ValueError(
                f"Expected vocab 49152, got {len(self.tokenizer)}"
            )

        if self.tokenizer.eos_token_id != 0:
            raise ValueError(
                f"Expected EOS 0, got {self.tokenizer.eos_token_id}"
            )

        self.eos_id = self.tokenizer.eos_token_id

        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        if shuffle_buffer > 0:
            self.dataset = self.dataset.shuffle(
                seed=seed,
                buffer_size=shuffle_buffer,
            )

        self.iterator = iter(self.dataset)

        self.buffer: list[int] = []

        self.accepted_docs = 0
        self.rejected_docs = 0
        self.blocks_yielded = 0
        self.raw_examples_seen = 0

    def _accept_text(self, example: dict[str, Any]) -> str | None:
        self.raw_examples_seen += 1

        text = example.get("text")

        if not isinstance(text, str):
            self.rejected_docs += 1
            return None

        if len(text) < self.min_chars:
            self.rejected_docs += 1
            return None

        score = example.get("score")
        int_score = example.get("int_score")

        if score is not None:
            if float(score) < self.min_score:
                self.rejected_docs += 1
                return None

        if int_score is not None:
            if int(int_score) < self.min_int_score:
                self.rejected_docs += 1
                return None

        return text[: self.max_chars]

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        while len(self.buffer) < self.block_size:
            example = next(self.iterator)

            text = self._accept_text(example)

            if text is None:
                continue

            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            if not ids:
                self.rejected_docs += 1
                continue

            self.buffer.extend(ids)
            self.buffer.append(self.eos_id)

            self.accepted_docs += 1

        block = self.buffer[: self.block_size]
        del self.buffer[: self.block_size]

        self.blocks_yielded += 1

        tensor = torch.tensor(
            block,
            dtype=torch.long,
        )

        if tensor.numel() != self.block_size:
            raise RuntimeError("Incorrect packed block size.")

        return tensor

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.state_dict(),
            "buffer": self.buffer.copy(),
            "accepted_docs": self.accepted_docs,
            "rejected_docs": self.rejected_docs,
            "blocks_yielded": self.blocks_yielded,
            "raw_examples_seen": self.raw_examples_seen,
            "block_size": self.block_size,
            "seed": self.seed,
            "shuffle_buffer": self.shuffle_buffer,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["block_size"] != self.block_size:
            raise ValueError("block_size mismatch")

        if state["seed"] != self.seed:
            raise ValueError("seed mismatch")

        if state["shuffle_buffer"] != self.shuffle_buffer:
            raise ValueError("shuffle_buffer mismatch")

        self.dataset.load_state_dict(state["dataset"])

        # Recreate iterator AFTER restoring dataset state.
        self.iterator = iter(self.dataset)

        self.buffer = list(state["buffer"])
        self.accepted_docs = int(state["accepted_docs"])
        self.rejected_docs = int(state["rejected_docs"])
        self.blocks_yielded = int(state["blocks_yielded"])
        self.raw_examples_seen = int(state["raw_examples_seen"])
