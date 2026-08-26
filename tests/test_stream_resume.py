import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data_stream import FineWebPackedStream


print("creating first stream...")

stream1 = FineWebPackedStream(
    block_size=2048,
    shuffle_buffer=0,
    seed=1337,
)

block0 = next(stream1)
block1 = next(stream1)

state = stream1.state_dict()

print("checkpoint saved")
print("blocks before checkpoint:", stream1.blocks_yielded)
print("buffer tokens saved:", len(state["buffer"]))
print("raw examples seen:", stream1.raw_examples_seen)

# This is what should come next.
expected = next(stream1)

print("expected next block collected")

del stream1


print()
print("creating fresh stream...")

stream2 = FineWebPackedStream(
    block_size=2048,
    shuffle_buffer=0,
    seed=1337,
)

stream2.load_state_dict(state)

resumed = next(stream2)

print("resumed block collected")

identical = torch.equal(
    expected,
    resumed,
)

difference = (
    expected != resumed
).sum().item()

print()
print("shape:", tuple(resumed.shape))
print("identical:", identical)
print("different token positions:", difference)
print("blocks after resume:", stream2.blocks_yielded)

assert block0.shape == (2048,)
assert block1.shape == (2048,)
assert resumed.shape == (2048,)
assert identical
assert difference == 0

print()
print("STATEFUL FINEWEB RESUME: PASS")
