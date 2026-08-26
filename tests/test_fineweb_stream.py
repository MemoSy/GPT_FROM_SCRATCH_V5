from datasets import load_dataset
from transformers import AutoTokenizer


DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"

MIN_CHARS = 300
MAX_CHARS = 50_000
MIN_SCORE = 3.0
MIN_INT_SCORE = 3
BLOCK_SIZE = 2048

tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer",
    local_files_only=True,
)

eos_id = tokenizer.eos_token_id

print("Loading FineWeb-Edu stream...")

dataset = load_dataset(
    DATASET,
    name=CONFIG,
    split="train",
    streaming=True,
)

buffer = []
accepted_docs = 0
rejected_docs = 0
blocks = []

for example in dataset:
    text = example.get("text")

    if not isinstance(text, str):
        rejected_docs += 1
        continue

    if len(text) < MIN_CHARS:
        rejected_docs += 1
        continue

    score = example.get("score")
    int_score = example.get("int_score")

    if score is not None and float(score) < MIN_SCORE:
        rejected_docs += 1
        continue

    if int_score is not None and int(int_score) < MIN_INT_SCORE:
        rejected_docs += 1
        continue

    text = text[:MAX_CHARS]

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not ids:
        rejected_docs += 1
        continue

    buffer.extend(ids)

    # Document boundary.
    buffer.append(eos_id)

    accepted_docs += 1

    while len(buffer) >= BLOCK_SIZE:
        block = buffer[:BLOCK_SIZE]
        del buffer[:BLOCK_SIZE]

        assert len(block) == BLOCK_SIZE
        assert min(block) >= 0
        assert max(block) < len(tokenizer)

        blocks.append(block)

        if len(blocks) >= 3:
            break

    if len(blocks) >= 3:
        break


print()
print("accepted docs:", accepted_docs)
print("rejected docs:", rejected_docs)
print("blocks produced:", len(blocks))

for i, block in enumerate(blocks):
    print()
    print(f"BLOCK {i}")
    print("tokens:", len(block))
    print("EOS count:", block.count(eos_id))

    preview = tokenizer.decode(block[:100])
    print("preview:")
    print(repr(preview[:500]))

assert len(blocks) == 3
assert all(len(block) == BLOCK_SIZE for block in blocks)

print()
print("FINEWEB STREAMING + FILTER + PACKING: PASS")
