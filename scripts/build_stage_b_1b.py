import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


TARGET_TOKENS = 1_000_243_200

TARGETS = {
    "dclm_edu":        531_600_578,
    "fineweb_edu":     354_400_383,
    "stack_edu":        62_359_301,
    "infimm_webmath":   19_954_976,
    "finemath":         16_961_730,
    "cosmopedia_v2":    14_966_232,
}

assert sum(TARGETS.values()) == TARGET_TOKENS

DATASET = "EleutherAI/SmolLM2-135M-10B"

OUT = Path("data/stage_b_1b")
OUT.mkdir(parents=True, exist_ok=True)

STATE_PATH = OUT / "build_state.json"
META_PATH = OUT / "metadata.json"

SEED = 1337
BUFFER_SIZE = 1000
SAVE_EVERY_ROWS = 5000


tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer",
    local_files_only=True,
)

tokenizer.model_max_length = 10**9
eos = tokenizer.eos_token_id


# ------------------------------------------------------------
# Resume state
# ------------------------------------------------------------

if STATE_PATH.exists():
    state = json.loads(STATE_PATH.read_text())

    rows_seen = int(state["rows_seen"])
    written = {
        k: int(v)
        for k, v in state["written"].items()
    }
    docs = {
        k: int(v)
        for k, v in state["docs"].items()
    }

    print("RESUMING BUILD")
    print("rows already processed:", f"{rows_seen:,}")
    print("tokens already written:", f"{sum(written.values()):,}")

else:
    rows_seen = 0
    written = {k: 0 for k in TARGETS}
    docs = {k: 0 for k in TARGETS}


# ------------------------------------------------------------
# Make files consistent with saved state
# ------------------------------------------------------------

for source in TARGETS:
    path = OUT / f"{source}.bin"
    expected_bytes = written[source] * 2

    if path.exists():
        actual = path.stat().st_size

        if actual != expected_bytes:
            print(
                f"Repairing {source}: "
                f"{actual:,} -> {expected_bytes:,} bytes"
            )

            with open(path, "r+b") as f:
                f.truncate(expected_bytes)

    elif expected_bytes != 0:
        raise RuntimeError(
            f"Missing binary file for {source}"
        )


files = {
    source: open(
        OUT / f"{source}.bin",
        "ab",
        buffering=1024 * 1024,
    )
    for source in TARGETS
}


def save_state():
    for f in files.values():
        f.flush()
        os.fsync(f.fileno())

    temp = STATE_PATH.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            {
                "rows_seen": rows_seen,
                "written": written,
                "docs": docs,
            },
            indent=2,
        )
    )

    os.replace(temp, STATE_PATH)


# ------------------------------------------------------------
# Streaming dataset
# ------------------------------------------------------------

print("\nFINAL STAGE-B TARGETS:")

for source, amount in TARGETS.items():
    print(
        f"{source:20} "
        f"{amount:>12,} "
        f"({amount / TARGET_TOKENS:6.2%})"
    )


ds = load_dataset(
    DATASET,
    split="train",
    streaming=True,
)

ds = ds.shuffle(
    seed=SEED,
    buffer_size=BUFFER_SIZE,
)

# Deterministic resume.
if rows_seen:
    print("\nSkipping previously processed rows...")
    ds = ds.skip(rows_seen)


bar = tqdm(
    total=TARGET_TOKENS,
    initial=sum(written.values()),
    desc="Building Stage B",
    unit="tok",
    unit_scale=True,
)


try:
    for row in ds:

        rows_seen += 1

        source = row.get("source")

        if source not in TARGETS:
            continue

        remaining = (
            TARGETS[source]
            - written[source]
        )

        if remaining <= 0:
            if all(
                written[s] >= TARGETS[s]
                for s in TARGETS
            ):
                break

            continue

        text = row.get("text", "")

        if not text or not text.strip():
            continue

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        ).input_ids

        if not ids:
            continue

        ids.append(eos)

        take = min(
            len(ids),
            remaining,
        )

        arr = np.asarray(
            ids[:take],
            dtype=np.uint16,
        )

        if arr.size and int(arr.max()) >= 65536:
            raise RuntimeError(
                "Token ID exceeds uint16."
            )

        arr.tofile(files[source])

        written[source] += take
        docs[source] += 1

        bar.update(take)

        if rows_seen % SAVE_EVERY_ROWS == 0:
            save_state()

            bar.set_postfix(
                rows=f"{rows_seen:,}"
            )

        if all(
            written[s] >= TARGETS[s]
            for s in TARGETS
        ):
            break


except KeyboardInterrupt:
    print("\nInterrupted — saving resume state...")
    save_state()

    for f in files.values():
        f.close()

    raise


# ------------------------------------------------------------
# Finalize
# ------------------------------------------------------------

bar.close()

save_state()

for f in files.values():
    f.close()


total = sum(written.values())

print("\nRESULT:")

for source in TARGETS:
    status = (
        "OK"
        if written[source] == TARGETS[source]
        else "INCOMPLETE"
    )

    print(
        f"{source:20} "
        f"{written[source]:>12,}/"
        f"{TARGETS[source]:>12,} "
        f"{status}"
    )


if total != TARGET_TOKENS:
    raise RuntimeError(
        f"Build incomplete: "
        f"{total:,}/{TARGET_TOKENS:,}"
    )


metadata = {
    "dataset": DATASET,
    "recipe": "Stage B / Pilot A validated mixture",
    "target_tokens": TARGET_TOKENS,
    "dtype": "uint16",
    "shuffle_seed": SEED,
    "rows_seen": rows_seen,
    "sources": {
        source: {
            "tokens": written[source],
            "documents": docs[source],
            "bytes": (
                OUT / f"{source}.bin"
            ).stat().st_size,
        }
        for source in TARGETS
    },
}


META_PATH.write_text(
    json.dumps(metadata, indent=2)
)


print()
print("TOTAL TOKENS:", f"{total:,}")
print("OUTPUT:", OUT)
print("STAGE B 1B BUILD: PASS")

# Avoid the HF streaming shutdown hang we saw twice.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
