from transformers import AutoTokenizer

NAME = "HuggingFaceTB/SmolLM2-135M"

tokenizer = AutoTokenizer.from_pretrained(NAME)

print("tokenizer:", NAME)
print("vocab size:", len(tokenizer))
print("EOS token:", tokenizer.eos_token)
print("EOS id:", tokenizer.eos_token_id)
print("BOS token:", tokenizer.bos_token)
print("BOS id:", tokenizer.bos_token_id)

texts = [
    "The young boy walked into the forest.",
    "Why did John leave the city?",
    "She looked at him and said, \"Don't go.\"",
]

for text in texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)

    print()
    print("TEXT:", text)
    print("TOKENS:", len(ids))
    print("IDS:", ids)
    print("DECODED:", decoded)

assert len(tokenizer) <= 49152

print()
print("V5 TOKENIZER TEST: PASS")
