import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.model import V5Config, V5Model


torch.manual_seed(1337)

config = V5Config()
model = V5Model(config)
model.eval()

ids = torch.tensor([
    [504, 7706, 6572, 260, 1573, 10091, 6644, 30]
])

with torch.no_grad():
    logits, model_loss = model(ids, ids)

manual_loss = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
    ids[:, 1:].contiguous().view(-1),
)

difference = abs(model_loss.item() - manual_loss.item())

print("sequence length:", ids.size(1))
print("next-token predictions:", ids.size(1) - 1)
print("model loss:", model_loss.item())
print("manual shifted loss:", manual_loss.item())
print("difference:", difference)

assert difference < 1e-6

print()
print("CAUSAL NEXT-TOKEN LOSS: PASS")
