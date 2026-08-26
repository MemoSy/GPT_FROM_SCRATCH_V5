vocab = 49152
hidden = 576
intermediate = 1536
layers = 30
heads = 9
kv_heads = 3
head_dim = 64

embedding = vocab * hidden

q = hidden * (heads * head_dim)
k = hidden * (kv_heads * head_dim)
v = hidden * (kv_heads * head_dim)
o = hidden * hidden

attention = q + k + v + o

swiglu = (
    hidden * intermediate +
    hidden * intermediate +
    intermediate * hidden
)

norms_per_layer = 2 * hidden

per_layer = attention + swiglu + norms_per_layer

transformer = layers * per_layer

final_norm = hidden

total = embedding + transformer + final_norm

print(f"Embedding:       {embedding:,}")
print(f"Attention/layer: {attention:,}")
print(f"SwiGLU/layer:    {swiglu:,}")
print(f"Norms/layer:     {norms_per_layer:,}")
print(f"Block total:     {per_layer:,}")
print(f"30 blocks:       {transformer:,}")
print(f"Final RMSNorm:   {final_norm:,}")
print("-" * 35)
print(f"TOTAL:           {total:,}")
print(f"TOTAL millions:  {total / 1e6:.3f}M")

assert total == 134_515_008
print("\nPARAMETER BUDGET: PASS")
