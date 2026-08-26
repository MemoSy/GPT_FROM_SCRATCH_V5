from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class V5Config:
    vocab_size: int = 49152
    block_size: int = 2048

    hidden_size: int = 576
    intermediate_size: int = 1536

    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)

        return (x_norm * self.weight.float()).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat(
        (-x2, x1),
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE head_dim must be even."
            )

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        # Llama convention:
        # [freq0 ... freq31, freq0 ... freq31]
        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        )

        self.register_buffer(
            "cos",
            emb.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            emb.sin(),
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ):
        seq_len = q.shape[-2]

        cos = self.cos[:seq_len].to(
            device=q.device,
            dtype=q.dtype,
        )

        sin = self.sin[:seq_len].to(
            device=q.device,
            dtype=q.dtype,
        )

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = (
            q * cos
            + rotate_half(q) * sin
        )

        k = (
            k * cos
            + rotate_half(k) * sin
        )

        return q, k
    

def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, config: V5Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(
            self.hidden_size,
            q_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            kv_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            kv_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            q_dim,
            self.hidden_size,
            bias=False,
        )

        self.attention_dropout = config.attention_dropout

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.block_size,
            base=config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        # GQA:
        # 9 Q heads share 3 K/V heads.
        #
        # We explicitly repeat K/V for compatibility across
        # CUDA and ROCm SDPA implementations.
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )
            v = v.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )

        dropout_p = (
            self.attention_dropout
            if self.training
            else 0.0
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.hidden_size)

        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: V5Config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: V5Config):
        super().__init__()

        self.input_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.attention = GQAAttention(config)

        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x)
        )

        x = x + self.mlp(
            self.post_attention_norm(x)
        )

        return x


class V5Model(nn.Module):
    def __init__(self, config: V5Config):
        super().__init__()

        self.config = config
        self.gradient_checkpointing = False

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):

        B, T = input_ids.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block_size {self.config.block_size}"
            )

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            # Causal language modeling:
            # token at position t predicts token at position t+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
    )
