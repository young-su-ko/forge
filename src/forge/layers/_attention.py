import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from forge.layers._rotary_embedding import RotaryEmbedding


class Gate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        return torch.tanh(self.alpha) * x


class FlagAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.x_qkv = nn.Linear(dim, dim * 3)
        self.c_kv = nn.Linear(dim, dim * 2)

        self.x_q_norm = nn.LayerNorm(dim)
        self.x_k_norm = nn.LayerNorm(dim)
        self.c_k_norm = nn.LayerNorm(dim)

        self.rotary = RotaryEmbedding(self.head_dim)
        self.gate = Gate(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_qkv = self.x_qkv(x)
        c_kv = self.c_kv(c)

        x_q, x_k, x_v = x_qkv.chunk(3, dim=-1)
        c_k, c_v = c_kv.chunk(2, dim=-1)

        x_q = self.x_q_norm(x_q)
        x_k = self.x_k_norm(x_k)
        c_k = self.c_k_norm(c_k)

        x_q = rearrange(x_q, "b s (h d) -> b h s d", h=self.num_heads)
        x_k = rearrange(x_k, "b s (h d) -> b h s d", h=self.num_heads)
        x_v = rearrange(x_v, "b s (h d) -> b h s d", h=self.num_heads)

        c_k = rearrange(c_k, "b s (h d) -> b h s d", h=self.num_heads)
        c_v = rearrange(c_v, "b s (h d) -> b h s d", h=self.num_heads)

        x_q = self.rotary(x_q)
        x_k = self.rotary(x_k)

        x_attn = F.scaled_dot_product_attention(x_q, x_k, x_v, is_causal=False)
        c_attn = F.scaled_dot_product_attention(x_q, c_k, c_v, is_causal=False)

        x_attn = rearrange(x_attn, "b h s d -> b s (h d)")
        c_attn = rearrange(c_attn, "b h s d -> b s (h d)")

        attn_out = x_attn + self.gate(c_attn)
        output = self.out_proj(attn_out)

        return output
