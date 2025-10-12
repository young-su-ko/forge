import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from forge.layers._rotary_embedding import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE
        q = self.rotary(q)
        k = self.rotary(k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=mask, is_causal=False
        )  # [batch, num_heads, seq_len, head_dim]

        # Merge heads
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        output = self.out_proj(attn_out)
        return output
