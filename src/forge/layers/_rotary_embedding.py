import torch
import torch.nn as nn


# https://github.com/facebookresearch/esm/blob/main/esm/rotary_embedding.py
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [batch_size, seq_len, embed_dim]
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # This gives us the list of thetas, in groups of 2
        inv_freq = 1.0 / (torch.pow(10000, torch.arange(0, dim, 2) / dim))
        # len (inv_freq) = self.dim // 2
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        if self._seq_len_cached is None or seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)

            # to get len to be equal to the dim, we repeat the freqs twice
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # [seq_len, dim]
            self._cos_cached = emb.cos().unsqueeze(0)  # [1, seq_len, dim]
            self._sin_cached = emb.sin().unsqueeze(0)  # [1, seq_len, dim]
        return self._cos_cached, self._sin_cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self._update_cos_sin_cache(x)
        return apply_rotary_pos_emb(x, cos, sin)
