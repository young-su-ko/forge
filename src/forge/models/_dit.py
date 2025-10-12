import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
import json

from forge.layers._dit_block import modulate, FlagDiTBlock
from forge.layers._time_embedder import TimestepEmbedder


class FinalLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, input_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FlagDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mlp_ratio: float,
        num_heads: int,
        num_layers: int,
        conditioning_dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        self.conditioning_dropout = conditioning_dropout
        self.timestep_embedder = TimestepEmbedder(input_dim)

        self.blocks = nn.ModuleList(
            [FlagDiTBlock(input_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(input_dim, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, L, raygun_dim)
            t: (bs, )
            c: (bs, L, raygun_dim)
        Returns:
            u_t^theta(x|c): (bs, L, raygun_dim)
        """
        if self.training and self.conditioning_dropout > 0:
            mask = (
                torch.rand(c.size(0), 1, 1, device=c.device) > self.conditioning_dropout
            ).float()
            c = c * mask

        c_pooled = torch.mean(c, dim=1)  # (bs, dim)
        c_pooled = self.timestep_embedder(t) + c_pooled

        for block in self.blocks:
            x = block(x, c, c_pooled)

        out = self.final_layer(x, c_pooled)

        return out

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str = "main", map_location="cpu"):
        """
        Load a DiT model from a Hugging Face Hub repo.

        Args:
            repo_id (str): e.g. "yk0/forge-small"
            revision (str): branch, tag, or commit (default: "main")
            map_location: device for weights ("cpu" or "cuda")
        """
        config_path = hf_hub_download(repo_id, "config.json", revision=revision)
        with open(config_path) as f:
            cfg = json.load(f)

        model = cls(**cfg)

        weights_path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)

        model.eval()
        return model
