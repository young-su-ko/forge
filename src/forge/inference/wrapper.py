import torch
import json
import os

from raygun.pretrained import raygun_8_8mil_800M
from forge.models._dit import FlagDiT
from forge.inference.esm_encoder import ESMEncoder
from forge.inference.flow_simulator import FlowSimulator


class InferenceWrapper:
    def __init__(
        self, velocity_model, guidance_scale: float, t_steps: int, device: str
    ):
        self.device = device
        self.velocity_model = velocity_model.eval().to(device)
        self.esm_encoder = ESMEncoder(self.device)
        self.raygun = raygun_8_8mil_800M().eval().to(self.device)
        self.flow_simulator = FlowSimulator(
            self.velocity_model,
            self.raygun,
            guidance_scale,
            t_steps,
            self.device,
        )

    @torch.no_grad()
    def generate_binder(
        self, target_sequence: str, binder_length: int, n_samples: int
    ) -> str:
        target_esm_embedding = self.esm_encoder.encode(target_sequence)
        return self.flow_simulator.sample(
            target_esm_embedding, binder_length, n_samples
        )

    @torch.no_grad()
    def generate_unconditionally(self, output_length: int, n_samples: int) -> str:
        return self.flow_simulator.sample(None, output_length, n_samples)

    @classmethod
    def from_pretrained(
        cls,
        repo_or_dir: str,
        guidance_scale: float,
        t_steps: int,
        device: str,
        map_location="cpu",
    ):
        """
        Load from either a local directory or a HF Hub repo.
        Local directory must contain `config.json` and `pytorch_model.bin`.
        """
        if os.path.isdir(repo_or_dir):
            config_path = os.path.join(repo_or_dir, "config.json")
            ckpt_path = os.path.join(repo_or_dir, "pytorch_model.bin")
        else:  # fallback to HF Hub
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_or_dir, "config.json")
            ckpt_path = hf_hub_download(repo_or_dir, "pytorch_model.bin")

        with open(config_path) as f:
            model_cfg = json.load(f)

        velocity_model = FlagDiT(**model_cfg)
        state_dict = torch.load(ckpt_path, map_location=map_location)
        velocity_model.load_state_dict(state_dict)

        return cls(velocity_model, guidance_scale, t_steps, device)