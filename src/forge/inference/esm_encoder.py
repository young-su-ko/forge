import torch

from esm.pretrained import esm2_t33_650M_UR50D


class ESMEncoder:
    def __init__(self, device):
        self.device = device
        self.model, self.alphabet = esm2_t33_650M_UR50D()
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def encode(self, sequence: str | list[str]) -> torch.Tensor:
        if isinstance(sequence, str):
            sequence = [sequence]
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequence)]
        _, _, tokens = self.batch_converter(data)
        output = self.model(
            tokens.to(self.device), repr_layers=[33], return_contacts=False
        )
        return output["representations"][33][:, 1:-1, :]
