import torch


class SequenceDecoder:
    def __init__(self, raygun):
        super().__init__()
        self.raygun = raygun

    @torch.no_grad()
    def decode(self, raygun_embedding: torch.Tensor, lengths: list[int]) -> list[str]:
        lengths = torch.tensor(
            lengths, dtype=torch.long, device=raygun_embedding.device
        )
        return self.raygun.get_sequences_from_fixed(raygun_embedding, lengths)
