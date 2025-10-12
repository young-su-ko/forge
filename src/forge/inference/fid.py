import torch
from torch import Tensor
from pathlib import Path

# From: https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/

def frechet_distance(mu_x: Tensor, sigma_x: Tensor, mu_y: Tensor, sigma_y: Tensor) -> Tensor:
    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)

    return a + b - 2 * c

class FIDCalculator:
    def __init__(self, reference_path: Path, device: torch.device):
        self.reference_stat_dict = torch.load(reference_path).to(device)
        self.reference_mu = self.reference_stat_dict["mu"]
        self.reference_sigma = self.reference_stat_dict["sigma"]
        self.device = device

    def __call__(self, sample_stats: dict[torch.Tensor]) -> Tensor:
        sample_mu = sample_stats["mu"]
        sample_sigma = sample_stats["sigma"]
        return frechet_distance(self.reference_mu, self.reference_sigma, sample_mu, sample_sigma)
    