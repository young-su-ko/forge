import torch
import pytest
from forge.inference.fid import FIDCalculator, frechet_distance

@pytest.mark.parametrize("dim", [1280])
def test_fidcalculator(dim):
    """
    Smoke test that loads a real FID reference .pt file
    and computes the Fréchet distance.
    """
    ref_path = "/new-stg/home/young/forge/data/10k_ref.pt"
    fid_calc = FIDCalculator(reference_path=ref_path)

    sample_mu = torch.randn(dim)
    sample_sigma = torch.eye(dim) * 1.1  # scaled covariance

    fid_value = fid_calc.compute(sample_mu, sample_sigma)

    assert isinstance(fid_value, torch.Tensor)
    assert fid_value.numel() == 1, "FID value should be scalar"
    assert torch.isfinite(fid_value).all(), "FID contains NaN or Inf"
    assert fid_value >= 0, "FID should be non-negative"


def test_frechet_distance():
    dim = 10
    mu_x = torch.zeros(dim)
    sigma_x = torch.eye(dim)

    fid_value = frechet_distance(mu_x, sigma_x, mu_x, sigma_x)

    assert fid_value == 0
