import torch
import pytest

from forge.inference.flow_simulator import ValFlowSimulator

class DummyVelocityModel(torch.nn.Module):
    def forward(self, x, t, condition):
        return condition - 0.1 * x

@pytest.mark.parametrize("batch_size, seq_len, dim", [(4, 50, 1280)])
@pytest.mark.parametrize("guidance_scale", [0.0, 1.0])
def test_valflowsimulator_smoketest(batch_size, seq_len, dim, guidance_scale):
    """Ensure ValFlowSimulator runs end-to-end with dummy velocity model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy condition tensor
    condition = torch.randn(batch_size, seq_len, dim, device=device)

    # Instantiate dummy velocity model and simulator
    model = DummyVelocityModel().to(device)
    simulator = ValFlowSimulator(
        velocity_model=model,
        guidance_scale=guidance_scale,
        t_steps=10,  # small number of steps for speed
    )

    # Run sampling
    with torch.no_grad():
        xt = simulator.sample(condition)

    # --- Assertions ---
    assert isinstance(xt, torch.Tensor)
    assert xt.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(xt).any(), "Output contains NaNs"
    assert torch.isfinite(xt).all(), "Output contains infs"