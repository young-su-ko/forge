import torch
import pytest
from omegaconf import OmegaConf
from forge.lightning_modules.lit_flow import LitFlow

# Dummy stub classes to bypass FID/ValFlowSimulator during smoke test
class DummyFIDCalculator:
    def __init__(self, *args, **kwargs):
        self.reference_mu = torch.zeros(1280)
        self.reference_sigma = torch.eye(1280)
    def compute(self, mu, sigma):
        return torch.tensor(0.0)

class DummyValFlowSimulator:
    def __init__(self, *args, **kwargs): pass
    def sample(self, c):  # simulate flow output
        bs, L, D = c.shape
        return torch.randn(bs, 50, D, device=c.device)

@pytest.mark.parametrize("batch_size, seq_len, dim", [(2, 50, 1280)])
def test_litflow_forward_and_validation(monkeypatch, batch_size, seq_len, dim):
    # Patch external dependencies so LitFlow doesn't need actual files
    monkeypatch.setattr("forge.lightning_modules.lit_flow.FIDCalculator", DummyFIDCalculator)
    monkeypatch.setattr("forge.lightning_modules.lit_flow.ValFlowSimulator", DummyValFlowSimulator)

    config = OmegaConf.create(
        {
            "model": {
                "_target_": "forge.models._dit.FlagDiT",
                "input_dim": dim,
                "hidden_dim": 64,
                "mlp_ratio": 1.0,
                "num_heads": 4,
                "num_layers": 2,
                "conditioning_dropout": 0.1,
            },
            "lightning_module": {
                "ema_decay": 0.9999,
                "learning_rate": 1e-4,
                "fid_reference_path": "dummy.pt",
                "guidance_scale": 1.0,
                "t_steps": 5,
            },
        }
    )

    model = LitFlow(config)
    model.train()

    # Fake batch
    x1 = torch.randn(batch_size, seq_len, dim)
    c = torch.randn(batch_size, seq_len, dim)
    batch = (x1, c)

    # --- Training step test ---
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients computed"

    # --- Validation step test ---
    model.eval()
    model.on_validation_start()
    with torch.no_grad():
        val_loss = model.validation_step(batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)
    model.on_validation_epoch_end()
    model.on_validation_end()
