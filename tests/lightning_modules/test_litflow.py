import torch
import pytest
from omegaconf import OmegaConf
from forge.lightning_modules.lit_flow import LitFlow


@pytest.mark.parametrize("batch_size, seq_len, dim", [(2, 50, 1280)])
def test_litflow_forward_and_backward(batch_size, seq_len, dim):
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
            },
        }
    )

    model = LitFlow(config)
    model.train()

    x1 = torch.randn(batch_size, seq_len, dim)
    c = torch.randn(batch_size, seq_len, dim)
    batch = (x1, c)

    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
