import torch
from torch import nn
import pytest
from omegaconf import OmegaConf

from forge.lightning_modules.lit_flow import LitFlow


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        torch.nn.init.constant_(self.linear.weight, 1.0)


@pytest.fixture
def litflow_instance():
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 2,
                "out_features": 2,
                "bias": False,
            },
            "lightning_module": {"ema_decay": 0.5, "learning_rate": 1e-3},
        }
    )
    litflow = LitFlow(cfg)

    # overwrite with dummy for testing
    litflow.model = DummyModel()
    litflow.ema_model = DummyModel()
    for p in litflow.ema_model.parameters():
        p.requires_grad_(False)

    return litflow


def test_ema_update_correctness(litflow_instance):
    litflow = litflow_instance

    torch.testing.assert_close(
        litflow.ema_model.linear.weight, litflow.model.linear.weight
    )

    with torch.no_grad():
        litflow.model.linear.weight.fill_(3.0)

    # Run EMA update
    litflow.on_train_batch_end(None, None, None)

    # Expected EMA: 0.5*1 + 0.5*3 = 2
    expected_ema = torch.full_like(litflow.ema_model.linear.weight, 2.0)
    torch.testing.assert_close(litflow.ema_model.linear.weight, expected_ema)

    # Model weights remain as updated (3.0)
    expected_model = torch.full_like(litflow.model.linear.weight, 3.0)
    torch.testing.assert_close(litflow.model.linear.weight, expected_model)
