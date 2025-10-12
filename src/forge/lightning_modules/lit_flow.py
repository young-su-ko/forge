import torch
import torch.nn as nn
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.optim import AdamW
from hydra.utils import instantiate
import copy

from forge.inference.fid import FIDCalculator
from forge.inference.flow_simulator import ValFlowSimulator

class LitFlow(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = instantiate(config.model)
        self.criterion = nn.MSELoss()
        self.ema_decay = config.lightning_module.ema_decay
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.fid_calculator = FIDCalculator(
            reference_path=config.lightning_module.fid_reference_path
        )
        self.val_simulator = ValFlowSimulator(
            self.model,
            guidance_scale=config.lightning_module.guidance_scale,
            t_steps=config.lightning_module.t_steps,
        )
        self.val_samples = []
        self.save_hyperparameters()

    def _shared_step(self, batch, prefix: str):
        x1, c = batch
        x0 = torch.randn_like(x1)  # (bs, L, raygun_dim)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1  # (bs, L, raygun_dim)
        dx_t = x1 - x0

        u_t = self.model(xt, t, c)
        loss = self.criterion(u_t, dx_t)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        x1, c = batch
        xt = self.val_simulator.sample(c)
        xt_pooled = xt.mean(dim=1)  # (bs, raygun_dim)
        self.val_samples.append(xt_pooled)
        return loss

    def on_validation_start(self):
        self.swap_to_ema()
        self.fid_calculator.reference_mu = self.fid_calculator.reference_mu.to(self.device)
        self.fid_calculator.reference_sigma = self.fid_calculator.reference_sigma.to(self.device)
        self.val_simulator.velocity_model = self.model
        self.val_samples.clear()

    def on_validation_epoch_end(self):
        samples = torch.cat(self.val_samples, dim=0).to(self.device)  # (bs, raygun_dim)

        sample_mu = samples.mean(dim=0)
        sample_sigma = torch.cov(samples.T)

        fid_value = self.fid_calculator.compute(sample_mu, sample_sigma)

        self.log("val_FID", fid_value, prog_bar=True, sync_dist=True)

    def on_validation_end(self):
        self.swap_to_model()

    def swap_to_ema(self):
        self._model_state_backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_model.state_dict())

    def swap_to_model(self):
        self.model.load_state_dict(self._model_state_backup)
        del self._model_state_backup

    def on_train_batch_end(self, outputs, batch, batch_idx):
        ema_state = self.ema_model.state_dict()
        model_state = self.model.state_dict()
        for key, ema_val in ema_state.items():
            model_val = model_state[key]
            if not torch.is_tensor(ema_val) or not torch.is_floating_point(ema_val):
                continue
            ema_val.mul_(self.ema_decay).add_(model_val, alpha=1.0 - self.ema_decay)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.config.lightning_module.learning_rate
        )

        return optimizer
