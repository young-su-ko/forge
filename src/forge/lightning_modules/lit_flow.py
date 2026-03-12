import torch
import torch.nn as nn
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.optim import AdamW
from hydra.utils import instantiate
import copy
import torchmetrics

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
        self.ema_model.eval()
        self.fid_sample_size = config.validation.fid_sample_size
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.fid_calculator = FIDCalculator(
            reference_path=config.validation.fid_reference_path
        )
        self.val_simulator = ValFlowSimulator(
            self.model,
            guidance_scale=config.validation.guidance_scale,
            t_steps=config.validation.t_steps,
        )
        self.val_samples_collected = 0
        self.val_samples_conditional = []
        self.val_samples_unconditional = []
        self.val_pearson = torchmetrics.PearsonCorrCoef()
        self.val_spearman = torchmetrics.SpearmanCorrCoef()
        self.save_hyperparameters()

    def _shared_step(self, batch, prefix: str):
        x1, c, l = batch
        x0 = torch.randn_like(x1)  # (bs, L, raygun_dim)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1  # (bs, L, raygun_dim)
        dx_t = x1 - x0

        u_t = self.model(xt, t, c, l)
        loss = self.criterion(u_t, dx_t)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        x1, c, l = batch

        if self.val_samples_collected < self.fid_sample_size:
            # Conditional generation
            xt, predicted_lengths = self.val_simulator.sample(c, l, unconditional=False)
            xt_cond_pooled = xt.mean(dim=1)  # (bs, raygun_dim)
            self.val_samples_conditional.append(xt_cond_pooled)

            # Unconditional generation
            xt_uncond, _ = self.val_simulator.sample(c, l, unconditional=True)
            xt_uncond_pooled = xt_uncond.mean(dim=1)  # (bs, raygun_dim)
            self.val_samples_unconditional.append(xt_uncond_pooled)

            self.val_samples_collected += xt_uncond.shape[0]
            self.val_pearson.update(
                predicted_lengths.float().view(-1), l.float().view(-1)
            )
            self.val_spearman.update(
                predicted_lengths.float().view(-1), l.float().view(-1)
            )
        return loss

    def on_validation_start(self):
        self.fid_calculator.reference_mu = self.fid_calculator.reference_mu.to(
            self.device
        )
        self.fid_calculator.reference_sigma = self.fid_calculator.reference_sigma.to(
            self.device
        )
        self.val_simulator.set_velocity_model(self.ema_model)
        self.val_simulator.length_predictor.to(self.device)

        self.val_pearson.reset()
        self.val_spearman.reset()

        self.val_samples_collected = 0
        self.val_samples_conditional.clear()
        self.val_samples_unconditional.clear()

    def on_validation_epoch_end(self):
        # Conditional FID
        samples_cond = torch.cat(self.val_samples_conditional, dim=0).to(
            self.device
        )  # (bs, raygun_dim)
        sample_cond_mu = samples_cond.mean(dim=0)
        sample_cond_sigma = torch.cov(samples_cond.T)
        cond_fid_value = self.fid_calculator.compute(sample_cond_mu, sample_cond_sigma)

        # Unconditional FID
        samples_uncond = torch.cat(self.val_samples_unconditional, dim=0).to(
            self.device
        )  # (bs, raygun_dim)
        sample_uncond_mu = samples_uncond.mean(dim=0)
        sample_uncond_sigma = torch.cov(samples_uncond.T)
        uncond_fid_value = self.fid_calculator.compute(
            sample_uncond_mu, sample_uncond_sigma
        )

        pearson = self.val_pearson.compute()
        spearman = self.val_spearman.compute()

        self.log("val_pearson", pearson, prog_bar=True, sync_dist=True)
        self.log("val_spearman", spearman, prog_bar=True, sync_dist=True)
        self.log("val_FID_conditional", cond_fid_value, prog_bar=True, sync_dist=True)
        self.log(
            "val_FID_unconditional", uncond_fid_value, prog_bar=True, sync_dist=True
        )

    # def on_validation_end(self):
    #     self.swap_to_model()

    # def swap_to_ema(self):
    #     self._model_state_backup = copy.deepcopy(self.model.state_dict())
    #     self.model.load_state_dict(self.ema_model.state_dict())

    # def swap_to_model(self):
    #     self.model.load_state_dict(self._model_state_backup)
    #     del self._model_state_backup

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
            self.model.parameters(), lr=self.config.lightning_module.learning_rate
        )

        return optimizer
