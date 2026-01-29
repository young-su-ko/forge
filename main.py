import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import wandb
import os
import uuid

from forge.datamodule import ForgeDataModule
from forge.lightning_modules.lit_flow import LitFlow


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config.seed)
    run_id = config.wandb.run_id or str(uuid.uuid4())[:8]
    wandb_logger = WandbLogger(
        project=config.wandb.project, name=run_id, id=run_id, resume="allow"
    )

    checkpoint_dir = f"checkpoints/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="fid_conditional_{epoch:02d}-{val_FID_conditional:.4f}",
            monitor="val_FID_conditional",
            mode="min",
            save_top_k=2,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="fid_unconditional_{epoch:02d}-{val_FID_unconditional:.4f}",
            monitor="val_FID_unconditional",
            mode="min",
            save_top_k=2,
        ),

        LearningRateMonitor(logging_interval="step"),
    ]

    lit_module = LitFlow(config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **config.trainer,
    )

    data_module = ForgeDataModule(config.datamodule)

    ckpt_path = config.checkpoint_path if config.checkpoint_path else None

    trainer.fit(model=lit_module, datamodule=data_module, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
