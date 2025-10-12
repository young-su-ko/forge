import lmdb
import numpy as np
import gzip
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig


class InteractionDataset(Dataset):
    def __init__(self, interaction_csv_path: Path):
        self.interactions = self._load_interactions(interaction_csv_path)

    def _load_interactions(self, path):
        interactions = []
        with gzip.open(path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                interactions.append(row)
        return interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        z_protein_id, conditioning_protein_id = self.interactions[idx]
        return z_protein_id, conditioning_protein_id


class LMDBCollator:
    def __init__(self, lmdb_path: Path, dtype=np.float32, shape=(50, 1280)):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        self.dtype = dtype
        self.shape = shape

    def __call__(self, batch):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

        z_protein_ids, conditioning_protein_ids = zip(*batch)

        with self.env.begin(write=False) as txn:
            z_embeddings = [
                torch.from_numpy(
                    np.frombuffer(txn.get(z.encode("utf-8")), dtype=self.dtype).reshape(
                        self.shape
                    )
                )
                for z in z_protein_ids
            ]
            c_embeddings = [
                torch.from_numpy(
                    np.frombuffer(txn.get(c.encode("utf-8")), dtype=self.dtype).reshape(
                        self.shape
                    )
                )
                for c in conditioning_protein_ids
            ]

        return torch.stack(z_embeddings), torch.stack(c_embeddings)


class ForgeDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.train_csv_path = Path(cfg.train_csv_path)
        self.val_csv_path = Path(cfg.val_csv_path)
        self.lmdb_path = Path(cfg.lmdb_path)
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers
        self.prefetch_factor = cfg.prefetch_factor

    def setup(self, stage: str):
        self.train_dataset = InteractionDataset(self.train_csv_path)
        self.val_dataset = InteractionDataset(self.val_csv_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=LMDBCollator(self.lmdb_path),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=LMDBCollator(self.lmdb_path),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
