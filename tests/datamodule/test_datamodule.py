import torch
from omegaconf import OmegaConf
from forge.datamodule import ForgeDataModule


# smoke test
def test_forge_datamodule():
    cfg = OmegaConf.create(
        {
            "train_csv_path": "/new-stg/home/young/forge/data/train_hq.csv.gz",
            "val_csv_path": "/new-stg/home/young/forge/data/val_hq.csv.gz",
            "lmdb_path": "/new-stg/home/young/forge/data/raygun_hq_embeddings_numpy.lmdb",
            "batch_size": 4,
            "num_workers": 8,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
        }
    )

    dm = ForgeDataModule(cfg)
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    z_protein_embeddings, conditioning_protein_embeddings = batch

    # Basic shape checks
    assert isinstance(z_protein_embeddings, torch.Tensor)
    assert isinstance(conditioning_protein_embeddings, torch.Tensor)

    # Batch sizes should match cfg
    assert z_protein_embeddings.shape[0] == cfg.batch_size
    assert conditioning_protein_embeddings.shape[0] == cfg.batch_size

    # Embedding dims should match
    assert z_protein_embeddings.shape[1:] == conditioning_protein_embeddings.shape[1:]
