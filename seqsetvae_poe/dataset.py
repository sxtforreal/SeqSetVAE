import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning.pytorch as pl
from typing import Dict, Any, List, Tuple, Optional


class PatientDataset(Dataset):
    """
    Per-patient Parquet reader compatible with LVCF outputs.

    Each parquet must include columns:
      - variable, value, time, set_index, age, is_carry, v0..v{D-1}

    File name (without extension) is used as the patient id.
    """

    def __init__(self, partition: str, saved_dir: str):
        assert partition in {"train", "valid", "test"}
        base = os.path.join(saved_dir, partition)
        self.files: List[str] = sorted(glob.glob(os.path.join(base, "*.parquet")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No parquet files under {base}")
        self.ids: List[str] = [os.path.splitext(os.path.basename(p))[0] for p in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[pd.DataFrame, str]:
        df = pd.read_parquet(self.files[idx], engine="pyarrow")
        return df, self.ids[idx]


def _detect_vcols(df: pd.DataFrame) -> List[str]:
    vcols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    if len(vcols) == 0:
        raise RuntimeError("No embedding columns found (expected v0..v{D-1})")
    # sort by numeric suffix to ensure correct order
    vcols = sorted(vcols, key=lambda c: int(c[1:]))
    return vcols


def _collate_lvcf(batch: List[Tuple[pd.DataFrame, str]], vcols: List[str]) -> Dict[str, Any]:
    B = len(batch)
    embed_dim = len(vcols)
    max_len = max(len(df) for df, _ in batch)
    var = torch.zeros(B, max_len, embed_dim, dtype=torch.float32)
    val = torch.zeros(B, max_len, 1, dtype=torch.float32)
    minute = torch.zeros(B, max_len, 1, dtype=torch.float32)
    set_id = torch.zeros(B, max_len, 1, dtype=torch.long)
    age = torch.zeros(B, max_len, 1, dtype=torch.float32)
    carry_mask = torch.zeros(B, max_len, 1, dtype=torch.float32)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)

    for b, (df, _) in enumerate(batch):
        n = len(df)
        if n == 0:
            continue
        # embeddings
        var_np = df[vcols].to_numpy(dtype=np.float32, copy=False)
        var[b, :n] = torch.from_numpy(var_np)
        # values already normalized by LVCF
        val[b, :n] = torch.from_numpy(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
        # time -> minute key expected by the model
        minute[b, :n] = torch.from_numpy(df["time"].to_numpy(dtype=np.float32)).view(-1, 1)
        # set index
        set_id[b, :n] = torch.from_numpy(df["set_index"].to_numpy(dtype=np.int64)).view(-1, 1)
        # optional extras
        if "age" in df.columns:
            age[b, :n] = torch.from_numpy(df["age"].to_numpy(dtype=np.float32)).view(-1, 1)
        if "is_carry" in df.columns:
            carry_mask[b, :n] = torch.from_numpy(df["is_carry"].to_numpy(dtype=np.float32)).view(-1, 1)
        padding_mask[b, :n] = False

    return {
        "var": var,
        "val": val,
        "minute": minute,
        "set_id": set_id,
        "age": age,
        "carry_mask": carry_mask,
        "padding_mask": padding_mask,
    }


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        saved_dir: str,
        params_map_path: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        smoke: bool = False,
        smoke_batch_size: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.saved_dir = saved_dir
        # params_map_path kept for backward-compatibility; not used with LVCF outputs
        self.params_map_path = params_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smoke = smoke
        self.smoke_batch_size = smoke_batch_size
        self.seed = seed

        self.vcols: Optional[List[str]] = None
        self._rng = random.Random(seed)

    def setup(self, stage=None):
        # Instantiate datasets
        self.train = PatientDataset("train", self.saved_dir)
        self.valid = PatientDataset("valid", self.saved_dir)
        self.test = PatientDataset("test", self.saved_dir)

        # Detect embedding columns from the first available file
        sample_df, _ = self.train[0]
        self.vcols = _detect_vcols(sample_df)

        # Optionally prepare smoke subset indices
        if self.smoke:
            n = len(self.train)
            k = min(self.smoke_batch_size, n)
            self._smoke_indices = self._rng.sample(range(n), k)
        else:
            self._smoke_indices = None

    def _loader(self, ds, shuffle):
        vcols = self.vcols
        assert vcols is not None, "DataModule.setup() must be called before creating dataloaders"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: _collate_lvcf(b, vcols),
        )

    def train_dataloader(self):
        if getattr(self, "_smoke_indices", None) is not None:
            subset = Subset(self.train, self._smoke_indices)
            # ensure one batch containing exactly smoke_batch_size samples
            return DataLoader(
                subset,
                batch_size=len(subset),
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=lambda b: _collate_lvcf(b, self.vcols),
            )
        return self._loader(self.train, True)

    def val_dataloader(self):
        return self._loader(self.valid, False)

    def test_dataloader(self):
        return self._loader(self.test, False)

    def build_smoke_batch(self) -> Dict[str, Any]:
        """
        Build and return a single collated batch of `smoke_batch_size` random
        train samples for quick end-to-end testing without a Trainer.
        """
        if self.vcols is None:
            # lazily call setup if not run
            self.setup()
        assert self.vcols is not None
        n = len(self.train)
        k = min(self.smoke_batch_size, n)
        idxs = self._rng.sample(range(n), k)
        batch_items: List[Tuple[pd.DataFrame, str]] = [self.train[i] for i in idxs]
        return _collate_lvcf(batch_items, self.vcols)

