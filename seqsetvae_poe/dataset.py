import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from typing import Dict, Any, List, Tuple, Optional


class PatientDataset(Dataset):
    """
    Per-patient Parquet reader. Each file contains a full patient timeline with
    at least the following columns:
      - variable: categorical code key matching `cached.csv` embeddings
      - value: numeric value
      - minute: absolute minute since admission
      - set_index: integer set id per unique minute (monotonic non-decreasing)
      - is_carry: 1 if the value is carry-forwarded from last observation
      - age: patient age at the event time (years)

    File name (without extension) is the patient id.
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


def _normalize_vals(df: pd.DataFrame, params_map: Dict[str, Dict[str, float]]) -> torch.Tensor:
    raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
    norm_vals = torch.zeros_like(raw_vals)
    for i, ev in enumerate(df["variable"].to_numpy()):
        if ev in params_map:
            m, s = params_map[ev]["mean"], params_map[ev]["std"]
            norm_vals[i] = (raw_vals[i] - m) / (s if s > 0 else 1.0)
        else:
            norm_vals[i] = raw_vals[i]
    return norm_vals


def _collate_with_padding(
    batch: List[Tuple[pd.DataFrame, str]],
    cached_embs: Dict[str, torch.Tensor],
    params_map: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    B = len(batch)
    embed_dim = next(iter(cached_embs.values())).shape[0]
    max_len = max(len(df) for df, _ in batch)
    var = torch.zeros(B, max_len, embed_dim)
    val = torch.zeros(B, max_len, 1)
    minute = torch.zeros(B, max_len, 1)
    set_id = torch.zeros(B, max_len, 1, dtype=torch.long)
    age = torch.zeros(B, max_len, 1)
    carry_mask = torch.zeros(B, max_len, 1)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)

    for b, (df, _) in enumerate(batch):
        n = len(df)
        if n == 0:
            continue
        var[b, :n] = torch.stack([cached_embs[v] for v in df["variable"]])
        val[b, :n] = _normalize_vals(df, params_map)
        minute[b, :n] = torch.tensor(df["minute"].to_numpy(dtype=np.float32)).view(-1, 1)
        if "set_index" in df.columns:
            set_id[b, :n] = torch.tensor(df["set_index"].to_numpy(dtype=np.int64)).view(-1, 1)
        else:
            set_id[b, :n] = torch.tensor((df["minute"].diff().fillna(0) != 0).cumsum().to_numpy(dtype=np.int64)).view(-1, 1)
        if "age" in df.columns:
            age[b, :n] = torch.tensor(df["age"].to_numpy(dtype=np.float32)).view(-1, 1)
        if "is_carry" in df.columns:
            carry_mask[b, :n] = torch.tensor(df["is_carry"].to_numpy(dtype=np.float32)).view(-1, 1)
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
        params_map_path: str,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.params_map_path = params_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        cached = pd.read_csv(os.path.join(saved_dir, "../cached.csv"))
        self.cached_embs: Dict[str, torch.Tensor] = {
            row["Key"]: torch.tensor(row.iloc[1:].to_numpy(dtype=np.float32))
            for _, row in cached.iterrows()
        }
        self.params_map: Optional[Dict[str, Dict[str, float]]] = None

    def setup(self, stage=None):
        stats = pd.read_csv(self.params_map_path)
        self.params_map = {
            r["variable"]: {"mean": r["mean"], "std": r["std"]} for _, r in stats.iterrows()
        }
        self.train = PatientDataset("train", self.saved_dir)
        self.valid = PatientDataset("valid", self.saved_dir)
        self.test = PatientDataset("test", self.saved_dir)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: _collate_with_padding(b, self.cached_embs, self.params_map),
        )

    def train_dataloader(self):
        return self._loader(self.train, True)

    def val_dataloader(self):
        return self._loader(self.valid, False)

    def test_dataloader(self):
        return self._loader(self.test, False)

