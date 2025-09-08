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


def _compress_like_raw(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Variant A: reverse expansion (exp->raw). Drop carry-forward tokens per set; drop empty sets.
    Operates on an already-collated batch dictionary.
    """
    device = batch["var"].device
    B, N, D = batch["var"].shape
    set_id = batch["set_id"].long().squeeze(-1)
    carry_mask = batch.get("carry_mask", None)
    if carry_mask is None:
        carry_mask = torch.zeros(B, N, 1, device=device)
    carry = carry_mask.squeeze(-1) > 0.5
    minute = batch["minute"].squeeze(-1)
    var = batch["var"]
    val = batch["val"].squeeze(-1)
    padding = batch.get("padding_mask", torch.zeros(B, N, dtype=torch.bool, device=device))

    new_vars: List[torch.Tensor] = []
    new_vals: List[torch.Tensor] = []
    new_minutes: List[torch.Tensor] = []
    new_setids: List[torch.Tensor] = []
    for b in range(B):
        mask_valid = ~padding[b]
        sid = set_id[b, mask_valid]
        car = carry[b, mask_valid]
        t = minute[b, mask_valid]
        v = var[b, mask_valid]
        x = val[b, mask_valid]
        uniq, counts = torch.unique_consecutive(sid, return_counts=True)
        idx_splits = torch.split(torch.arange(len(sid), device=device), [int(c) for c in counts])
        kept_tokens: List[torch.Tensor] = []
        kept_minutes: List[torch.Tensor] = []
        kept_setids: List[torch.Tensor] = []
        running_sid = 0
        for idx in idx_splits:
            keep = ~car[idx]
            if keep.sum() == 0:
                continue
            kept_tokens.append(v[idx][keep])
            kept_minutes.append(t[idx][keep])
            kept_setids.append(torch.full((int(keep.sum()),), running_sid, dtype=torch.long, device=device))
            running_sid += 1
        if len(kept_tokens) == 0:
            kept_tokens = [torch.zeros(1, D, device=device)]
            kept_minutes = [torch.zeros(1, device=device)]
            kept_setids = [torch.zeros(1, dtype=torch.long, device=device)]
        v_new = torch.cat(kept_tokens, dim=0)
        t_new = torch.cat(kept_minutes, dim=0)
        s_new = torch.cat(kept_setids, dim=0)
        x_new = torch.ones(len(v_new), device=device)
        new_vars.append(v_new)
        new_vals.append(x_new)
        new_minutes.append(t_new)
        new_setids.append(s_new)

    max_len = max(v.shape[0] for v in new_vars)
    out_var = torch.zeros(B, max_len, D, device=device)
    out_val = torch.zeros(B, max_len, 1, device=device)
    out_minute = torch.zeros(B, max_len, 1, device=device)
    out_setid = torch.zeros(B, max_len, 1, dtype=torch.long, device=device)
    out_age = torch.zeros(B, max_len, 1, device=device)
    out_carry = torch.zeros(B, max_len, 1, device=device)
    out_pad = torch.ones(B, max_len, dtype=torch.bool, device=device)
    for b in range(B):
        n = new_vars[b].shape[0]
        out_var[b, :n] = new_vars[b]
        out_val[b, :n, 0] = new_vals[b]
        out_minute[b, :n, 0] = new_minutes[b]
        out_setid[b, :n, 0] = new_setids[b]
        out_pad[b, :n] = False
    return {
        "var": out_var,
        "val": out_val,
        "minute": out_minute,
        "set_id": out_setid,
        "age": out_age,
        "carry_mask": out_carry,
        "padding_mask": out_pad,
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
        apply_A: bool = False,
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
        self.apply_A = apply_A

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
        def _collate(b):
            out = _collate_lvcf(b, vcols)
            if self.apply_A:
                out = _compress_like_raw(out)
            return out
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
        )

    def train_dataloader(self):
        if getattr(self, "_smoke_indices", None) is not None:
            subset = Subset(self.train, self._smoke_indices)
            # ensure one batch containing exactly smoke_batch_size samples
            def _collate(b):
                out = _collate_lvcf(b, self.vcols)
                if self.apply_A:
                    out = _compress_like_raw(out)
                return out
            return DataLoader(
                subset,
                batch_size=len(subset),
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=_collate,
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

