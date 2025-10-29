import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
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


def _collate_lvcf(
    batch: List[Tuple[pd.DataFrame, str]],
    vcols: List[str],
    name_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    B = len(batch)
    embed_dim = len(vcols)
    max_len = max(len(df) for df, _ in batch)
    var = torch.zeros(B, max_len, embed_dim, dtype=torch.float32)
    val = torch.zeros(B, max_len, 1, dtype=torch.float32)
    minute = torch.zeros(B, max_len, 1, dtype=torch.float32)
    set_id = torch.zeros(B, max_len, 1, dtype=torch.long)
    age = torch.zeros(B, max_len, 1, dtype=torch.float32)
    carry_mask = torch.zeros(B, max_len, 1, dtype=torch.float32)
    # New: feature id per token (from 'feature_id' column, or mapped from 'variable')
    feat_id = torch.zeros(B, max_len, 1, dtype=torch.long)
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
        # Extract feature id preference order: feature_id -> mapped variable name -> raw variable
        try:
            if "feature_id" in df.columns:
                ids = df["feature_id"].to_numpy(dtype=np.int64, copy=False)
                feat_id[b, :n] = torch.from_numpy(ids).view(-1, 1)
            elif "variable" in df.columns:
                var_col = df["variable"]
                if name_to_id is not None and not pd.api.types.is_integer_dtype(var_col):
                    # Map string variable names to global ids
                    mapped = var_col.map(name_to_id)
                    if mapped.isnull().any():
                        # Fallback to categorical codes for unknowns
                        codes = var_col.astype("category").cat.codes.to_numpy(dtype=np.int64)
                        ids = codes
                    else:
                        ids = mapped.to_numpy(dtype=np.int64)
                elif pd.api.types.is_integer_dtype(var_col):
                    ids = var_col.to_numpy(dtype=np.int64, copy=False)
                else:
                    # Fallback: per-file categorical codes (not globally stable)
                    ids = var_col.astype("category").cat.codes.to_numpy(dtype=np.int64)
                feat_id[b, :n] = torch.from_numpy(ids).view(-1, 1)
        except Exception:
            # Leave zeros if extraction fails
            pass
        padding_mask[b, :n] = False

    return {
        "var": var,
        "val": val,
        "minute": minute,
        "set_id": set_id,
        "age": age,
        "carry_mask": carry_mask,
        "padding_mask": padding_mask,
        "feat_id": feat_id,
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
        schema_dir: Optional[str] = None,
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
        self.schema_dir = schema_dir

        self.vcols: Optional[List[str]] = None
        self._rng = random.Random(seed)
        self._name_to_id: Optional[Dict[str, int]] = None

    def setup(self, stage=None):
        # Instantiate datasets
        self.train = PatientDataset("train", self.saved_dir)
        self.valid = PatientDataset("valid", self.saved_dir)
        self.test = PatientDataset("test", self.saved_dir)

        # Detect embedding columns from the first available file
        sample_df, _ = self.train[0]
        self.vcols = _detect_vcols(sample_df)

        # Optional: load schema mapping (name -> feature_id)
        self._name_to_id = None
        if isinstance(self.schema_dir, str) and self.schema_dir:
            import pandas as _pd
            schema_path = self.schema_dir
            if os.path.isdir(schema_path):
                schema_path = os.path.join(schema_path, "schema.csv")
            if os.path.isfile(schema_path):
                try:
                    sdf = _pd.read_csv(schema_path)
                    if "name" in sdf.columns and "feature_id" in sdf.columns:
                        self._name_to_id = {str(n): int(i) for n, i in zip(sdf["name"].astype(str), sdf["feature_id"].astype(int))}
                except Exception:
                    self._name_to_id = None

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
            out = _collate_lvcf(b, vcols, self._name_to_id)
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
                out = _collate_lvcf(b, self.vcols, self._name_to_id)
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


# ---------------- Mortality label data module (Stage C) ---------------- #

def _pid_variants(raw_id: Any) -> List[str]:
    variants: List[str] = []
    try:
        s = str(raw_id).strip()
    except Exception:
        return variants
    if s == "":
        return variants
    variants.append(s)
    try:
        f = float(s)
        i = int(f)
        variants.append(str(i))
        variants.append(f"{i}.0")
    except Exception:
        pass
    seen = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _read_label_map(label_csv: str) -> Dict[str, int]:
    import csv
    mp: Dict[str, int] = {}
    with open(label_csv, "r") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ts_id" not in reader.fieldnames or "in_hospital_mortality" not in reader.fieldnames:
            raise ValueError("label_csv must contain columns ts_id,in_hospital_mortality")
        for row in reader:
            pid_raw = row["ts_id"]
            y_raw = str(row["in_hospital_mortality"]).strip()
            if y_raw == "" or y_raw.lower() == "nan":
                lab = 0
            else:
                try:
                    lab = 1 if int(float(y_raw)) == 1 else 0
                except Exception:
                    lab = 0
            for pid in _pid_variants(pid_raw):
                if pid == "":
                    continue
                mp[pid] = lab
    return mp


class MortalityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        saved_dir: str,
        label_csv: str,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
        smoke: bool = False,
        smoke_batch_size: int = 8,
        use_weighted_sampler: bool = True,
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smoke = smoke
        self.smoke_batch_size = smoke_batch_size
        self.id_to_label: Dict[str, int] = {}
        self.vcols: Optional[List[str]] = None
        self.use_weighted_sampler = bool(use_weighted_sampler)
        self._train_sample_weights: Optional[torch.Tensor] = None
        self._train_labels: Optional[List[int]] = None

    def setup(self, stage=None):
        self.id_to_label = _read_label_map(self.label_csv)
        self.train = PatientDataset("train", self.saved_dir)
        self.valid = PatientDataset("valid", self.saved_dir)
        self.test = PatientDataset("test", self.saved_dir)
        sample_df, _ = self.train[0]
        self.vcols = _detect_vcols(sample_df)
        tr_labels = [self.id_to_label.get(pid, 0) for pid in self.train.ids]
        n_pos = max(1, sum(tr_labels))
        n_neg = max(1, len(tr_labels) - n_pos)
        self.pos_weight = float(n_neg) / float(n_pos)
        self._train_labels = tr_labels
        # Prepare per-sample weights for class-balanced sampling (positives ~ minority)
        if self.use_weighted_sampler:
            N = max(1, len(tr_labels))
            w_pos = float(N) / (2.0 * float(n_pos))
            w_neg = float(N) / (2.0 * float(n_neg))
            weights = [w_pos if lab == 1 else w_neg for lab in tr_labels]
            self._train_sample_weights = torch.tensor(weights, dtype=torch.double)
        if self.smoke:
            n = len(self.train)
            k = min(self.smoke_batch_size, n)
            self._smoke_idx = list(range(k))
        else:
            self._smoke_idx = None

    def _collate_with_label(self, batch: List[Tuple[Any, str]]):
        assert self.vcols is not None
        out = _collate_lvcf(batch, self.vcols)
        labels_list: List[float] = []
        missing: List[str] = []
        for (_, pid) in batch:
            lab = self.id_to_label.get(pid, None)
            if lab is None:
                for v in _pid_variants(pid):
                    if v in self.id_to_label:
                        lab = self.id_to_label[v]
                        break
            if lab is None:
                missing.append(str(pid))
            else:
                labels_list.append(float(lab))
        if len(missing) > 0:
            raise ValueError(f"Missing labels for IDs (sample): {missing[:10]} (total {len(missing)})")
        out["y"] = torch.tensor(labels_list, dtype=torch.float32)
        return out

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_with_label,
        )

    def train_dataloader(self):
        if getattr(self, "_smoke_idx", None) is not None:
            subset = Subset(self.train, self._smoke_idx)
            return DataLoader(
                subset,
                batch_size=len(subset),
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_with_label,
            )
        if self.use_weighted_sampler and self._train_sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self._train_sample_weights,
                num_samples=len(self.train),
                replacement=True,
            )
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_with_label,
            )
        return self._loader(self.train, True)

    def val_dataloader(self):
        return self._loader(self.valid, False)

    def test_dataloader(self):
        return self._loader(self.test, False)

