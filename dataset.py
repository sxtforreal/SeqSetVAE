import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning.pytorch as pl
import os
import glob
import re
from functools import lru_cache
from typing import Tuple, List, Dict, Any
import random


##### SetVAE
class SetVAEDataset(Dataset):
    def __init__(
        self,
        partition,
        saved_dir="/home/sunx/data/aiiih/data/mimic/processed/saved_sets",
    ):
        if partition not in ["train", "valid", "test"]:
            raise ValueError("Partition must be 'train', 'valid', or 'test'")

        # Find all Parquet files for the partition
        partition_dir = os.path.join(saved_dir, partition)
        self.parquet_files = sorted(
            glob.glob(os.path.join(partition_dir, "length_*.parquet")),
            key=lambda x: (
                int(re.search(r"length_(\d+)", x).group(1))
                if re.search(r"length_(\d+)", x)
                else 0
            ),
        )
        if not self.parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for partition {partition} in {partition_dir}. Run parser.py first."
            )

        # Track set counts and lengths
        self.set_counts = []
        self.lengths = []
        self.offsets = [0]
        for file in self.parquet_files:
            df = pd.read_parquet(file, columns=["set_index"], engine="pyarrow")
            set_count = df["set_index"].nunique()  # Number of unique sets
            self.set_counts.append(set_count)
            length = int(re.search(r"length_(\d+)", file).group(1))
            self.lengths.append(length)
            self.offsets.append(self.offsets[-1] + set_count)

    def __len__(self):
        return self.offsets[-1]

    @lru_cache(maxsize=128)
    def _load_df(self, file_index):
        return pd.read_parquet(self.parquet_files[file_index], engine="pyarrow")

    def __getitem__(self, idx):
        # Find which file contains the idx
        for i in range(len(self.parquet_files)):
            if idx < self.offsets[i + 1]:
                df = self._load_df(i)
                length = self.lengths[i]
                # Find the set_index corresponding to the idx
                set_indices = df["set_index"].unique()
                local_set_idx = idx - self.offsets[i]
                target_set_index = set_indices[local_set_idx]
                # Extract sub-DataFrame for the set
                sub_df = df[df["set_index"] == target_set_index].copy()
                if len(sub_df) != length:
                    raise ValueError(
                        f"Expected sub_df length {length}, got {len(sub_df)}"
                    )
                return sub_df


class LengthWeightedSampler(Sampler):
    def __init__(self, dataset, w=0.5):
        self.dataset = dataset
        self.w = w
        self.length_indices = {}
        for i, length in enumerate(dataset.lengths):
            start, end = dataset.offsets[i], dataset.offsets[i + 1]
            self.length_indices[length] = list(range(start, end))
        self.lengths = list(self.length_indices.keys())
        self.num_lengths = len(self.lengths)
        self.length_probs = [w / l + (1 - w) / self.num_lengths for l in self.lengths]
        self.length_probs = [p / sum(self.length_probs) for p in self.length_probs]

    def __iter__(self):
        indices = []
        for _ in range(len(self.dataset)):
            length = random.choices(self.lengths, weights=self.length_probs, k=1)[0]
            idx = random.choice(self.length_indices[length])
            indices.append(idx)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class SetVAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        saved_dir="/home/sunx/data/aiiih/data/mimic/processed/saved_sets",
        params_map_path="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.params_map_path = params_map_path
        cached = pd.read_csv(os.path.join(saved_dir, "../cached.csv"))
        self.cached_embs = {
            row["Key"]: torch.tensor(
                row.iloc[1:].to_numpy(dtype=np.float32), dtype=torch.float32
            )
            for _, row in cached.iterrows()
        }
        self.params_map = None

    def setup(self):
        stats = pd.read_csv(self.params_map_path)
        self.params_map = {
            row["variable"]: {"mean": row["mean"], "std": row["std"]}
            for _, row in stats.iterrows()
        }
        self.train_dataset = SetVAEDataset("train", self.saved_dir)
        self.val_dataset = SetVAEDataset("valid", self.saved_dir)
        self.test_dataset = SetVAEDataset("test", self.saved_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=LengthWeightedSampler(self.train_dataset, w=0.5),
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):

        assert len(batch) == 1, "Batch size must be 1"
        sub_df = batch[0]
        if not isinstance(sub_df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(sub_df)}")

        try:
            var_tsr = torch.stack([self.cached_embs[var] for var in sub_df["variable"]])
            var_tsr = var_tsr.unsqueeze(0)

            val_tsr = torch.tensor(sub_df["value"].to_numpy(dtype=np.float32)).reshape(
                -1, 1
            )
            event_types = sub_df["variable"].to_numpy()
            normalized_vals = torch.zeros_like(val_tsr)
            for i, event in enumerate(event_types):
                if event in self.params_map:
                    mean = self.params_map[event]["mean"]
                    std = self.params_map[event]["std"]
                    normalized_vals[i] = (val_tsr[i] - mean) / std
                else:
                    normalized_vals[i] = val_tsr[i]

            normalized_vals = normalized_vals.unsqueeze(0)
            return {"var": var_tsr, "val": normalized_vals}

        except KeyError as e:
            raise ValueError(f"Variable not found in cached_embs or params_map: {e}")


# if __name__ == "__main__":

#     data_module = SetVAEDataModule()
#     data_module.setup()
#     print("Number of training data:", len(data_module.train_dataset))
#     print("Number of validation data:", len(data_module.val_dataset))
#     print("Number of test data:", len(data_module.test_dataset))

#     train_loader = data_module.train_dataloader()
#     for batch in train_loader:
#         print("Batch var shape:", batch["var"].shape)  # (1, N, 768) - not standardized
#         print("Batch val shape:", batch["val"].shape)  # (1, N, 1)
#         break


##### SeqSetVAE


class SeqSetVAEDataset(Dataset):
    """Each parquet file stores the events of **one** patient.

    The file name itself (without extension) is treated as the unique patient id (tsid),
    e.g.  ``patient_ehr/train/200001.0.parquet``  →  tsid = "200001.0".
    """

    def __init__(self, partition: str, saved_dir: str):
        if partition not in {"train", "valid", "test"}:
            raise ValueError("partition must be train/valid/test")

        # collect parquet paths
        partition_dir = os.path.join(saved_dir, partition)
        self.parquet_files: List[str] = sorted(
            glob.glob(os.path.join(partition_dir, "*.parquet"))
        )
        if len(self.parquet_files) == 0:
            raise FileNotFoundError(f"no parquet under {partition_dir}")

        # patient ids extracted from filenames (basename without extension)
        self.patient_ids: List[str] = [
            os.path.splitext(os.path.basename(p))[0] for p in self.parquet_files
        ]

    def __len__(self):
        return len(self.parquet_files)

    @lru_cache(maxsize=128)
    def _load_df(self, file_idx: int) -> pd.DataFrame:
        return pd.read_parquet(self.parquet_files[file_idx], engine="pyarrow")

    def __getitem__(self, idx: int) -> Tuple[pd.DataFrame, str]:
        df = self._load_df(idx)
        tsid = self.patient_ids[idx]
        return df, tsid


def dynamic_collate_fn(batch: List[Tuple[pd.DataFrame, str]], cached_embs: Dict, params_map: Dict, label_map: Dict) -> Dict[str, Any]:
    """
    Enhanced collate function that supports variable batch sizes with dynamic padding.
    
    Args:
        batch: List of (DataFrame, tsid) tuples
        cached_embs: Dictionary mapping variable names to embeddings
        params_map: Dictionary for value normalization
        label_map: Dictionary mapping tsid to labels
    
    Returns:
        Dictionary with batched and padded tensors
    """
    batch_size = len(batch)
    
    # Process each patient in the batch
    all_vars, all_vals, all_minutes, all_set_ids, all_labels = [], [], [], [], []
    max_events = 0
    
    for df, tsid in batch:
        # Event variable embeddings
        var_tensors = [cached_embs[v] for v in df["variable"]]
        var_tsr = torch.stack(var_tensors) if var_tensors else torch.empty(0, cached_embs[next(iter(cached_embs))].shape[0])
        
        # Value normalization
        raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
        norm_vals = torch.zeros_like(raw_vals)
        for i, ev in enumerate(df["variable"].to_numpy()):
            if ev in params_map:
                m, s = params_map[ev]["mean"], params_map[ev]["std"]
                norm_vals[i] = (raw_vals[i] - m) / s
            else:
                norm_vals[i] = raw_vals[i]
        
        # Minute tensor
        minute_tsr = torch.tensor(df["minute"].to_numpy(dtype=np.float32)).view(-1, 1)
        
        # Set ID tensor (derive from minute changes if not present)
        if "set_index" in df.columns:
            setids = torch.tensor(df["set_index"].to_numpy(dtype=np.int64)).view(-1, 1)
        else:
            setids = torch.tensor(
                (df["minute"].diff().fillna(0) != 0).cumsum().to_numpy(dtype=np.int64)
            ).view(-1, 1)
        
        # Label lookup
        label_val = label_map.get(int(float(tsid)), 0)  # Default to 0 if not found
        
        all_vars.append(var_tsr)
        all_vals.append(norm_vals)
        all_minutes.append(minute_tsr)
        all_set_ids.append(setids)
        all_labels.append(label_val)
        
        max_events = max(max_events, len(df))
    
    # Pad sequences to max length in batch
    if max_events == 0:
        # Handle empty batch case
        embed_dim = cached_embs[next(iter(cached_embs))].shape[0]
        return {
            "var": torch.zeros(batch_size, 1, embed_dim),
            "val": torch.zeros(batch_size, 1, 1),
            "minute": torch.zeros(batch_size, 1, 1),
            "set_id": torch.zeros(batch_size, 1, 1, dtype=torch.long),
            "label": torch.tensor(all_labels, dtype=torch.long),
            "padding_mask": torch.ones(batch_size, 1, dtype=torch.bool)
        }
    
    # Create padded tensors
    embed_dim = all_vars[0].shape[1] if len(all_vars[0]) > 0 else cached_embs[next(iter(cached_embs))].shape[0]
    
    padded_vars = torch.zeros(batch_size, max_events, embed_dim)
    padded_vals = torch.zeros(batch_size, max_events, 1)
    padded_minutes = torch.zeros(batch_size, max_events, 1)
    padded_set_ids = torch.zeros(batch_size, max_events, 1, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_events, dtype=torch.bool)
    
    for i, (var_tsr, val_tsr, min_tsr, set_tsr) in enumerate(zip(all_vars, all_vals, all_minutes, all_set_ids)):
        seq_len = len(var_tsr)
        if seq_len > 0:
            padded_vars[i, :seq_len] = var_tsr
            padded_vals[i, :seq_len] = val_tsr
            padded_minutes[i, :seq_len] = min_tsr
            padded_set_ids[i, :seq_len] = set_tsr
            padding_mask[i, :seq_len] = False  # False indicates real data
    
    return {
        "var": padded_vars,
        "val": padded_vals,
        "minute": padded_minutes,
        "set_id": padded_set_ids,
        "label": torch.tensor(all_labels, dtype=torch.long),
        "padding_mask": padding_mask
    }


class SeqSetVAEDataModule(pl.LightningDataModule):
    """Lightning DataModule handling patient-level parquet & label lookup (oc)."""

    def __init__(
        self,
        saved_dir: str,
        params_map_path: str,
        label_path: str,
        batch_size: int = 1,
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.params_map_path = params_map_path
        self.label_path = label_path
        self.batch_size = batch_size

        cached = pd.read_csv(os.path.join(saved_dir, "../cached.csv"))
        self.cached_embs: Dict[str, torch.Tensor] = {
            row["Key"]: torch.tensor(row.iloc[1:].to_numpy(dtype=np.float32))
            for _, row in cached.iterrows()
        }

        self.params_map: Dict[str, Dict[str, float]] | None = None
        self.label_map: Dict[str, int] | None = None

    def setup(self, stage=None):
        stats = pd.read_csv(self.params_map_path)
        self.params_map = {
            row["variable"]: {"mean": row["mean"], "std": row["std"]}
            for _, row in stats.iterrows()
        }

        # label map: expecting CSV with columns tsid,label
        label_df = pd.read_csv(self.label_path)
        self.label_map = {
            int(r["ts_id"]): int(r["in_hospital_mortality"])
            for _, r in label_df.iterrows()
        }

        self.train_dataset = SeqSetVAEDataset("train", self.saved_dir)
        self.val_dataset = SeqSetVAEDataset("valid", self.saved_dir)
        self.test_dataset = SeqSetVAEDataset("test", self.saved_dir)

    def _create_loader(self, ds, shuffle=False):
        # Use standard collate function for batch_size=1, enhanced for larger batches
        if self.batch_size == 1:
            collate_fn = self._collate_fn
        else:
            collate_fn = lambda batch: dynamic_collate_fn(batch, self.cached_embs, self.params_map, self.label_map)
            
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_loader(self.test_dataset, shuffle=False)

    def _collate_fn(self, batch: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        # Standard collate function for batch_size=1 (patient-level)
        assert len(batch) == 1, "batch size must be 1 (one patient)"
        df, tsid = batch[0]

        # event variable embedding
        var_tsr = torch.stack([self.cached_embs[v] for v in df["variable"]]).unsqueeze(
            0
        )

        # value normalization
        raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
        norm_vals = torch.zeros_like(raw_vals)
        for i, ev in enumerate(df["variable"].to_numpy()):
            if ev in self.params_map:
                m, s = self.params_map[ev]["mean"], self.params_map[ev]["std"]
                norm_vals[i] = (raw_vals[i] - m) / s
            else:
                norm_vals[i] = raw_vals[i]
        norm_vals = norm_vals.unsqueeze(0)

        # minute tensor
        minute_tsr = torch.tensor(df["minute"].to_numpy(dtype=np.float32)).view(
            1, -1, 1
        )

        # setid tensor (if absent derive by minute change)
        if "set_index" in df.columns:
            setids = torch.tensor(df["set_index"].to_numpy(dtype=np.int64)).view(
                1, -1, 1
            )
        else:
            setids = torch.tensor(
                (df["minute"].diff().fillna(0) != 0).cumsum().to_numpy(dtype=np.int64)
            ).view(1, -1, 1)

        # label lookup from oc
        label_val = self.label_map.get(int(float(tsid)))
        label_tsr = torch.tensor([label_val]).long()

        return {
            "var": var_tsr,
            "val": norm_vals,
            "minute": minute_tsr,
            "set_id": setids,
            "label": label_tsr,
        }


if __name__ == "__main__":

    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    print("Number of training data:", len(data_module.train_dataset))
    print("Number of validation data:", len(data_module.val_dataset))
    print("Number of test data:", len(data_module.test_dataset))

    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print("Batch var shape:", batch["var"].shape)  # (1, N, 768) - not standardized
        print("Batch val shape:", batch["val"].shape)  # (1, N, 1)
        print("Batch minute shape:", batch["minute"].shape)  # (1, N, 1)
        print("Batch set_id shape:", batch["set_id"].shape)  # (1, N, 1)
        print("Batch label shape:", batch["label"].shape)
        break
