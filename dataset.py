import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import lightning.pytorch as pl
import os
import glob
import re
from functools import lru_cache
from typing import Tuple, List, Dict, Any, Optional
import random


##### SetVAE Dataset Classes

class SetVAEDataset(Dataset):
    """
    Dataset class for SetVAE that handles medical event sets.
    
    This dataset loads pre-processed medical events stored in Parquet files,
    where each file contains events grouped by set length. The dataset supports
    efficient loading and caching of variable-length event sets.
    
    Args:
        partition (str): Data partition - 'train', 'valid', or 'test'
        saved_dir (str): Directory containing the processed Parquet files
    """
    
    def __init__(
        self,
        partition,
        saved_dir="/home/sunx/data/aiiih/data/mimic/processed/saved_sets",
    ):
        if partition not in ["train", "valid", "test"]:
            raise ValueError("Partition must be 'train', 'valid', or 'test'")

        # Find all Parquet files for the specified partition
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

        # Track set counts and lengths for efficient indexing
        self.set_counts = []  # Number of sets in each file
        self.lengths = []     # Event set lengths for each file
        self.offsets = [0]    # Cumulative offsets for global indexing
        
        # Pre-compute metadata for all files
        for file in self.parquet_files:
            df = pd.read_parquet(file, columns=["set_index"], engine="pyarrow")
            set_count = df["set_index"].nunique()  # Number of unique sets
            self.set_counts.append(set_count)
            
            # Extract set length from filename (e.g., "length_5.parquet" -> 5)
            length = int(re.search(r"length_(\d+)", file).group(1))
            self.lengths.append(length)
            
            # Update cumulative offsets for global indexing
            self.offsets.append(self.offsets[-1] + set_count)

    def __len__(self):
        """Return total number of event sets across all files."""
        return self.offsets[-1]

    @lru_cache(maxsize=128)
    def _load_df(self, file_index):
        """
        Load and cache DataFrame from Parquet file.
        
        Uses LRU cache to avoid repeated I/O operations for frequently
        accessed files during training.
        
        Args:
            file_index (int): Index of the file to load
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        return pd.read_parquet(self.parquet_files[file_index], engine="pyarrow")

    def __getitem__(self, idx):
        """
        Retrieve a single event set by global index.
        
        Args:
            idx (int): Global index across all files
            
        Returns:
            pd.DataFrame: Event set containing variables, values, and metadata
        """
        # Find which file contains the requested index
        for i in range(len(self.parquet_files)):
            if idx < self.offsets[i + 1]:
                df = self._load_df(i)
                length = self.lengths[i]
                
                # Find the specific set within the file
                set_indices = df["set_index"].unique()
                local_set_idx = idx - self.offsets[i]
                target_set_index = set_indices[local_set_idx]
                
                # Extract the event set
                sub_df = df[df["set_index"] == target_set_index].copy()
                if len(sub_df) != length:
                    raise ValueError(
                        f"Expected sub_df length {length}, got {len(sub_df)}"
                    )
                return sub_df


class LengthWeightedSampler(Sampler):
    """
    Custom sampler that balances sampling across different event set lengths.
    
    This sampler helps prevent bias towards shorter or longer sequences by
    weighting the sampling probability based on set length and uniform distribution.
    
    Args:
        dataset (SetVAEDataset): Dataset to sample from
        w (float): Weight parameter balancing length-based vs uniform sampling
                  w=0: pure uniform sampling, w=1: pure length-based sampling
    """
    
    def __init__(self, dataset, w=0.5):
        self.dataset = dataset
        self.w = w
        
        # Group indices by set length
        self.length_indices = {}
        for i, length in enumerate(dataset.lengths):
            start, end = dataset.offsets[i], dataset.offsets[i + 1]
            self.length_indices[length] = list(range(start, end))
        
        self.lengths = list(self.length_indices.keys())
        self.num_lengths = len(self.lengths)
        
        # Calculate sampling probabilities for each length
        # Combines inverse length weighting with uniform distribution
        self.length_probs = [w / l + (1 - w) / self.num_lengths for l in self.lengths]
        self.length_probs = [p / sum(self.length_probs) for p in self.length_probs]

    def __iter__(self):
        """Generate indices for one epoch of training."""
        indices = []
        for _ in range(len(self.dataset)):
            # Sample a length based on computed probabilities
            length = random.choices(self.lengths, weights=self.length_probs, k=1)[0]
            # Sample a random index from that length group
            idx = random.choice(self.length_indices[length])
            indices.append(idx)
        
        # Shuffle to avoid any ordering bias
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """Return the number of samples per epoch."""
        return len(self.dataset)


class SetVAEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for SetVAE training.
    
    Handles data loading, preprocessing, and batch creation for medical event sets.
    Includes embedding lookup and value normalization.
    
    Args:
        saved_dir (str): Directory containing processed data files
        params_map_path (str): Path to statistics file for value normalization
    """
    
    def __init__(
        self,
        saved_dir="/home/sunx/data/aiiih/data/mimic/processed/saved_sets",
        params_map_path="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.params_map_path = params_map_path
        
        # Load pre-computed embeddings for medical variables
        cached = pd.read_csv(os.path.join(saved_dir, "../cached.csv"))
        self.cached_embs = {
            row["Key"]: torch.tensor(
                row.iloc[1:].to_numpy(dtype=np.float32), dtype=torch.float32
            )
            for _, row in cached.iterrows()
        }
        self.params_map = None

    def setup(self, stage=None):
        """
        Set up datasets and preprocessing parameters.
        
        Args:
            stage (str, optional): Training stage ('fit', 'test', etc.)
        """
        # Load normalization statistics for medical variables
        stats = pd.read_csv(self.params_map_path)
        self.params_map = {
            row["variable"]: {"mean": row["mean"], "std": row["std"]}
            for _, row in stats.iterrows()
        }
        
        # Initialize datasets for each partition
        self.train_dataset = SetVAEDataset("train", self.saved_dir)
        self.val_dataset = SetVAEDataset("valid", self.saved_dir)
        self.test_dataset = SetVAEDataset("test", self.saved_dir)

    def train_dataloader(self):
        """Create training data loader with length-weighted sampling."""
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Process one event set at a time
            sampler=LengthWeightedSampler(self.train_dataset, w=0.5),
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Create test data loader."""
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle=False):
        """
        Helper method to create a data loader with standard settings.
        
        Args:
            dataset: Dataset to load from
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: Configured data loader
        """
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
        """
        Collate function to process a batch of event sets.
        
        Converts raw DataFrame data into tensors suitable for model training.
        Performs embedding lookup and value normalization.
        
        Args:
            batch (list): List containing one DataFrame (batch_size=1)
            
        Returns:
            dict: Dictionary with 'var' (embeddings) and 'val' (normalized values)
        """
        assert len(batch) == 1, "Batch size must be 1"
        sub_df = batch[0]
        if not isinstance(sub_df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(sub_df)}")

        try:
            # Look up embeddings for each medical variable
            var_tsr = torch.stack([self.cached_embs[var] for var in sub_df["variable"]])
            var_tsr = var_tsr.unsqueeze(0)  # Add batch dimension

            # Extract and normalize medical values
            val_tsr = torch.tensor(sub_df["value"].to_numpy(dtype=np.float32)).reshape(
                -1, 1
            )
            event_types = sub_df["variable"].to_numpy()
            normalized_vals = torch.zeros_like(val_tsr)
            
            # Apply z-score normalization using pre-computed statistics
            for i, event in enumerate(event_types):
                if event in self.params_map:
                    mean = self.params_map[event]["mean"]
                    std = self.params_map[event]["std"]
                    normalized_vals[i] = (val_tsr[i] - mean) / std
                else:
                    # Keep original value if no normalization stats available
                    normalized_vals[i] = val_tsr[i]

            normalized_vals = normalized_vals.unsqueeze(0)  # Add batch dimension
            return {"var": var_tsr, "val": normalized_vals}

        except KeyError as e:
            raise ValueError(f"Variable not found in cached_embs or params_map: {e}")


##### SeqSetVAE Dataset Classes

# -------------------- LVCF-aware helpers --------------------
def _has_v_columns(df: pd.DataFrame) -> bool:
    """Detect whether DataFrame contains materialized embedding columns v0..v{D-1}."""
    return any(c.startswith("v") and c[1:].isdigit() for c in df.columns)


def _v_columns(df: pd.DataFrame) -> List[str]:
    """Return sorted embedding column names v0..v{D-1} if present."""
    vcols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    vcols.sort(key=lambda x: int(x[1:]))
    return vcols


def _extract_time_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Return time tensor [N,1] using 'time' if present else 'minute'."""
    if "time" in df.columns:
        return torch.tensor(df["time"].to_numpy(dtype=np.float32)).view(-1, 1)
    return torch.tensor(df["minute"].to_numpy(dtype=np.float32)).view(-1, 1)


def _compute_set_ids(df: pd.DataFrame) -> torch.Tensor:
    """Return set ids [N,1] using 'set_index' if present, else derive from time changes."""
    if "set_index" in df.columns:
        return torch.tensor(df["set_index"].to_numpy(dtype=np.int64)).view(-1, 1)
    time_col = "time" if "time" in df.columns else "minute"
    return torch.tensor((df[time_col].diff().fillna(0) != 0).cumsum().to_numpy(dtype=np.int64)).view(-1, 1)

class SeqSetVAEDataset(Dataset):
    """
    Dataset class for Sequential SetVAE that handles patient-level medical data.
    
    Each Parquet file stores the complete medical history of one patient.
    The filename (without extension) serves as the unique patient ID.
    
    Example: "patient_ehr/train/200001.0.parquet" → patient ID = "200001.0"
    
    Args:
        partition (str): Data partition - 'train', 'valid', or 'test'
        saved_dir (str): Directory containing patient-level Parquet files
    """

    def __init__(self, partition: str, saved_dir: str):
        if partition not in {"train", "valid", "test"}:
            raise ValueError("partition must be train/valid/test")

        # Collect all patient files for this partition
        partition_dir = os.path.join(saved_dir, partition)
        self.parquet_files: List[str] = sorted(
            glob.glob(os.path.join(partition_dir, "*.parquet"))
        )
        if len(self.parquet_files) == 0:
            raise FileNotFoundError(f"no parquet under {partition_dir}")

        # Extract patient IDs from filenames (basename without extension)
        self.patient_ids: List[str] = [
            os.path.splitext(os.path.basename(p))[0] for p in self.parquet_files
        ]

    def __len__(self):
        """Return number of patients in this partition."""
        return len(self.parquet_files)

    @lru_cache(maxsize=128)
    def _load_df(self, file_idx: int) -> pd.DataFrame:
        """
        Load and cache patient data from Parquet file.
        
        Args:
            file_idx (int): Index of the patient file to load
            
        Returns:
            pd.DataFrame: Patient's complete medical event history
        """
        return pd.read_parquet(self.parquet_files[file_idx], engine="pyarrow")

    def __getitem__(self, idx: int) -> Tuple[pd.DataFrame, str]:
        """
        Retrieve a patient's medical data and ID.
        
        Args:
            idx (int): Patient index
            
        Returns:
            tuple: (DataFrame with patient events, patient ID string)
        """
        df = self._load_df(idx)
        tsid = self.patient_ids[idx]
        return df, tsid


def dynamic_collate_fn(batch: List[Tuple[pd.DataFrame, str]], cached_embs: Dict, params_map: Dict, label_map: Dict) -> Dict[str, Any]:
    """
    Enhanced collate function supporting variable batch sizes with dynamic padding.
    
    This function processes multiple patients in a batch, handling variable-length
    sequences by padding them to the maximum length in the batch.
    
    Args:
        batch: List of (DataFrame, patient_id) tuples
        cached_embs: Dictionary mapping variable names to embeddings
        params_map: Dictionary for value normalization (mean/std)
        label_map: Dictionary mapping patient_id to outcome labels
    
    Returns:
        Dictionary with batched and padded tensors:
        - var: Variable embeddings [batch_size, max_events, embed_dim]
        - val: Normalized values [batch_size, max_events, 1]
        - minute: Time stamps [batch_size, max_events, 1]
        - set_id: Set identifiers [batch_size, max_events, 1]
        - label: Outcome labels [batch_size]
        - padding_mask: Boolean mask indicating padded positions [batch_size, max_events]
    """
    batch_size = len(batch)
    
    # Process each patient in the batch
    all_vars, all_vals, all_minutes, all_set_ids, all_ages, all_carries, all_labels = [], [], [], [], [], [], []
    max_events = 0
    
    for df, tsid in batch:
        # Prefer materialized v* embeddings
        if _has_v_columns(df):
            vcols = _v_columns(df)
            var_tsr = torch.tensor(df[vcols].to_numpy(dtype=np.float32))
            # Values are already normalized by LVCF pipeline
            norm_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
        else:
            var_tensors = [cached_embs[v] for v in df["variable"]]
            var_tsr = torch.stack(var_tensors) if var_tensors else torch.empty(0, next(iter(cached_embs.values())).shape[0])
            # Normalize with stats
            raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
            norm_vals = torch.zeros_like(raw_vals)
            for i, ev in enumerate(df["variable"].to_numpy()):
                if ev in params_map:
                    m, s = params_map[ev]["mean"], params_map[ev]["std"]
                    norm_vals[i] = (raw_vals[i] - m) / (s if s > 0 else 1.0)
                else:
                    norm_vals[i] = raw_vals[i]

        # Time and set ids (prefer 'time' and 'set_index')
        minute_tsr = _extract_time_tensor(df)
        setids = _compute_set_ids(df)

        # Ancillary features
        if "age" in df.columns:
            age_tsr = torch.tensor(df["age"].to_numpy(dtype=np.float32)).view(-1, 1)
        else:
            age_tsr = torch.zeros_like(minute_tsr)
        if "is_carry" in df.columns:
            carry_tsr = torch.tensor(df["is_carry"].to_numpy(dtype=np.float32)).view(-1, 1)
        else:
            carry_tsr = torch.zeros_like(minute_tsr)

        # Look up outcome label
        label_val = label_map.get(int(float(tsid)), 0)  # Default to 0 if not found

        # Collect all processed data
        all_vars.append(var_tsr)
        all_vals.append(norm_vals)
        all_minutes.append(minute_tsr)
        all_set_ids.append(setids)
        all_ages.append(age_tsr)
        all_carries.append(carry_tsr)
        all_labels.append(label_val)
        
        # Track maximum sequence length for padding
        max_events = max(max_events, len(df))
    
    # Handle empty batch case
    if max_events == 0:
        embed_dim = cached_embs[next(iter(cached_embs))].shape[0]
        return {
            "var": torch.zeros(batch_size, 1, embed_dim),
            "val": torch.zeros(batch_size, 1, 1),
            "minute": torch.zeros(batch_size, 1, 1),
            "set_id": torch.zeros(batch_size, 1, 1, dtype=torch.long),
            "label": torch.tensor(all_labels, dtype=torch.long),
            "age": torch.zeros(batch_size, 1, 1),
            "carry_mask": torch.zeros(batch_size, 1, 1),
            "padding_mask": torch.ones(batch_size, 1, dtype=torch.bool)
        }
    
    # Create padded tensors for the batch
    first_non_empty = next((v for v in all_vars if v.numel() > 0), None)
    embed_dim = first_non_empty.shape[1] if first_non_empty is not None else cached_embs[next(iter(cached_embs))].shape[0]
    
    padded_vars = torch.zeros(batch_size, max_events, embed_dim)
    padded_vals = torch.zeros(batch_size, max_events, 1)
    padded_minutes = torch.zeros(batch_size, max_events, 1)
    padded_set_ids = torch.zeros(batch_size, max_events, 1, dtype=torch.long)
    padded_ages = torch.zeros(batch_size, max_events, 1)
    padded_carries = torch.zeros(batch_size, max_events, 1)
    padding_mask = torch.ones(batch_size, max_events, dtype=torch.bool)
    
    # Fill in actual data and create padding mask
    for i, (var_tsr, val_tsr, min_tsr, set_tsr, age_tsr, carry_tsr) in enumerate(zip(all_vars, all_vals, all_minutes, all_set_ids, all_ages, all_carries)):
        seq_len = len(var_tsr)
        if seq_len > 0:
            padded_vars[i, :seq_len] = var_tsr
            padded_vals[i, :seq_len] = val_tsr
            padded_minutes[i, :seq_len] = min_tsr
            padded_set_ids[i, :seq_len] = set_tsr
            padded_ages[i, :seq_len] = age_tsr
            padded_carries[i, :seq_len] = carry_tsr
            padding_mask[i, :seq_len] = False  # False indicates real data
    
    return {
        "var": padded_vars,
        "val": padded_vals,
        "minute": padded_minutes,
        "set_id": padded_set_ids,
        "age": padded_ages,
        "carry_mask": padded_carries,
        "label": torch.tensor(all_labels, dtype=torch.long),
        "padding_mask": padding_mask
    }


class SeqSetVAEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Sequential SetVAE training.
    
    Handles patient-level data loading with outcome labels for supervised learning.
    Supports both single-patient batches and multi-patient batches with dynamic padding.
    
    Args:
        saved_dir (str): Directory containing patient-level Parquet files
        params_map_path (str): Path to statistics file for value normalization
        label_path (str): Path to CSV file containing patient outcome labels
        batch_size (int): Batch size for training (1 for single-patient, >1 for multi-patient)
    """

    def __init__(
        self,
        saved_dir: str,
        params_map_path: str,
        label_path: str,
        batch_size: int = 1,
        max_sequence_length: int = None,  # New: Maximum sequence length limit
        use_dynamic_padding: bool = True,  # New: Whether to use dynamic padding
        num_workers: int = 4,  # Number of data loader workers
        pin_memory: bool = True,  # Whether to use pin memory
        smoke: bool = False,  # Smoke mode: build a single test batch from train
        smoke_size: int = 10,  # Number of patients in smoke batch
        smoke_random: bool = True,  # Randomly sample patients for smoke batch
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.params_map_path = params_map_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.use_dynamic_padding = use_dynamic_padding
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smoke = smoke
        self.smoke_size = smoke_size
        self.smoke_random = smoke_random
        self._smoke_indices: Optional[List[int]] = None

        # Load pre-computed embeddings for medical variables
        cached = pd.read_csv(os.path.join(saved_dir, "../cached.csv"))
        self.cached_embs: Dict[str, torch.Tensor] = {
            row["Key"]: torch.tensor(row.iloc[1:].to_numpy(dtype=np.float32))
            for _, row in cached.iterrows()
        }

        self.params_map: Optional[Dict[str, Dict[str, float]]] = None
        self.label_map: Optional[Dict[str, int]] = None

    def setup(self, stage=None):
        """
        Set up datasets and preprocessing parameters.
        
        Args:
            stage (str, optional): Training stage identifier
        """
        # Load normalization statistics
        stats = pd.read_csv(self.params_map_path)
        self.params_map = {
            row["variable"]: {"mean": row["mean"], "std": row["std"]}
            for _, row in stats.iterrows()
        }

        # Load outcome labels (expecting CSV with columns: ts_id, in_hospital_mortality)
        label_df = pd.read_csv(self.label_path)
        self.label_map = {
            int(r["ts_id"]): int(r["in_hospital_mortality"])
            for _, r in label_df.iterrows()
        }

        # Initialize datasets for each partition
        self.train_dataset = SeqSetVAEDataset("train", self.saved_dir)
        self.val_dataset = SeqSetVAEDataset("valid", self.saved_dir)
        self.test_dataset = SeqSetVAEDataset("test", self.saved_dir)

        # Prepare smoke indices from train split if enabled
        if self.smoke:
            total = len(self.train_dataset)
            k = min(self.smoke_size, total)
            if self.smoke_random:
                rng = np.random.default_rng(42)
                self._smoke_indices = rng.choice(total, size=k, replace=False).tolist()
            else:
                self._smoke_indices = list(range(k))

    def _create_loader(self, ds, shuffle=False):
        """
        Create a data loader with appropriate collate function and optimized settings.
        
        Uses standard collate function for batch_size=1, enhanced dynamic collate
        function for larger batches.
        
        Args:
            ds: Dataset to load from
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: Configured data loader
        """
        if self.batch_size == 1:
            collate_fn = self._collate_fn
        else:
            # Use improved dynamic collate function
            collate_fn = lambda batch: self._dynamic_collate_fn(batch)
        
        # Optimized worker count based on batch size and system capabilities
        num_workers = self.num_workers
        pin_memory = self.pin_memory
        
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,  # Don't drop the last incomplete batch
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        )

    def smoke_dataloader(self):
        """
        Build a single test dataloader from 'train' split with `smoke_size` patients.
        Returns a DataLoader that yields one batch collated with dynamic padding.
        """
        if not self.smoke:
            raise ValueError("Smoke mode is disabled. Initialize with smoke=True to use smoke_dataloader().")
        if self._smoke_indices is None or len(self._smoke_indices) == 0:
            raise RuntimeError("Smoke indices are not prepared. Call setup() before smoke_dataloader().")
        subset = Subset(self.train_dataset, self._smoke_indices)
        # Force dynamic collate for batch > 1
        return DataLoader(
            subset,
            batch_size=len(self._smoke_indices),
            shuffle=False,
            num_workers=min(2, self.num_workers),
            collate_fn=lambda batch: self._dynamic_collate_fn(batch),
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=False,
        )

    def get_smoke_batch(self) -> Dict[str, Any]:
        """Convenience helper to return a single smoke batch dict."""
        loader = self.smoke_dataloader()
        return next(iter(loader))

    def _dynamic_collate_fn(self, batch: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Improved dynamic collate function that supports batch training and dynamic padding.
        
        Args:
            batch: List of (DataFrame, patient_id) tuples
            
        Returns:
            Dictionary with batched and padded tensors
        """
        batch_size = len(batch)
        
        # Collect all data and calculate maximum length
        all_vars, all_vals, all_minutes, all_set_ids, all_ages, all_carries, all_labels = [], [], [], [], [], [], []
        max_events = 0
        
        for df, tsid in batch:
            # Limit sequence length (if maximum length is set)
            if self.max_sequence_length and len(df) > self.max_sequence_length:
                df = df.head(self.max_sequence_length)
            
            # Process variable embeddings (prefer LVCF v* columns)
            if _has_v_columns(df):
                vcols = _v_columns(df)
                var_tsr = torch.tensor(df[vcols].to_numpy(dtype=np.float32))
                norm_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
            else:
                var_tensors = [self.cached_embs[v] for v in df["variable"]]
                var_tsr = torch.stack(var_tensors) if var_tensors else torch.empty(0, next(iter(self.cached_embs.values())).shape[0])
                # Normalize
                raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
                norm_vals = torch.zeros_like(raw_vals)
                for i, ev in enumerate(df["variable"].to_numpy()):
                    if ev in self.params_map:
                        m, s = self.params_map[ev]["mean"], self.params_map[ev]["std"]
                        norm_vals[i] = (raw_vals[i] - m) / (s if s > 0 else 1.0)
                    else:
                        norm_vals[i] = raw_vals[i]

            # Process time information
            minute_tsr = _extract_time_tensor(df)

            # Process set IDs
            setids = _compute_set_ids(df)

            # Ancillary features
            if "age" in df.columns:
                age_tsr = torch.tensor(df["age"].to_numpy(dtype=np.float32)).view(-1, 1)
            else:
                age_tsr = torch.zeros_like(minute_tsr)
            if "is_carry" in df.columns:
                carry_tsr = torch.tensor(df["is_carry"].to_numpy(dtype=np.float32)).view(-1, 1)
            else:
                carry_tsr = torch.zeros_like(minute_tsr)
            
            # Look up outcome label
            label_val = self.label_map.get(int(float(tsid)), 0)
            
            # Collect all processed data
            all_vars.append(var_tsr)
            all_vals.append(norm_vals)
            all_minutes.append(minute_tsr)
            all_set_ids.append(setids)
            all_ages.append(age_tsr)
            all_carries.append(carry_tsr)
            all_labels.append(label_val)
            
            # Track maximum sequence length
            max_events = max(max_events, len(df))
        
        # Handle empty batch case
        if max_events == 0:
            embed_dim = next(iter(self.cached_embs.values())).shape[0]
            return {
                "var": torch.zeros(batch_size, 1, embed_dim),
                "val": torch.zeros(batch_size, 1, 1),
                "minute": torch.zeros(batch_size, 1, 1),
                "set_id": torch.zeros(batch_size, 1, 1, dtype=torch.long),
                "label": torch.tensor(all_labels, dtype=torch.long),
                "padding_mask": torch.ones(batch_size, 1, dtype=torch.bool)
            }
        
        # Create padded tensors
        embed_dim = all_vars[0].shape[1] if len(all_vars[0]) > 0 else next(iter(self.cached_embs.values())).shape[0]
        
        padded_vars = torch.zeros(batch_size, max_events, embed_dim)
        padded_vals = torch.zeros(batch_size, max_events, 1)
        padded_minutes = torch.zeros(batch_size, max_events, 1)
        padded_set_ids = torch.zeros(batch_size, max_events, 1, dtype=torch.long)
        padded_ages = torch.zeros(batch_size, max_events, 1)
        padded_carries = torch.zeros(batch_size, max_events, 1)
        padding_mask = torch.ones(batch_size, max_events, dtype=torch.bool)  # True indicates padding positions
        
        # Fill in actual data and create padding mask
        for i, (var_tsr, val_tsr, min_tsr, set_tsr, age_tsr, carry_tsr) in enumerate(zip(all_vars, all_vals, all_minutes, all_set_ids, all_ages, all_carries)):
            seq_len = len(var_tsr)
            if seq_len > 0:
                padded_vars[i, :seq_len] = var_tsr
                padded_vals[i, :seq_len] = val_tsr
                padded_minutes[i, :seq_len] = min_tsr
                padded_set_ids[i, :seq_len] = set_tsr
                padded_ages[i, :seq_len] = age_tsr
                padded_carries[i, :seq_len] = carry_tsr
                padding_mask[i, :seq_len] = False  # False indicates real data
        
        return {
            "var": padded_vars,
            "val": padded_vals,
            "minute": padded_minutes,
            "set_id": padded_set_ids,
            "age": padded_ages,
            "carry_mask": padded_carries,
            "label": torch.tensor(all_labels, dtype=torch.long),
            "padding_mask": padding_mask
        }

    def train_dataloader(self):
        """Create training data loader with shuffling."""
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Create validation data loader without shuffling."""
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Create test data loader without shuffling."""
        return self._create_loader(self.test_dataset, shuffle=False)

    def _collate_fn(self, batch: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Standard collate function for single-patient batches (batch_size=1).
        
        Processes one patient's complete medical history into tensors suitable
        for sequential modeling.
        
        Args:
            batch: List containing one (DataFrame, patient_id) tuple
            
        Returns:
            Dictionary with processed tensors:
            - var: Variable embeddings [1, num_events, embed_dim]
            - val: Normalized values [1, num_events, 1]
            - minute: Time stamps [1, num_events, 1]
            - set_id: Set identifiers [1, num_events, 1]
            - label: Outcome label [1]
        """
        assert len(batch) == 1, "batch size must be 1 (one patient)"
        df, tsid = batch[0]

        # Limit sequence length (if maximum length is set)
        if self.max_sequence_length and len(df) > self.max_sequence_length:
            df = df.head(self.max_sequence_length)

        # Process variable embeddings (prefer LVCF v* columns)
        if _has_v_columns(df):
            vcols = _v_columns(df)
            var_tsr = torch.tensor(df[vcols].to_numpy(dtype=np.float32)).unsqueeze(0)
            norm_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(1, -1, 1)
        else:
            var_tsr = torch.stack([self.cached_embs[v] for v in df["variable"]]).unsqueeze(0)
            # Normalize medical values
            raw_vals = torch.tensor(df["value"].to_numpy(dtype=np.float32)).view(-1, 1)
            norm_vals = torch.zeros_like(raw_vals)
            for i, ev in enumerate(df["variable"].to_numpy()):
                if ev in self.params_map:
                    m, s = self.params_map[ev]["mean"], self.params_map[ev]["std"]
                    norm_vals[i] = (raw_vals[i] - m) / (s if s > 0 else 1.0)
                else:
                    norm_vals[i] = raw_vals[i]
            norm_vals = norm_vals.unsqueeze(0)

        # Process time information
        minute_tsr = _extract_time_tensor(df).view(1, -1, 1)

        # Process set IDs (derive from time changes if not present)
        setids = _compute_set_ids(df).view(1, -1, 1)

        # Ancillary features
        if "age" in df.columns:
            age_tsr = torch.tensor(df["age"].to_numpy(dtype=np.float32)).view(1, -1, 1)
        else:
            age_tsr = torch.zeros_like(minute_tsr)
        if "is_carry" in df.columns:
            carry_tsr = torch.tensor(df["is_carry"].to_numpy(dtype=np.float32)).view(1, -1, 1)
        else:
            carry_tsr = torch.zeros_like(minute_tsr)

        # Look up outcome label
        label_val = self.label_map.get(int(float(tsid)), 0)
        label_tsr = torch.tensor([label_val]).long()

        return {
            "var": var_tsr,
            "val": norm_vals,
            "minute": minute_tsr,
            "set_id": setids,
            "age": age_tsr,
            "carry_mask": carry_tsr,
            "label": label_tsr,
        }


# Example usage and testing code
if __name__ == "__main__":
    # Test SeqSetVAE data module
    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    
    print("Number of training data:", len(data_module.train_dataset))
    print("Number of validation data:", len(data_module.val_dataset))
    print("Number of test data:", len(data_module.test_dataset))

    # Test data loading
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print("Batch var shape:", batch["var"].shape)      # [1, N, 768] - embeddings
        print("Batch val shape:", batch["val"].shape)      # [1, N, 1] - values
        print("Batch minute shape:", batch["minute"].shape) # [1, N, 1] - time
        print("Batch set_id shape:", batch["set_id"].shape) # [1, N, 1] - set IDs
        print("Batch label shape:", batch["label"].shape)   # [1] - outcome
        break
