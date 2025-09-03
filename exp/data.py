from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .utils import load_npz_data


class SeqDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.mu = arrays["mu"]
        self.logvar = arrays["logvar"]
        self.dt = arrays["dt"]
        self.mask = arrays["mask"]
        self.y = arrays["y"]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.mu[i],
            self.logvar[i],
            self.dt[i],
            self.mask[i],
            self.y[i].astype(np.float32),
        )


def collate(batch):
    mu = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    logvar = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    dt = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
    mask = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.bool)
    y = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)
    return mu, logvar, dt, mask, y


def split_indices(num_samples: int, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    n_val = int(num_samples * val_ratio)
    n_test = int(num_samples * test_ratio)
    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val+n_test]
    train_idx = indices[n_val+n_test:]
    return train_idx, val_idx, test_idx


def make_loaders(
    arrays: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
):
    ds_train = SeqDataset(arrays, train_idx)
    ds_val = SeqDataset(arrays, val_idx)
    ds_test = SeqDataset(arrays, test_idx)

    y_train = arrays["y"][train_idx].astype(np.int64)
    class_counts = np.bincount(y_train, minlength=2)
    # Balanced per-class sampling
    total = len(y_train)
    w_pos = total / (2 * max(1, class_counts[1]))
    w_neg = total / (2 * max(1, class_counts[0]))
    samples_weight = np.where(y_train == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), num_samples=len(samples_weight), replacement=True)

    loader_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, collate_fn=collate, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    return loader_train, loader_val, loader_test, class_counts

