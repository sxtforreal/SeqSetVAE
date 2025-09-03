import json
import os
from typing import Dict

import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int):
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_npz_data(path: str) -> Dict[str, np.ndarray]:
    arrays = dict(np.load(path, allow_pickle=True))
    # Normalize shapes
    mu = arrays["mu"]
    logvar = arrays["logvar"]
    dt = arrays["dt"]
    mask = arrays["mask"]
    y = arrays["y"].astype(np.int64)
    if dt.ndim == 2:
        dt = dt[..., None]
    arrays = {"mu": mu, "logvar": logvar, "dt": dt, "mask": mask.astype(bool), "y": y}
    return arrays


def mask_last_index(mask: torch.Tensor) -> torch.Tensor:
    # mask: [B,T] bool, returns index of last True for each row
    B, T = mask.shape
    # convert to int and accumulate
    idx = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)
    masked = torch.where(mask, idx + 1, torch.zeros_like(idx))
    last = masked.max(dim=1).values - 1
    last = torch.clamp(last, min=0)
    return last


def time_encoding_sin(dt: torch.Tensor, num_feats: int = 8) -> torch.Tensor:
    # dt: [B,T,1]
    freq = torch.logspace(0, np.log10(1000.0), steps=num_feats, device=dt.device)
    angles = dt * freq.view(1, 1, -1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

