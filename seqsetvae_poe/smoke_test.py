import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW

from .model import PoESeqSetVAEPretrain


def build_synthetic_batch(batch_size: int = 2, max_sets: int = 8, tokens_per_set: int = 16, embed_dim: int = 768):
    """
    Create a synthetic batch compatible with DataModule/_collate_lvcf outputs.
    Shapes:
      - var: [B, N, D]
      - val: [B, N, 1]
      - minute: [B, N, 1]
      - set_id: [B, N, 1]
      - age: [B, N, 1]
      - carry_mask: [B, N, 1]
      - padding_mask: [B, N]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = batch_size
    # total tokens N per patient
    S = max_sets
    n_per_set = tokens_per_set
    N = S * n_per_set

    var = torch.randn(B, N, embed_dim, device=device)
    # values: mostly small; some zeros for masked
    val = (torch.randn(B, N, 1, device=device) * 0.5).clamp_(-3, 3)
    # time: minutes increasing; each set one minute apart
    base = torch.arange(S, device=device).float().view(1, S, 1)
    minute = base.repeat(B, 1, 1).repeat_interleave(n_per_set, dim=1)
    minute = minute + torch.zeros(B, N, 1, device=device)
    # set ids
    set_id = torch.arange(S, device=device).view(1, S, 1).repeat(B, 1, 1).repeat_interleave(n_per_set, dim=1)
    # age (time since last observation for carry-forwarded tokens)
    age = torch.zeros(B, N, 1, device=device)
    # carry mask: within each set, randomly mark ~40% as carry
    carry_mask = (torch.rand(B, N, 1, device=device) < 0.4).float()
    # padding mask: no padding in synthetic batch
    padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    return {
        "var": var,
        "val": val,
        "minute": minute,
        "set_id": set_id,
        "age": age,
        "carry_mask": carry_mask,
        "padding_mask": padding_mask,
    }


def run_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyperparameters (small for speed)
    input_dim = 768
    reduced_dim = 256
    latent_dim = 128
    levels = 2
    heads = 4
    m = 16
    ff_dim = 512
    transformer_heads = 4
    transformer_layers = 2

    model = PoESeqSetVAEPretrain(
        input_dim=input_dim,
        reduced_dim=reduced_dim,
        latent_dim=latent_dim,
        levels=levels,
        heads=heads,
        m=m,
        beta=1.0,
        lr=3e-4,
        ff_dim=ff_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        transformer_dropout=0.1,
        warmup_beta=True,
        max_beta=0.05,
        beta_warmup_steps=1000,
        free_bits=0.03,
        stale_dropout_p=0.2,
        set_mae_ratio=0.15,
        enable_next_change=True,
        next_change_weight=0.3,
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)

    # Batch
    batch = build_synthetic_batch()

    # One forward/backward step
    model.train()
    recon, kl, next_c = model.forward(batch)
    beta = model._beta()
    total = recon + beta * kl + 0.3 * next_c
    total.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print({
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "next_change": float(next_c.detach().cpu()),
        "beta": float(beta),
        "total": float(total.detach().cpu()),
        "device": str(device),
    })


if __name__ == "__main__":
    run_smoke()

