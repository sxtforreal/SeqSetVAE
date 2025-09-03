#!/usr/bin/env python3
"""
Cache per-set posterior features produced by the pretrained SeqSetVAE encoder.

For each patient (train/valid/test), this script saves:
- mu: [S, D]    per-set posterior means (diagonal Gaussian)
- logvar: [S, D]  per-set posterior log-variances
- minutes: [S]  per-set time (minutes)
- label: int    patient outcome label (0/1)
- meta: dict    {patient_id, num_sets}

Additionally, it accumulates dimension-wise statistics for quick analysis:
- mean_mu2: E[mu^2]
- mean_var: E[exp(logvar)]
- mean_logvar: E[logvar]
- mean_kl: E[0.5*(exp(logvar) + mu^2 - 1 - logvar)]

These stats are saved per partition to support later dimension selection tests.

Usage example:
  python cache_features.py \
    --data_dir /path/to/patient_ehr \
    --params_map_path /path/to/stats.csv \
    --label_path /path/to/oc.csv \
    --pretrained_ckpt /path/to/SeqSetVAE_pretrain.ckpt \
    --output_dir /path/to/cached_features
"""

import os
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F

from dataset import SeqSetVAEDataModule
from model import SeqSetVAEPretrain, load_checkpoint_weights
import config


@torch.no_grad()
def extract_per_set_posteriors(set_encoder, var: torch.Tensor, val: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Run the SetVAE encoder on a single set (batch=1) and return mu/logvar.
    Inputs:
      - var: [1, N, E]
      - val: [1, N, 1]
    Returns:
      dict(mu=[D], logvar=[D])
    """
    set_encoder.eval()
    z_list, _ = set_encoder.encode_from_var_val(var, val)  # last level posterior
    _, mu, logvar = z_list[-1]
    mu = mu.squeeze(0).squeeze(0)       # [D]
    logvar = logvar.squeeze(0).squeeze(0)  # [D]
    return {"mu": mu, "logvar": logvar}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_pretrained_pretrain_model(args_device, pretrained_ckpt: str):
    """Instantiate SeqSetVAEPretrain and load weights for the set encoder."""
    model = SeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        free_bits=config.free_bits,
        transformer_dropout=config.transformer_dropout,
    )
    model.eval()
    model.to(args_device)

    if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
        try:
            state = load_checkpoint_weights(pretrained_ckpt, device=args_device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
    else:
        print("⚠️  No pretrained checkpoint provided or path not found; using random weights.")

    return model


def partition_loader(dm: SeqSetVAEDataModule, partition: str):
    if partition == "train":
        return dm.train_dataloader()
    if partition == "valid":
        return dm.val_dataloader()
    if partition == "test":
        return dm.test_dataloader()
    raise ValueError(f"Unknown partition: {partition}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--params_map_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, default=getattr(config, "pretrained_ckpt", None))
    parser.add_argument("--output_dir", type=str, default="/workspace/cached_features")
    parser.add_argument("--batch_size", type=int, default=1, help="Patient batch size for loading (use 1 to simplify groups)")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = config.device if hasattr(config, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    # Data module for patient-level sequences
    dm = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        use_dynamic_padding=(args.batch_size > 1),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dm.setup()

    # Build pretrained model and extract the set encoder
    pretrain_model = build_pretrained_pretrain_model(device, args.pretrained_ckpt)
    set_encoder = pretrain_model.set_encoder
    set_encoder.eval()
    set_encoder.to(device)

    ensure_dir(args.output_dir)

    partitions = ["train", "valid", "test"]
    manifest: Dict[str, Any] = {}

    for part in partitions:
        print(f"\n📦 Processing partition: {part}")
        part_dir = os.path.join(args.output_dir, part)
        ensure_dir(part_dir)

        # Running sums for stats
        sum_mu2 = torch.zeros(config.latent_dim, dtype=torch.float64, device=device)
        sum_var = torch.zeros(config.latent_dim, dtype=torch.float64, device=device)
        sum_logvar = torch.zeros(config.latent_dim, dtype=torch.float64, device=device)
        sum_kl = torch.zeros(config.latent_dim, dtype=torch.float64, device=device)
        count_sets = 0

        file_map = {}
        loader = partition_loader(dm, part)

        for i, batch in enumerate(loader):
            # Unpack batch
            var = batch["var"]  # [B, M, E]
            val = batch.get("val")  # [B, M, 1]
            minute = batch.get("minute")  # [B, M, 1]
            set_id = batch.get("set_id")  # [B, M, 1]
            labels = batch.get("label")  # [B]

            if var is None or val is None or set_id is None or minute is None:
                print("Skipping malformed batch: missing keys")
                continue

            B = var.size(0)
            for b in range(B):
                var_b = var[b].to(device)
                val_b = val[b].to(device)
                minute_b = minute[b].to(device)
                setid_b = set_id[b].to(device)
                label_b = int(labels[b].item()) if labels is not None else 0

                # Determine valid indices (ignore zero-padded rows if any)
                # Heuristic: sum(abs(var)) + abs(val) > 0
                valid_mask = (var_b.abs().sum(dim=-1) + val_b.abs().sum(dim=-1).squeeze(-1)) > 0
                var_b = var_b[valid_mask]
                val_b = val_b[valid_mask]
                minute_b = minute_b[valid_mask]
                setid_b = setid_b[valid_mask]

                if var_b.numel() == 0:
                    continue

                # Group by set_id values (in order of appearance)
                unique_set_ids = torch.unique_consecutive(setid_b.view(-1))
                mu_list, logvar_list, minutes_list = [], [], []
                for sid in unique_set_ids.tolist():
                    mask = (setid_b.view(-1) == sid)
                    if not mask.any():
                        continue
                    v_set = var_b[mask].unsqueeze(0)  # [1, N_i, E]
                    x_set = val_b[mask].unsqueeze(0)  # [1, N_i, 1]
                    feats = extract_per_set_posteriors(set_encoder, v_set, x_set)
                    mu_i, logvar_i = feats["mu"], feats["logvar"]  # [D]

                    # Compute per-dim stats
                    var_i = logvar_i.exp()
                    kl_i = 0.5 * (var_i + mu_i.pow(2) - 1.0 - logvar_i)
                    sum_mu2 += mu_i.double().pow(2)
                    sum_var += var_i.double()
                    sum_logvar += logvar_i.double()
                    sum_kl += kl_i.double()
                    count_sets += 1

                    mu_list.append(mu_i.cpu())
                    logvar_list.append(logvar_i.cpu())
                    # minute constant within a set; take first
                    minutes_list.append(minute_b[mask][0, 0].detach().cpu())

                if len(mu_list) == 0:
                    continue

                mu_seq = torch.stack(mu_list, dim=0)           # [S, D]
                logvar_seq = torch.stack(logvar_list, dim=0)   # [S, D]
                minutes_seq = torch.stack(minutes_list, dim=0) # [S]

                # Save patient file
                # Try to recover patient id from dataset: SeqSetVAEDataset returns (df, tsid)
                # Our collate loses tsid, so we approximate an incremental id if not available
                # Prefer to pull id from DataFrame, but here we fallback to running index
                # So filename is just running idx within partition
                pid = f"{part}_{i:07d}_b{b}"
                out_path = os.path.join(part_dir, f"{pid}.pt")
                torch.save({
                    "mu": mu_seq,
                    "logvar": logvar_seq,
                    "minutes": minutes_seq,
                    "label": int(label_b),
                    "meta": {"patient_id": pid, "num_sets": int(mu_seq.size(0))}
                }, out_path)
                file_map[pid] = out_path

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1} patient batches...")

        # Save manifest and stats for this partition
        manifest_path = os.path.join(part_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(file_map, f, indent=2)

        stats = {
            "count_sets": int(count_sets),
            "mean_mu2": (sum_mu2 / max(1, count_sets)).cpu().tolist(),
            "mean_var": (sum_var / max(1, count_sets)).cpu().tolist(),
            "mean_logvar": (sum_logvar / max(1, count_sets)).cpu().tolist(),
            "mean_kl": (sum_kl / max(1, count_sets)).cpu().tolist(),
        }
        torch.save(stats, os.path.join(part_dir, "dim_stats.pt"))
        with open(os.path.join(part_dir, "dim_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Finished {part}: {len(file_map)} patients, {count_sets} sets; stats saved.")

    print("\nAll partitions completed. Cached features at:", args.output_dir)


if __name__ == "__main__":
    main()

