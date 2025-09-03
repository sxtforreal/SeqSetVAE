#!/usr/bin/env python3
import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dataset import SeqSetVAEDataModule, SeqSetVAEDataset
from model import SeqSetVAE
import config as cfg


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def extract_patient_features(model: SeqSetVAE, dm: SeqSetVAEDataModule, df, tsid: str, device: torch.device) -> Dict[str, torch.Tensor]:
    # Prepare one-patient batch via datamodule's collate
    batch = dm._collate_fn([(df, tsid)])
    # Move tensors to device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    # Split into sets using model helper
    patient_sets = model._split_sets(
        var=batch["var"],
        val=batch["val"],
        time=batch["minute"],
        set_ids=batch["set_id"],
        padding_mask=batch.get("padding_mask", None),
    )[0]

    # Iterate sets and collect posterior stats
    setvae_inner = model.setvae.setvae
    mu_list: List[torch.Tensor] = []
    logvar_list: List[torch.Tensor] = []
    fused_list: List[torch.Tensor] = []
    minutes_list: List[torch.Tensor] = []

    for s_dict in patient_sets:
        var_t = s_dict["var"].to(device)
        val_t = s_dict["val"].to(device)
        time_t = s_dict["minute"].to(device)

        if hasattr(setvae_inner, "encode_from_var_val"):
            z_list, _ = setvae_inner.encode_from_var_val(var_t, val_t)
        else:
            _, z_list, _ = setvae_inner(var_t, val_t)

        z_sample, mu, logvar = z_list[-1]
        mu = mu.squeeze(0).squeeze(0)       # [D]
        logvar = logvar.squeeze(0).squeeze(0)  # [D]

        fused = model._fuse_vae_features(mu.unsqueeze(0), logvar.unsqueeze(0)).squeeze(0)

        # Each set's minute should be constant; take unique
        minute_val = torch.unique(time_t).float().view(-1)
        if minute_val.numel() != 1:
            minute_val = minute_val[:1]

        mu_list.append(mu.detach().cpu())
        logvar_list.append(logvar.detach().cpu())
        fused_list.append(fused.detach().cpu())
        minutes_list.append(minute_val.detach().cpu())

    if len(mu_list) == 0:
        return {
            "mu": torch.empty(0, cfg.latent_dim),
            "logvar": torch.empty(0, cfg.latent_dim),
            "fused": torch.empty(0, cfg.latent_dim),
            "h": torch.empty(0, cfg.latent_dim),
            "minutes": torch.empty(0),
        }

    mu_seq = torch.stack(mu_list, dim=0)          # [S, D]
    logvar_seq = torch.stack(logvar_list, dim=0)  # [S, D]
    fused_seq = torch.stack(fused_list, dim=0)    # [S, D]
    minutes = torch.stack(minutes_list, dim=0).view(-1)  # [S]

    # Build transformer-enriched features h_seq using model's utilities
    z_seq = fused_seq.unsqueeze(0).to(device)  # [1, S, D]
    pos_tensor = minutes.unsqueeze(0).to(device)  # [1, S]

    z_seq = model._apply_positional_encoding(z_seq, pos_tensor, None)
    z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])
    attn_mask = model._build_causal_time_bias_mask(pos_tensor.view(-1))
    h_seq = model.transformer(z_seq, mask=attn_mask)
    h_seq = model.post_transformer_norm(h_seq)
    h_seq = h_seq.squeeze(0).detach().cpu()  # [S, D]

    return {
        "mu": mu_seq,
        "logvar": logvar_seq,
        "fused": fused_seq,
        "h": h_seq,
        "minutes": minutes,
    }


def build_model(checkpoint_path: str, device: torch.device) -> SeqSetVAE:
    model = SeqSetVAE(
        input_dim=cfg.input_dim,
        reduced_dim=cfg.reduced_dim,
        latent_dim=cfg.latent_dim,
        levels=cfg.levels,
        heads=cfg.heads,
        m=cfg.m,
        beta=cfg.beta,
        lr=cfg.lr,
        num_classes=cfg.num_classes,
        ff_dim=cfg.ff_dim,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
        pretrained_ckpt=checkpoint_path,
        head_type="advanced",
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_split(split: str, dm: SeqSetVAEDataModule, model: SeqSetVAE, out_root: str, device: torch.device, max_patients: int = None):
    if split == "train":
        dataset = dm.train_dataset
    elif split == "valid":
        dataset = dm.val_dataset
    elif split == "test":
        dataset = dm.test_dataset
    else:
        raise ValueError(f"Unknown split: {split}")

    out_dir = os.path.join(out_root, split)
    ensure_dir(out_dir)

    total = len(dataset)
    limit = min(total, max_patients) if max_patients is not None else total
    print(f"Processing split={split} patients: {limit}/{total}")

    for idx in range(limit):
        df, tsid = dataset[idx]
        tsid_str = str(tsid)
        out_path = os.path.join(out_dir, f"{tsid_str}.pt")
        if os.path.exists(out_path):
            # Skip existing
            continue
        try:
            feats = extract_patient_features(model, dm, df, tsid_str, device)
            # Attach metadata
            label_val = dm.label_map.get(int(float(tsid_str)), 0) if dm.label_map is not None else None
            save_obj = {
                "tsid": tsid_str,
                "label": int(label_val) if label_val is not None else None,
                **feats,
            }
            torch.save(save_obj, out_path)
        except Exception as e:
            print(f"Failed {split} idx={idx} tsid={tsid_str}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Freeze pretrained model and extract latent features per split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Patient-level data directory (root contains train/valid/test)")
    parser.add_argument("--params_map_path", type=str, required=True, help="Path to stats.csv for normalization")
    parser.add_argument("--label_path", type=str, required=True, help="Path to labels CSV (ts_id,in_hospital_mortality)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to store features per split")
    parser.add_argument("--splits", type=str, default="train,valid,test", help="Comma-separated splits to process")
    parser.add_argument("--max_patients", type=int, default=None, help="Optional limit per split for quick runs")
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Optional truncate sequence length per patient")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device selection")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Data module (batch_size=1)
    dm = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=1,
        max_sequence_length=args.max_sequence_length,
        use_dynamic_padding=False,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    # Model
    model = build_model(args.checkpoint, device)

    # Output root
    ensure_dir(args.output_dir)

    # Process splits
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        process_split(split, dm, model, args.output_dir, device, args.max_patients)

    print("Done.")


if __name__ == "__main__":
    main()

