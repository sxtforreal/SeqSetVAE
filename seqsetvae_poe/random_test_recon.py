#!/usr/bin/env python3
"""
Randomly pick one set from the test split, reconstruct it, and print
concrete event names with values for both original and reconstruction.

Usage:
  python -u seqsetvae_poe/random_test_recon.py \
    --checkpoint /path/to/setvae_PT.ckpt \
    --data_dir /path/to/SeqSetVAE \
    [--device auto|cpu|cuda] [--seed 42] [--ckpt_type auto|setvae|poe] \
    [--stats_csv /path/to/stats.csv]

Notes:
  - Values printed are de-normalized to original units using the stats CSV.
  - Event names come from the `variable` column of the chosen set.
  - Reconstruction uses the SetVAE encoder/decoder in reduced space if
    a dim reducer exists; mapping back to names is done via cosine NN
    against the set's normalized variable directions, with greedy 1-1 matching.
  - Events carried from previous time (is_carry==1) are annotated as [carry];
    PT-style random-masked tokens are annotated as [mask].
"""

import os
import sys
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd

# Ensure local imports work even if launched from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

import config as cfg  # type: ignore
from dataset import PatientDataset, _detect_vcols  # type: ignore
from model import SetVAEOnlyPretrain, PoESeqSetVAEPretrain  # type: ignore


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict")
    # Strip leading 'model.' (Lightning)
    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    return state


def _detect_ckpt_type(state: Dict[str, torch.Tensor], prefer: str = "auto") -> str:
    if prefer in {"poe", "setvae"}:
        return prefer
    keys = list(state.keys())
    has_prior_head = any(k.startswith("prior_head.") for k in keys)
    # Older PoE might also have 'gru_cell.' or 'rel_time_bucket_embed.'
    has_poe_markers = has_prior_head or any(k.startswith("gru_cell.") for k in keys)
    return "poe" if has_poe_markers else "setvae"


def _detect_num_flows_in_state(state: Dict[str, torch.Tensor]) -> int:
    flow_indices = set()
    for k in state.keys():
        if ".flows." in k:
            try:
                idx = int(k.split(".flows.")[1].split(".")[0])
                flow_indices.add(idx)
            except Exception:
                pass
    return (max(flow_indices) + 1) if flow_indices else 0


def _build_model(ckpt_type: str, lr: float, state: Optional[Dict[str, torch.Tensor]] = None) -> torch.nn.Module:
    if ckpt_type == "poe":
        model = PoESeqSetVAEPretrain(
            input_dim=getattr(cfg, "input_dim", 768),
            reduced_dim=getattr(cfg, "reduced_dim", 256),
            latent_dim=getattr(cfg, "latent_dim", 128),
            levels=getattr(cfg, "levels", 2),
            heads=getattr(cfg, "heads", 2),
            m=getattr(cfg, "m", 16),
            beta=getattr(cfg, "beta", 0.1),
            lr=lr,
            ff_dim=getattr(cfg, "ff_dim", 512),
            transformer_heads=getattr(cfg, "transformer_heads", 8),
            transformer_layers=getattr(cfg, "transformer_layers", 4),
            transformer_dropout=getattr(cfg, "transformer_dropout", 0.15),
            warmup_beta=getattr(cfg, "warmup_beta", True),
            max_beta=getattr(cfg, "max_beta", 0.05),
            beta_warmup_steps=getattr(cfg, "beta_warmup_steps", 8000),
            free_bits=getattr(cfg, "free_bits", 0.03),
            stale_dropout_p=getattr(cfg, "stale_dropout_p", 0.2),
            set_mae_ratio=getattr(cfg, "set_mae_ratio", 0.0),
            enable_next_change=True,
            next_change_weight=0.3,
        )
    else:
        num_flows = _detect_num_flows_in_state(state) if state is not None else 0
        use_flows = num_flows > 0
        model = SetVAEOnlyPretrain(
            input_dim=getattr(cfg, "input_dim", 768),
            reduced_dim=getattr(cfg, "reduced_dim", 256),
            latent_dim=getattr(cfg, "latent_dim", 128),
            levels=getattr(cfg, "levels", 2),
            heads=getattr(cfg, "heads", 2),
            m=getattr(cfg, "m", 16),
            beta=getattr(cfg, "beta", 0.1),
            lr=lr,
            warmup_beta=getattr(cfg, "warmup_beta", True),
            max_beta=getattr(cfg, "max_beta", 0.2),
            beta_warmup_steps=getattr(cfg, "beta_warmup_steps", 8000),
            free_bits=getattr(cfg, "free_bits", 0.05),
            p_stale=getattr(cfg, "stale_dropout_p", 0.5),
            p_live=0.05,
            set_mae_ratio=0.0,
            small_set_mask_prob=0.0,
            small_set_threshold=5,
            max_masks_per_set=0,
            val_noise_std=0.0,
            dir_noise_std=0.0,
            train_decoder_noise_std=0.0,
            eval_decoder_noise_std=0.0,
            use_flows=use_flows,
            num_flows=num_flows,
        )
    return model


@torch.no_grad()
def _encode_decode_set(encoder, var_reduced: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    var_reduced: [1, N, R] (already reduced if reducer exists, else original dim)
    values: [1, N, 1]
    Returns recon in same space: [1, N, R]
    """
    norms = torch.norm(var_reduced, p=2, dim=-1, keepdim=True)
    var_dirs = var_reduced / (norms + 1e-8)
    x_target = var_dirs * values
    z_list, _ = encoder.encode(x_target)
    recon = encoder.decode(z_list, target_n=x_target.size(1), use_mean=True, noise_std=0.0)
    return recon


def _apply_pt_style_random_masks(
    values: torch.Tensor,
    carry_mask: torch.Tensor,
    *,
    p_stale: float,
    p_live: float,
    set_mae_ratio: float,
    small_set_mask_prob: float,
    small_set_threshold: int,
    max_masks_per_set: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply PT-style masking used in SetVAE pretraining to a single set's values.
    Returns (values_masked, rand_mask) where rand_mask indicates tokens masked by
    random procedures (value dropout or Set-MAE), excluding carries.
    - values: [1,N,1] float tensor
    - carry_mask: [N] or [1,N] or [1,N,1] with 1 for carried tokens
    """
    device = values.device
    val = values.clone()
    # Normalize carry shape to [1,N,1]
    if carry_mask.dim() == 1:
        carry = carry_mask.view(1, -1, 1).to(device)
    elif carry_mask.dim() == 2:
        carry = carry_mask.unsqueeze(-1).to(device)
    else:
        carry = carry_mask.to(device)
    # Value dropout
    stale_mask = (torch.rand_like(val) < p_stale) & (carry > 0.5)
    live_mask = (torch.rand_like(val) < p_live) & (carry <= 0.5)
    rand_mask = stale_mask | live_mask
    val = val.masked_fill(rand_mask, 0.0)
    # Set-MAE token masking
    N = val.size(1)
    if N > 0:
        if N <= small_set_threshold:
            if torch.rand((), device=device) < small_set_mask_prob:
                idx = torch.randint(0, N, (1,), device=device)
                val[:, idx, :] = 0.0
                rand_mask[:, idx, :] = True
        else:
            k = int(np.ceil(set_mae_ratio * float(N)))
            k = max(0, min(k, max_masks_per_set))
            if k > 0:
                idx = torch.randperm(N, device=device)[:k]
                val[:, idx, :] = 0.0
                rand_mask[:, idx, :] = True
        # Ensure at least one non-carry token remains unmasked if possible
        non_carry = (carry <= 0.5).squeeze(-1)
        if non_carry.any():
            mask_zero = (val.abs() <= 1e-8).squeeze(-1)
            if torch.all(mask_zero | (~non_carry)):
                indices = torch.nonzero(non_carry, as_tuple=False).squeeze(-1)
                pick = indices[torch.randint(0, len(indices), (1,), device=device)]
                val[:, pick, :] = values[:, pick, :]
                rand_mask[:, pick, :] = False
    return val, rand_mask.squeeze(-1).squeeze(0).to(values.dtype)


def _greedy_match_recon_to_vars(
    recon: np.ndarray,
    var_dirs: np.ndarray,
    variable_names: List[str],
) -> List[Tuple[str, float, float]]:
    """
    Greedy 1-1 assignment of recon vectors to variable directions by cosine similarity.
    Returns list of tuples: (event_name, predicted_value, cosine_similarity)
    """
    eps = 1e-8
    recon_norm = np.linalg.norm(recon, axis=1) + eps
    recon_dir = recon / recon_norm[:, None]
    # similarity matrix [N, M]
    sim = np.matmul(recon_dir, var_dirs.T)
    N, M = sim.shape
    recon_order = np.argsort(recon_norm)[::-1]
    taken_vars = set()
    assignments: List[Tuple[str, float, float]] = []
    for i in recon_order:
        # choose best unmatched variable
        best_j, best_s = -1, -1.0
        for j in range(M):
            if j in taken_vars:
                continue
            s = float(sim[i, j])
            if s > best_s:
                best_s = s
                best_j = j
        if best_j < 0:
            # fallback: allow duplicates if all taken
            best_j = int(np.argmax(sim[i]))
            best_s = float(sim[i, best_j])
        taken_vars.add(best_j)
        # predicted value is projection onto unit direction
        pred_val = float(np.dot(recon[i], var_dirs[best_j]))
        assignments.append((variable_names[best_j], pred_val, best_s))
    # restore original recon order
    out = [None] * N  # type: ignore
    for rank, i in enumerate(recon_order):
        out[i] = assignments[rank]
    return out  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Random test set reconstruction with event names and values")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_type", type=str, choices=["auto", "setvae", "poe"], default="auto")
    ap.add_argument("--patient_index", type=int, default=None)
    ap.add_argument("--set_index", type=int, default=None)
    ap.add_argument("--stats_csv", type=str, default=getattr(cfg, "params_map_path", ""))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load checkpoint and model
    state = _load_state_dict(args.checkpoint)
    ckpt_type = _detect_ckpt_type(state, prefer=args.ckpt_type)
    model = _build_model(ckpt_type, lr=getattr(cfg, "lr", 3e-4), state=state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # Device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    if encoder is None:
        raise RuntimeError("Model does not expose a SetVAE encoder")
    encoder = encoder  # alias

    # Load stats for de-normalization (variable -> mean/std)
    stats_map: Dict[str, Tuple[float, float]] = {}
    if args.stats_csv and os.path.exists(args.stats_csv):
        try:
            stats_df = pd.read_csv(args.stats_csv)
            # Detect key column
            key_col = None
            for c in ["Key", "variable", "event", "name", "key"]:
                if c in stats_df.columns:
                    key_col = c
                    break
            if key_col is None:
                key_col = stats_df.columns[0]
            # Normalize mean/std column names
            rename = {}
            for c in stats_df.columns:
                lc = c.lower()
                if lc == "mean":
                    rename[c] = "mean"
                if lc in ("std", "stdev", "stddev"):
                    rename[c] = "std"
            stats_df = stats_df.rename(columns=rename)
            if "mean" in stats_df.columns and "std" in stats_df.columns:
                for _, r in stats_df.iterrows():
                    v = str(r[key_col])
                    m = float(r["mean"]) if pd.notna(r["mean"]) else 0.0
                    s_raw = float(r["std"]) if pd.notna(r["std"]) else 1.0
                    s = s_raw if abs(s_raw) > 1e-12 else 1.0
                    stats_map[v] = (m, s)
        except Exception as e:
            print(f"[WARN] Failed to load stats_csv '{args.stats_csv}': {e}")

    def _denorm_value(var_name: str, val_norm: float) -> float:
        m, s = stats_map.get(var_name, (0.0, 1.0))
        return val_norm * s + m

    # Dataset: pick one patient + one set from test
    ds = PatientDataset("test", saved_dir=args.data_dir)
    if args.patient_index is None:
        pidx = random.randrange(len(ds))
    else:
        if args.patient_index < 0 or args.patient_index >= len(ds):
            raise IndexError(f"patient_index out of range [0,{len(ds)-1}]")
        pidx = args.patient_index
    df, patient_id = ds[pidx]

    vcols = _detect_vcols(df)
    set_ids = df["set_index"].to_numpy()
    uniq_sets = sorted(set(int(s) for s in set_ids.tolist()))
    if not uniq_sets:
        raise RuntimeError("Selected patient has no sets")
    if args.set_index is None:
        chosen_set = random.choice(uniq_sets)
    else:
        if int(args.set_index) not in uniq_sets:
            raise ValueError(f"set_index {args.set_index} not found in patient {patient_id}")
        chosen_set = int(args.set_index)
    df_set = df[df["set_index"] == chosen_set].reset_index(drop=True)

    # Build tensors for this set
    var_np = df_set[vcols].to_numpy(dtype=np.float32, copy=False)
    val_np = df_set["value"].to_numpy(dtype=np.float32)
    mask_np = df_set["is_carry"].to_numpy(dtype=np.float32) if "is_carry" in df_set.columns else np.zeros_like(val_np)
    var_t = torch.from_numpy(var_np).unsqueeze(0).to(device)           # [1,N,D]
    val_t = torch.from_numpy(val_np).view(1, -1, 1).to(device)         # [1,N,1]

    # Reduce variable vectors if reducer exists
    if getattr(encoder, "dim_reducer", None) is not None:
        var_red = encoder.dim_reducer(var_t)
    else:
        var_red = var_t

    # Apply PT-style random masking in addition to carries (for visualization parity)
    # Pull PT hyperparameters from model where available, with safe defaults
    p_stale = float(getattr(model, "p_stale", getattr(model, "stale_dropout_p", 0.1)))
    p_live = float(getattr(model, "p_live", 0.02))
    set_mae_ratio = float(getattr(model, "set_mae_ratio", 0.02))
    small_set_mask_prob = float(getattr(model, "small_set_mask_prob", 0.1))
    small_set_threshold = int(getattr(model, "small_set_threshold", 5))
    max_masks_per_set = int(getattr(model, "max_masks_per_set", 2))

    val_masked_t, rand_mask_t = _apply_pt_style_random_masks(
        val_t,
        torch.from_numpy(mask_np).to(device),
        p_stale=p_stale,
        p_live=p_live,
        set_mae_ratio=set_mae_ratio,
        small_set_mask_prob=small_set_mask_prob,
        small_set_threshold=small_set_threshold,
        max_masks_per_set=max_masks_per_set,
    )

    # Encode-decode using masked values
    recon_t = _encode_decode_set(encoder, var_red, val_masked_t)        # [1,N,R]

    # Map recon to concrete event names + predicted values
    with torch.no_grad():
        var_dirs = var_red / (torch.norm(var_red, p=2, dim=-1, keepdim=True) + 1e-8)
    var_dirs_np = var_dirs.squeeze(0).detach().cpu().numpy()            # [N,R]
    recon_np = recon_t.squeeze(0).detach().cpu().numpy()                # [N,R]
    event_names: List[str] = df_set["variable"].astype(str).tolist()
    # Build name -> mask map (1 for carried, 0 for observed). If duplicates, keep max (mask if any).
    mask_by_name: Dict[str, float] = {}
    for name, m in zip(event_names, mask_np.tolist()):
        mask_by_name[name] = max(mask_by_name.get(name, 0.0), float(m))
    # Build name -> random mask map
    rand_mask_np = rand_mask_t.detach().cpu().numpy().astype(np.float32)
    rand_by_name: Dict[str, float] = {}
    for name, m in zip(event_names, rand_mask_np.tolist()):
        rand_by_name[name] = max(rand_by_name.get(name, 0.0), float(m))

    assignments = _greedy_match_recon_to_vars(recon_np, var_dirs_np, event_names)

    # Print results
    print("================ Random Test Set Reconstruction ================")
    print(f"Patient ID: {patient_id}")
    print(f"Set index:  {chosen_set}")
    print(f"#Events:    {len(event_names)}")
    # Carry-over and random-mask summaries
    num_carry = int((mask_np > 0.5).sum())
    pct_carry = (num_carry / max(1, len(event_names))) * 100.0
    carried_names = [n for n, m in zip(event_names, mask_np.tolist()) if float(m) > 0.5]
    num_rand = int((rand_mask_np > 0.5).sum())
    pct_rand = (num_rand / max(1, len(event_names))) * 100.0
    rand_names = [n for n, m in zip(event_names, rand_mask_np.tolist()) if float(m) > 0.5]
    print((f"#Carried:   {num_carry} ({pct_carry:.1f}%) -> {', '.join(carried_names)}") if num_carry > 0 else "#Carried:   0 (0.0%)")
    print((f"#Mask:      {num_rand} ({pct_rand:.1f}%) -> {', '.join(rand_names)}") if num_rand > 0 else "#Mask:      0 (0.0%)")
    print("---------------------------------------------------------------")
    print("原始事件（名称 -> 去归一化前的原始数值）:")
    for name, val_norm, is_carry, is_rand in zip(event_names, val_np.tolist(), mask_np.tolist(), rand_mask_np.tolist()):
        val_orig = _denorm_value(name, float(val_norm))
        tag_parts = []
        if float(is_carry) > 0.5:
            tag_parts.append("carry")
        if float(is_rand) > 0.5:
            tag_parts.append("mask")
        tag = (" [" + ",".join(tag_parts) + "]") if tag_parts else ""
        print(f"  - {name}: {val_orig:.6f}{tag}")
    print("---------------------------------------------------------------")
    print("重构结果（匹配到的事件名 -> 去归一化后的预测数值，余弦相似度）:")
    for (name, pred_val_norm, cos_sim) in assignments:
        pred_orig = _denorm_value(name, float(pred_val_norm))
        tag_parts = []
        if mask_by_name.get(name, 0.0) > 0.5:
            tag_parts.append("carry")
        if rand_by_name.get(name, 0.0) > 0.5:
            tag_parts.append("mask")
        tag = (" [" + ",".join(tag_parts) + "]") if tag_parts else ""
        print(f"  - {name}: {pred_orig:.6f}{tag}  (cos={cos_sim:.3f})")
    print("===============================================================")


if __name__ == "__main__":
    main()

