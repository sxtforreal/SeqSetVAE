#!/usr/bin/env python3
"""Comprehensive SetVAE evaluation suite.

This script assembles posterior-health diagnostics, reconstruction analyses,
and downstream quality checks for the dual-head (Sinkhorn + probabilistic)
SeqSetVAE model.  It follows the evaluation blueprint shared in the
requirements document and exposes a modular API so downstream notebooks can
reuse the individual metrics functions.

Current implementation priorities:
- Posterior collapse alarms (per-dimension KL, MI proxy, latent traversals)
- Latent usage fairness (entropy / Gini)
- Skeleton stubs for advanced analyses (feature probes, OT visualizations,
  subsystem discovery, etc.) so the evaluation contract is already in place.

Several advanced routines still return placeholder values (marked with
"TODO") because they require additional project-specific utilities that
aren't present in this repository snapshot.  The structure, inputs and return
schemas are wired up so that filling in the TODOs later will not require
refactoring the orchestration logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:  # optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - keep optional
    plt = None  # type: ignore

from dataset import _collate_lvcf, _detect_vcols  # type: ignore

try:
    from modules import recon_loss as _set_recon_loss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _set_recon_loss = None  # type: ignore


# ---------------------------------------------------------------------------
# Logging helpers


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


LOGGER = logging.getLogger("eval_setvae")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Schema utilities


@dataclass
class FeatureSchema:
    feature_id: int
    type_code: int  # 0=cont, 1=bin, 2=cat
    name: Optional[str] = None
    cardinality: int = 0


@dataclass
class SchemaBundle:
    features: List[FeatureSchema]

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def feature_types(self) -> List[int]:
        return [f.type_code for f in self.features]

    @property
    def categorical_ids(self) -> List[int]:
        return [f.feature_id for f in self.features if f.type_code == 2]

    @property
    def categorical_cards(self) -> List[int]:
        return [f.cardinality for f in self.features if f.type_code == 2]


def load_schema(path: str) -> SchemaBundle:
    """Load schema metadata from CSV or JSON."""

    ext = Path(path).suffix.lower()
    if ext in {".csv", ""}:
        df = pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported schema format: {path}")

    required = {"feature_id", "type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Schema file missing required columns: {required - set(df.columns)}")

    features: List[FeatureSchema] = []
    type_map = {"cont": 0, "continuous": 0, "bin": 1, "binary": 1, "cat": 2, "categorical": 2}

    for _, row in df.iterrows():
        raw_type = str(row["type"]).lower()
        type_code = type_map.get(raw_type, None)
        if type_code is None:
            raise ValueError(f"Unknown feature type '{row['type']}' in schema")
        feat = FeatureSchema(
            feature_id=int(row["feature_id"]),
            type_code=type_code,
            name=str(row["name"]) if "name" in row and not pd.isna(row["name"]) else None,
            cardinality=int(row["cardinality"]) if "cardinality" in row and not pd.isna(row["cardinality"]) else 0,
        )
        features.append(feat)

    features.sort(key=lambda f: f.feature_id)
    return SchemaBundle(features=features)


# ---------------------------------------------------------------------------
# Checkpoint loading helpers (adapted from seqsetvae_poe.evaluate)


def _load_ckpt_bundle(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw if isinstance(raw, dict) else {}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    hparams: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("hyper_parameters", "hparams"):
            hp = raw.get(key, None)
            if isinstance(hp, dict):
                hparams = hp
                break
    return state, hparams


def _infer_dims_from_state(state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    w = state.get("set_encoder.dim_reducer.weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        R, D = w.shape
        dims["reduced_dim"] = int(R)
        dims["input_dim"] = int(D)
    ew = state.get("set_encoder.embed.0.weight", None)
    if isinstance(ew, torch.Tensor) and ew.dim() == 2:
        ld, embed_in = ew.shape
        dims.setdefault("latent_dim", int(ld))
        dims.setdefault("reduced_dim", int(embed_in))
        dims.setdefault("input_dim", int(embed_in))
    ow = state.get("set_encoder.out.weight", None)
    if isinstance(ow, torch.Tensor) and ow.dim() == 2:
        out_out, ld2 = ow.shape
        dims.setdefault("latent_dim", int(ld2))
        dims.setdefault("reduced_dim", int(out_out))
        dims.setdefault("input_dim", int(out_out))
    mlw = state.get("set_encoder.mu_logvar.4.weight", None)
    if isinstance(mlw, torch.Tensor) and mlw.dim() == 2:
        dims["latent_dim"] = int(mlw.shape[1])
    return dims


def _detect_num_flows(state: Mapping[str, torch.Tensor]) -> int:
    flow_indices: set[int] = set()
    for key in state.keys():
        if ".flows." not in key:
            continue
        parts = key.split(".flows.")
        if len(parts) < 2:
            continue
        suffix = parts[1].split(".")[0]
        if suffix.isdigit():
            flow_indices.add(int(suffix))
    if not flow_indices:
        return 0
    return max(flow_indices) + 1


def _build_model(state: Mapping[str, torch.Tensor], hparams: Mapping[str, Any], schema: SchemaBundle) -> torch.nn.Module:
    try:
        from seqsetvae_poe.model import SetVAEOnlyPretrain  # type: ignore
    except Exception:  # pragma: no cover - fallback when run as script
        from model import SetVAEOnlyPretrain  # type: ignore

    dims = _infer_dims_from_state(state)
    num_flows = _detect_num_flows(state)

    def _hp(name: str, default: Any) -> Any:
        val = hparams.get(name, default)
        if isinstance(val, torch.Tensor):
            try:
                return val.detach().cpu().item()
            except Exception:
                return default
        return val

    model = SetVAEOnlyPretrain(
        input_dim=int(dims.get("input_dim", _hp("input_dim", 768))),
        reduced_dim=int(dims.get("reduced_dim", _hp("reduced_dim", 256))),
        latent_dim=int(dims.get("latent_dim", _hp("latent_dim", 128))),
        levels=int(_hp("levels", 2)),
        heads=int(_hp("heads", 2)),
        m=int(_hp("m", 16)),
        beta=float(_hp("beta", 0.1)),
        lr=float(_hp("lr", 1e-3)),
        warmup_beta=bool(_hp("warmup_beta", True)),
        max_beta=float(_hp("max_beta", 0.2)),
        beta_warmup_steps=int(_hp("beta_warmup_steps", 8000)),
        free_bits=float(_hp("free_bits", 0.05)),
        p_stale=float(_hp("stale_dropout_p", 0.5)),
        p_live=float(_hp("p_live", 0.05)),
        set_mae_ratio=float(_hp("set_mae_ratio", 0.0)),
        small_set_mask_prob=float(_hp("small_set_mask_prob", 0.0)),
        small_set_threshold=int(_hp("small_set_threshold", 5)),
        max_masks_per_set=int(_hp("max_masks_per_set", 0)),
        val_noise_std=float(_hp("val_noise_std", 0.0)),
        dir_noise_std=float(_hp("dir_noise_std", 0.0)),
        train_decoder_noise_std=float(_hp("train_decoder_noise_std", 0.0)),
        eval_decoder_noise_std=float(_hp("eval_decoder_noise_std", 0.0)),
        use_flows=bool(num_flows > 0),
        num_flows=num_flows,
        use_sinkhorn=bool(_hp("use_sinkhorn", True)),
        sinkhorn_eps=float(_hp("sinkhorn_eps", 0.1)),
        sinkhorn_iters=int(_hp("sinkhorn_iters", 100)),
        enable_prob_head=True,
        num_features=schema.num_features,
        feature_types=schema.feature_types,
        categorical_feat_ids=schema.categorical_ids,
        categorical_cardinalities=schema.categorical_cards,
        id_embed_dim=int(_hp("id_embed_dim", 64)),
        state_embed_dim=int(_hp("state_embed_dim", 64)),
        token_mlp_hidden=int(_hp("token_mlp_hidden", 128)),
        prob_head_hidden=int(_hp("prob_head_hidden", 256)),
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning("Loaded state dict with missing=%d unexpected=%d", len(missing), len(unexpected))
    return model


# ---------------------------------------------------------------------------
# Validation set utilities


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_validation_batches(data_dir: str, schema: SchemaBundle, batch_size: int) -> List[Dict[str, torch.Tensor]]:
    """Load validation batches from parquet files stored under <data_dir>/valid."""

    valid_dir = Path(data_dir) / "valid"
    if not valid_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

    files = sorted(valid_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {valid_dir}")

    sample_df = pd.read_parquet(files[0], engine="pyarrow")
    vcols = _detect_vcols(sample_df)
    name_to_id = {f.name: f.feature_id for f in schema.features if f.name is not None}
    name_to_id_or_none = name_to_id if name_to_id else None

    batches: List[Dict[str, torch.Tensor]] = []
    current: List[Tuple[pd.DataFrame, str]] = []

    for path in files:
        df = pd.read_parquet(path, engine="pyarrow")
        pid = path.stem
        current.append((df, pid))
        if len(current) >= batch_size:
            batches.append(_collate_lvcf(current, vcols, name_to_id_or_none))
            current = []

    if current:
        batches.append(_collate_lvcf(current, vcols, name_to_id_or_none))

    return batches


# ---------------------------------------------------------------------------
# Latent statistics helpers


def _extract_set_posteriors(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Collect per-set latent means/logvars using the set encoder."""

    var = batch["var"]
    val = batch["val"]
    minutes = batch["minute"]
    set_id = batch["set_id"]
    padding = batch.get("padding_mask")
    carry = batch.get("carry_mask")
    feat_id = batch.get("feat_id")

    try:
        all_sets = model._split_sets(var, val, minutes, set_id, padding, carry, feat_id)
    except TypeError:
        all_sets = model._split_sets(var, val, minutes, set_id, padding, carry)

    mus: List[torch.Tensor] = []
    logvars: List[torch.Tensor] = []
    for patient_sets in all_sets:
        for s in patient_sets:
            z_list, _ = model.set_encoder.encode_from_var_val(s["var"], s["val"])
            mu = z_list[-1][1].squeeze(1).detach()
            logvar = z_list[-1][2].squeeze(1).detach()
            mus.append(mu)
            logvars.append(logvar)
    return mus, logvars


def _stack_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("No tensors to stack")
    return torch.stack([t.detach().cpu() for t in tensors], dim=0)


def _kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


def _sample_diag_gaussians(mu: torch.Tensor, logvar: torch.Tensor, num_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    noise = torch.randn((num_samples,) + mu.shape, generator=generator, device=mu.device)
    return mu.unsqueeze(0) + noise * std.unsqueeze(0)


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sq_dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)
        k = 0.0
        for bw in bandwidths:
            gamma = 1.0 / (2.0 * (bw ** 2))
            k = k + torch.exp(-gamma * sq_dist)
        return k / float(len(bandwidths))

    k_xx = _kernel(x, x)
    k_yy = _kernel(y, y)
    k_xy = _kernel(x, y)
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return mmd


# ---------------------------------------------------------------------------
# Information gain & self-consistency helpers


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(arr)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))
    arr_sorted = arr[sorter]
    unique_vals, idx_start, counts = np.unique(arr_sorted, return_counts=True, return_index=True)
    ranks = np.zeros(len(arr), dtype=np.float64)
    for start, count in zip(idx_start, counts):
        rank = start + (count - 1) / 2.0
        ranks[start:start + count] = rank
    return ranks[inv]


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = np.sum((rx - rx_mean) * (ry - ry_mean))
    den = np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2))
    if den == 0:
        return float("nan")
    return float(num / den)


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    conc = 0
    disc = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                ties_xy += 1
            elif abs(dx) < 1e-12:
                ties_x += 1
            elif abs(dy) < 1e-12:
                ties_y += 1
            elif dx * dy > 0:
                conc += 1
            elif dx * dy < 0:
                disc += 1
    denom = np.sqrt((conc + disc + ties_x) * (conc + disc + ties_y))
    if denom == 0:
        return float("nan"), float("nan")
    tau = (conc - disc) / denom
    # Normal approximation for p-value
    s = conc - disc
    var_s = (
        (n * (n - 1) * (2 * n + 5)) / 18
        + (ties_x * (ties_x - 1) * (2 * ties_x + 5)) / 18
        + (ties_y * (ties_y - 1) * (2 * ties_y + 5)) / 18
    )
    if var_s <= 0:
        p = float("nan")
    else:
        z = s / np.sqrt(var_s)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(tau), float(p)


def _ks_test_uniform(samples: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    arr = np.sort(arr)
    cdf = np.arange(1, n + 1, dtype=np.float64) / n
    d_plus = np.max(cdf - arr)
    d_minus = np.max(arr - (cdf - 1.0 / n))
    d_stat = max(d_plus, d_minus)
    # Approximate p-value using asymptotic formula
    if n > 0:
        lambda_val = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d_stat
        p = 2.0 * sum(((-1) ** (k - 1)) * math.exp(-2.0 * (lambda_val ** 2) * (k ** 2)) for k in range(1, 10))
        p = max(0.0, min(1.0, p))
    else:
        p = float("nan")
    return float(d_stat), float(p)


def _expected_calibration_error(probs: Sequence[float], labels: Sequence[float], num_bins: int = 10) -> float:
    if len(probs) == 0:
        return float("nan")
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(probs_arr)
    for i in range(num_bins):
        mask = (probs_arr >= bins[i]) & (probs_arr <= bins[i + 1] if i == num_bins - 1 else probs_arr < bins[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        acc = labels_arr[mask].mean()
        conf = probs_arr[mask].mean()
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def _clone_set_dict(s: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in s.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


def _slice_set_by_indices(s: Mapping[str, Any], keep: Sequence[int]) -> Optional[Dict[str, Any]]:
    if len(keep) == 0:
        return None
    out: Dict[str, Any] = {}
    device = None
    if torch.is_tensor(s.get("var", None)):
        device = s["var"].device
    idx_tensor = torch.tensor(keep, dtype=torch.long, device=device)
    for key, value in s.items():
        if value is None:
            out[key] = None
        elif torch.is_tensor(value) and value.dim() >= 2:
            out[key] = value.index_select(1, idx_tensor).clone()
        elif torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


def _encode_set_latent(model: torch.nn.Module, s: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    v_norm, _ = model._compute_target(s)
    if hasattr(model, "_build_token_inputs") and getattr(model, "num_features", 0) > 0:
        x_input = model._build_token_inputs(v_norm, s)
    else:
        x_input = v_norm * s["val"]
    z_list, _ = model.set_encoder.encode(x_input)
    mu = z_list[-1][1].squeeze(1)
    logvar = z_list[-1][2].squeeze(1)
    return mu, logvar


def _predict_head_outputs(model: torch.nn.Module, z_mu: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if not getattr(model, "enable_prob_head", False):
        return None
    if getattr(model, "prob_shared", None) is None:
        return None
    h = model.prob_shared(z_mu)
    cont_mu = model.prob_cont_mu(h)
    cont_logvar = model.prob_cont_logvar(h)
    bin_logit = model.prob_bin_logit(h)
    cat_logits = model.prob_cat_logits(h) if getattr(model, "prob_cat_logits", None) is not None else None
    return cont_mu, cont_logvar, bin_logit, cat_logits


def _compute_uncertainty_scalar(model: torch.nn.Module, head_outputs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]) -> float:
    if head_outputs is None or getattr(model, "num_features", 0) <= 0:
        return float("nan")
    cont_mu, cont_logvar, bin_logit, cat_logits = head_outputs
    feature_types = getattr(model, "feature_types", None)
    if feature_types is None:
        return float("nan")
    scores: List[float] = []
    cont_var = torch.exp(cont_logvar)
    bin_prob = torch.sigmoid(bin_logit)
    for fid in range(model.num_features):
        tcode = int(feature_types[fid].item())
        if tcode == 0:
            scores.append(float(cont_var[0, fid].item()))
        elif tcode == 1:
            p = float(bin_prob[0, fid].item())
            scores.append(p * (1.0 - p))
        elif tcode == 2 and cat_logits is not None:
            offset = int(model.cat_offsets[fid].item()) if hasattr(model, "cat_offsets") else -1
            card = int(model.cat_cardinalities[fid].item()) if hasattr(model, "cat_cardinalities") else 0
            if offset >= 0 and card > 0:
                logits = cat_logits[0, offset : offset + card]
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)))
                scores.append(float(entropy.item()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _iterate_sets(model: torch.nn.Module, batches: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> Iterable[Dict[str, Any]]:
    for batch in batches:
        batch_dev = _to_device(batch, device)
        try:
            all_sets = model._split_sets(
                batch_dev["var"],
                batch_dev["val"],
                batch_dev["minute"],
                batch_dev["set_id"],
                batch_dev.get("padding_mask"),
                batch_dev.get("carry_mask"),
                batch_dev.get("feat_id"),
            )
        except TypeError:
            all_sets = model._split_sets(
                batch_dev["var"],
                batch_dev["val"],
                batch_dev["minute"],
                batch_dev["set_id"],
                batch_dev.get("padding_mask"),
                batch_dev.get("carry_mask"),
            )
        for patient_sets in all_sets:
            for s in patient_sets:
                yield s


# ---------------------------------------------------------------------------
# Evaluation functions (aligned with spec)


def evaluate_posterior_health(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    n_z_samples: int,
    save_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    LOGGER.info("Evaluating posterior health on %d batches", len(batches))
    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    kl_vectors: List[torch.Tensor] = []
    per_set_kl: List[float] = []
    mus_all: List[torch.Tensor] = []
    logvars_all: List[torch.Tensor] = []

    for batch in batches:
        batch_dev = _to_device(batch, device)
        with torch.no_grad():
            if hasattr(model, "_compute_kl_dim_stats"):
                kl_dim = model._compute_kl_dim_stats(batch_dev)
                kl_vectors.append(kl_dim.detach().cpu())
                per_set_kl.append(float(kl_dim.sum().item()))
            mus, logvars = _extract_set_posteriors(model, batch_dev)
            if mus:
                mus_all.extend(mus)
                logvars_all.extend(logvars)

    if not kl_vectors:
        raise RuntimeError("No KL statistics collected; ensure batches are non-empty")

    kl_matrix = torch.stack(kl_vectors, dim=0)
    kl_mean = kl_matrix.mean(dim=0)
    kl_std = kl_matrix.std(dim=0, unbiased=False)
    kl_total = float(kl_mean.sum().item())

    eps = 1e-8
    active_dims = {
        "@0.1": int((kl_mean > 0.1).sum().item()),
        "@0.05": int((kl_mean > 0.05).sum().item()),
        "ratio@0.1": float((kl_mean > 0.1).sum().item()) / max(1, kl_mean.numel()),
    }

    # Cumulative coverage for plotting
    sorted_kl, _ = torch.sort(kl_mean, descending=True)
    cumulative = torch.cumsum(sorted_kl, dim=0)
    coverage = (cumulative / (sorted_kl.sum() + eps)).detach().cpu().numpy()

    mu_stats: Dict[str, Any] = {}
    logvar_stats: Dict[str, Any] = {}
    collapse_thresholds = {
        "kl": 0.01,
        "mu_abs": 0.05,
        "var_dev": 0.05,
    }
    collapse_report: Dict[str, Any] = {
        "collapsed_dims": [],
        "collapsed_ratio": 0.0,
        "thresholds": collapse_thresholds,
    }

    # Mutual information proxy & latent diagnostics
    if mus_all:
        mu_stack = _stack_tensors(mus_all)
        logvar_stack = _stack_tensors(logvars_all)
        kl_per_set = _kl_diag_gaussian(mu_stack, logvar_stack).sum(dim=1)
        mi_first = float(kl_per_set.mean().item())

        samples_per_set = min(max(1, n_z_samples), 32)
        q_samples = _sample_diag_gaussians(mu_stack, logvar_stack, samples_per_set, generator=rng)
        q_samples_flat = q_samples.reshape(-1, mu_stack.size(-1))
        p_samples = torch.randn_like(q_samples_flat, generator=rng)
        mmd = _mmd_rbf(q_samples_flat, p_samples, bandwidths=(0.2, 0.5, 1.0, 2.0, 5.0))
        mi_proxy = float(mi_first - mmd.item())

        mu_mean = mu_stack.mean(dim=0)
        mu_std = mu_stack.std(dim=0, unbiased=False)
        mu_abs_mean = mu_stack.abs().mean(dim=0)
        logvar_mean = logvar_stack.mean(dim=0)
        logvar_std = logvar_stack.std(dim=0, unbiased=False)
        var_stack = torch.exp(logvar_stack)
        var_mean = var_stack.mean(dim=0)
        var_std = var_stack.std(dim=0, unbiased=False)

        mu_np = mu_abs_mean.detach().cpu().numpy()
        var_mean_np = var_mean.detach().cpu().numpy()

        mu_stats = {
            "mean_per_dim": mu_mean.detach().cpu().tolist(),
            "std_per_dim": mu_std.detach().cpu().tolist(),
            "abs_mean_per_dim": mu_abs_mean.detach().cpu().tolist(),
            "abs_mean_global": float(mu_np.mean()) if mu_np.size > 0 else float("nan"),
            "abs_mean_p95": float(np.percentile(mu_np, 95)) if mu_np.size > 0 else float("nan"),
        }
        logvar_stats = {
            "mean_per_dim": logvar_mean.detach().cpu().tolist(),
            "std_per_dim": logvar_std.detach().cpu().tolist(),
            "var_mean_per_dim": var_mean.detach().cpu().tolist(),
            "var_std_per_dim": var_std.detach().cpu().tolist(),
            "var_mean_global": float(var_mean_np.mean()) if var_mean_np.size > 0 else float("nan"),
        }

        collapse_mask = (
            (kl_mean < 0.01)
            & (mu_abs_mean < 0.05)
            & (var_mean - 1.0).abs() < 0.05
        )
        collapsed_idx = collapse_mask.nonzero(as_tuple=False).view(-1).tolist()
        collapse_report = {
            "collapsed_dims": collapsed_idx,
            "collapsed_ratio": float(len(collapsed_idx) / max(1, kl_mean.numel())),
            "thresholds": collapse_thresholds,
            "suspicious_dims": (
                (kl_mean < 0.02).nonzero(as_tuple=False).view(-1).tolist()
            ),
            "per_set_kl_stats": {
                "mean": float(kl_per_set.mean().item()),
                "std": float(kl_per_set.std(unbiased=False).item()),
            },
        }
    else:
        mi_proxy = float("nan")

    results: Dict[str, Any] = {
        "KL_per_dim_mean": kl_mean.detach().cpu().tolist(),
        "KL_per_dim_std": kl_std.detach().cpu().tolist(),
        "KL_total": kl_total,
        "ActiveDims": active_dims,
        "coverage_curve": coverage.tolist(),
        "MI_proxy": mi_proxy,
        "KL_samples": per_set_kl,
        "mu_summary": mu_stats,
        "logvar_summary": logvar_stats,
        "collapse_report": collapse_report,
    }

    if plt is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax_primary = plt.subplots(figsize=(10, 4))
        dims = np.arange(len(kl_mean))
        ax_primary.bar(dims, kl_mean.numpy(), color="#2a9d8f")
        ax_primary.set_xlabel("Latent dimension")
        ax_primary.set_ylabel("KL mean (nats)")
        ax_secondary = ax_primary.twinx()
        ax_secondary.plot(dims, coverage, color="#e76f51", marker="o", linewidth=2)
        ax_secondary.set_ylabel("Cumulative KL share")
        ax_secondary.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(save_dir / "posterior_kl_per_dim.png", dpi=200)
        plt.close(fig)

    top_dims = torch.argsort(kl_mean, descending=True)[: min(10, kl_mean.numel())].tolist()
    results["top_dimensions"] = [
        {
            "dim": int(idx),
            "kl": float(kl_mean[idx].item()),
            "kl_std": float(kl_std[idx].item()),
            "mu_abs_mean": float(
                mu_stats.get("abs_mean_per_dim", [float("nan")])[idx]
            )
            if mu_stats and idx < len(mu_stats.get("abs_mean_per_dim", []))
            else float("nan"),
        }
        for idx in top_dims
    ]

    return results


def evaluate_information_distribution(
    kl_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    kl_mean_vec = torch.tensor(kl_stats["KL_per_dim_mean"])
    total = float(kl_mean_vec.sum().item())
    if total <= 0:
        return {"entropy": float("nan"), "entropy_norm": float("nan"), "gini": float("nan")}

    p = (kl_mean_vec / total).numpy()
    eps = 1e-12
    entropy = float(-np.sum(p * np.log(p + eps)))
    max_entropy = math.log(len(p)) if len(p) > 0 else 1.0
    gini = float(1.0 - np.sum(np.minimum.outer(p, p)) * 2.0)
    herfindahl = float(np.sum(p ** 2))
    effective_rank = float(math.exp(entropy)) if entropy > -float("inf") else float("nan")
    sorted_mass = np.sort(p)[::-1]
    top1 = float(sorted_mass[0]) if sorted_mass.size > 0 else float("nan")
    top5 = float(np.sum(sorted_mass[:5])) if sorted_mass.size >= 5 else float(np.sum(sorted_mass))
    cumulative = np.cumsum(sorted_mass)
    half_mass_idx = int(np.searchsorted(cumulative, 0.5)) if cumulative.size > 0 else -1
    kl_np = kl_mean_vec.detach().cpu().numpy()

    return {
        "entropy": entropy,
        "entropy_norm": float(entropy / max_entropy) if max_entropy > 0 else float("nan"),
        "gini": gini,
        "herfindahl_index": herfindahl,
        "effective_rank": effective_rank,
        "top1_mass": top1,
        "top5_mass": top5,
        "dims_for_50pct_kl": int(half_mass_idx + 1) if half_mass_idx >= 0 else -1,
        "kl_percentiles": {
            "p50": float(np.percentile(kl_np, 50)) if kl_np.size > 0 else float("nan"),
            "p90": float(np.percentile(kl_np, 90)) if kl_np.size > 0 else float("nan"),
            "p99": float(np.percentile(kl_np, 99)) if kl_np.size > 0 else float("nan"),
        },
    }


def evaluate_reconstruction_quality(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    max_sets: int = 128,
) -> Dict[str, Any]:
    encoder = getattr(model, "set_encoder", None)
    if encoder is None:
        return {"status": "skipped", "reason": "model missing set encoder"}

    feature_types_tensor = getattr(model, "feature_types", None)
    feature_types_list: Optional[List[int]] = None
    if feature_types_tensor is not None:
        feature_types_list = [int(x) for x in torch.flatten(feature_types_tensor).tolist()]

    cat_offsets_tensor = getattr(model, "cat_offsets", None)
    cat_cards_tensor = getattr(model, "cat_cardinalities", None)
    cat_offsets: Optional[List[int]] = None
    cat_cards: Optional[List[int]] = None
    if cat_offsets_tensor is not None:
        cat_offsets = [int(x) for x in torch.flatten(cat_offsets_tensor).tolist()]
    if cat_cards_tensor is not None:
        cat_cards = [int(x) for x in torch.flatten(cat_cards_tensor).tolist()]

    rec_losses: List[float] = []
    token_mae: List[float] = []
    token_cos: List[float] = []
    magnitude_pairs: List[Tuple[float, float]] = []

    cont_abs_err: List[float] = []
    cont_sq_err: List[float] = []
    cont_nll: List[float] = []
    cont_ci_hits: List[int] = []

    bin_probs: List[float] = []
    bin_labels: List[float] = []
    bin_nll: List[float] = []
    bin_correct: List[int] = []

    cat_true_prob: List[float] = []
    cat_nll: List[float] = []
    cat_correct: List[int] = []

    per_feature_scores: Dict[int, List[float]] = defaultdict(list)

    processed = 0
    for set_dict in _iterate_sets(model, batches, device):
        if processed >= max_sets:
            break

        var = set_dict.get("var")
        val = set_dict.get("val")
        feat_id = set_dict.get("feat_id")
        if var is None or val is None:
            continue

        with torch.no_grad():
            var = var.to(device)
            val = val.to(device)
            if getattr(encoder, "dim_reducer", None) is not None:
                reduced = encoder.dim_reducer(var)
            else:
                reduced = var
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            x_target = (reduced / (norms + 1e-8)) * val
            z_list, _ = encoder.encode(x_target)
            recon = encoder.decode(
                z_list,
                target_n=int(x_target.size(1)),
                use_mean=True,
                noise_std=0.0,
            )

        if recon is None:
            continue

        if _set_recon_loss is not None:
            try:
                rec = float(_set_recon_loss(recon, x_target).item())
            except Exception:
                rec = float(torch.mean((recon - x_target) ** 2).item())
        else:
            rec = float(torch.mean((recon - x_target) ** 2).item())
        rec_losses.append(rec)

        per_token_mae = torch.norm(recon - x_target, p=1, dim=-1)
        token_mae.append(float(per_token_mae.mean().item()))
        token_cos.append(float(F.cosine_similarity(recon, x_target, dim=-1).mean().item()))
        magnitudes_real = torch.norm(x_target, p=2, dim=-1)
        magnitudes_recon = torch.norm(recon, p=2, dim=-1)
        magnitude_pairs.extend(
            (
                float(magnitudes_real[0, i].item()),
                float(magnitudes_recon[0, i].item()),
            )
            for i in range(magnitudes_real.size(1))
        )

        with torch.no_grad():
            mu_full, _ = _encode_set_latent(model, set_dict)
            head_outputs = _predict_head_outputs(model, mu_full)

        if head_outputs is not None and feature_types_list is not None and feat_id is not None:
            cont_mu, cont_logvar, bin_logit, cat_logits = head_outputs
            num_tokens = int(val.size(1))
            for i in range(num_tokens):
                fid = int(feat_id[0, i, 0].item()) if feat_id is not None else i
                if fid >= len(feature_types_list):
                    continue
                tcode = feature_types_list[fid]
                value = float(val[0, i, 0].item())

                if tcode == 0:
                    mu_pred = float(cont_mu[0, fid].item())
                    logvar = float(cont_logvar[0, fid].item())
                    var = max(math.exp(logvar), 1e-8)
                    sigma = math.sqrt(var)
                    abs_err = abs(value - mu_pred)
                    cont_abs_err.append(abs_err)
                    cont_sq_err.append((value - mu_pred) ** 2)
                    cont_nll.append(0.5 * (math.log(2 * math.pi) + logvar + ((value - mu_pred) ** 2) / var))
                    ci_hit = 1 if abs(value - mu_pred) <= 1.96 * sigma else 0
                    cont_ci_hits.append(ci_hit)
                    per_feature_scores[fid].append(abs_err)
                elif tcode == 1:
                    prob = float(torch.sigmoid(bin_logit[0, fid]).item())
                    label = 1.0 if value > 0.5 else 0.0
                    bin_probs.append(prob)
                    bin_labels.append(label)
                    eps = 1e-8
                    bin_nll.append(-(label * math.log(max(prob, eps)) + (1 - label) * math.log(max(1 - prob, eps))))
                    pred = 1.0 if prob >= 0.5 else 0.0
                    bin_correct.append(1 if pred == label else 0)
                    per_feature_scores[fid].append(abs(prob - label))
                elif tcode == 2 and cat_logits is not None and cat_offsets is not None and cat_cards is not None:
                    offset = cat_offsets[fid] if fid < len(cat_offsets) else -1
                    card = cat_cards[fid] if fid < len(cat_cards) else 0
                    if offset < 0 or card <= 0:
                        continue
                    logits = cat_logits[0, offset : offset + card]
                    probs = torch.softmax(logits, dim=-1)
                    klass = int(round(value))
                    klass = max(0, min(card - 1, klass))
                    prob_true = float(probs[klass].item())
                    cat_true_prob.append(prob_true)
                    eps = 1e-12
                    cat_nll.append(-math.log(max(prob_true, eps)))
                    pred_class = int(torch.argmax(probs).item())
                    cat_correct.append(1 if pred_class == klass else 0)
                    per_feature_scores[fid].append(1.0 - prob_true)

        processed += 1

    if processed == 0:
        return {"status": "skipped", "reason": "no evaluable sets"}

    save_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None and magnitude_pairs:
        xs, ys = zip(*magnitude_pairs[:2000])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xs, ys, alpha=0.4, s=10, color="#2a9d8f")
        ax.plot([min(xs + ys), max(xs + ys)], [min(xs + ys), max(xs + ys)], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("||x_target||_2")
        ax.set_ylabel("||recon||_2")
        ax.set_title("Set-level reconstruction magnitude comparison")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(save_dir / "reconstruction_magnitude_scatter.png", dpi=220)
        plt.close(fig)

    sinkhorn_summary = {
        "reconstruction_loss_mean": float(np.mean(rec_losses)) if rec_losses else float("nan"),
        "reconstruction_loss_std": float(np.std(rec_losses)) if rec_losses else float("nan"),
        "token_mae_mean": float(np.mean(token_mae)) if token_mae else float("nan"),
        "token_cosine_mean": float(np.mean(token_cos)) if token_cos else float("nan"),
    }

    prob_head_summary: Dict[str, Any] = {"available": False}
    if cont_abs_err or bin_probs or cat_true_prob:
        prob_head_summary = {
            "available": True,
            "continuous": {
                "rmse": float(math.sqrt(np.mean(cont_sq_err))) if cont_sq_err else float("nan"),
                "mae": float(np.mean(cont_abs_err)) if cont_abs_err else float("nan"),
                "nll": float(np.mean(cont_nll)) if cont_nll else float("nan"),
                "ci95_hit_rate": float(np.mean(cont_ci_hits)) if cont_ci_hits else float("nan"),
            },
            "binary": {
                "accuracy": float(np.mean(bin_correct)) if bin_correct else float("nan"),
                "nll": float(np.mean(bin_nll)) if bin_nll else float("nan"),
                "ece": _expected_calibration_error(bin_probs, bin_labels) if bin_probs else float("nan"),
            },
            "categorical": {
                "accuracy": float(np.mean(cat_correct)) if cat_correct else float("nan"),
                "nll": float(np.mean(cat_nll)) if cat_nll else float("nan"),
            },
            "feature_rankings": sorted(
                (
                    {
                        "feature_id": int(fid),
                        "mean_abs_error": float(np.mean(vals)),
                        "count": len(vals),
                    }
                    for fid, vals in per_feature_scores.items()
                    if vals
                ),
                key=lambda item: item["mean_abs_error"],
                reverse=True,
            )[:20],
        }

    return {
        "status": "ok",
        "sets_evaluated": processed,
        "sinkhorn": sinkhorn_summary,
        "probabilistic_head": prob_head_summary,
    }


def evaluate_conditional_inference(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    mask_ratios: Sequence[float] = (0.2, 0.5, 0.8),
    per_ratio_sets: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    feature_types_tensor = getattr(model, "feature_types", None)
    if feature_types_tensor is None or getattr(model, "enable_prob_head", False) is False:
        return {"status": "skipped", "reason": "probabilistic head or feature schema unavailable"}

    feature_types_list = [int(x) for x in torch.flatten(feature_types_tensor).tolist()]
    cat_offsets_tensor = getattr(model, "cat_offsets", None)
    cat_cards_tensor = getattr(model, "cat_cardinalities", None)
    cat_offsets = [int(x) for x in torch.flatten(cat_offsets_tensor).tolist()] if cat_offsets_tensor is not None else []
    cat_cards = [int(x) for x in torch.flatten(cat_cards_tensor).tolist()] if cat_cards_tensor is not None else []

    ratios = [float(max(1e-3, min(0.95, r))) for r in mask_ratios]
    rng = np.random.default_rng(seed)

    ratio_state: Dict[float, Dict[str, Any]] = {}
    for r in ratios:
        ratio_state[r] = {
            "sets": 0,
            "cont_abs_err": [],
            "cont_sq_err": [],
            "cont_nll": [],
            "cont_ci_hits": [],
            "bin_probs": [],
            "bin_labels": [],
            "bin_nll": [],
            "bin_correct": [],
            "cat_true_prob": [],
            "cat_nll": [],
            "cat_correct": [],
        }

    def _all_done() -> bool:
        return all(state["sets"] >= per_ratio_sets for state in ratio_state.values())

    for set_dict in _iterate_sets(model, batches, device):
        if _all_done():
            break
        feat = set_dict.get("feat_id")
        val = set_dict.get("val")
        if feat is None or val is None:
            continue
        num_tokens = int(val.size(1))
        if num_tokens < 2:
            continue

        order = rng.permutation(num_tokens).tolist()

        for r in ratios:
            state = ratio_state[r]
            if state["sets"] >= per_ratio_sets:
                continue

            mask_cnt = max(1, int(round(r * num_tokens)))
            if mask_cnt >= num_tokens:
                mask_cnt = num_tokens - 1
            if mask_cnt <= 0:
                continue

            masked_idx = sorted(order[:mask_cnt])
            observed_idx = sorted(order[mask_cnt:])
            observed = _slice_set_by_indices(set_dict, observed_idx)
            if observed is None:
                continue

            with torch.no_grad():
                mu_cond, _ = _encode_set_latent(model, observed)
                head_cond = _predict_head_outputs(model, mu_cond)

            if head_cond is None:
                continue

            cont_mu, cont_logvar, bin_logit, cat_logits = head_cond
            skip_feature = False
            for idx in observed_idx:
                fid = int(feat[0, idx, 0].item())
                if fid >= len(feature_types_list):
                    skip_feature = True
                    break
            if skip_feature:
                continue

            for idx in masked_idx:
                fid = int(feat[0, idx, 0].item())
                if fid >= len(feature_types_list):
                    continue
                tcode = feature_types_list[fid]
                value = float(val[0, idx, 0].item())

                if tcode == 0:
                    mu_pred = float(cont_mu[0, fid].item())
                    logvar = float(cont_logvar[0, fid].item())
                    var = max(math.exp(logvar), 1e-8)
                    sigma = math.sqrt(var)
                    state["cont_abs_err"].append(abs(value - mu_pred))
                    state["cont_sq_err"].append((value - mu_pred) ** 2)
                    state["cont_nll"].append(0.5 * (math.log(2 * math.pi) + logvar + ((value - mu_pred) ** 2) / var))
                    state["cont_ci_hits"].append(1 if abs(value - mu_pred) <= 1.96 * sigma else 0)
                elif tcode == 1:
                    prob = float(torch.sigmoid(bin_logit[0, fid]).item())
                    label = 1.0 if value > 0.5 else 0.0
                    eps = 1e-8
                    state["bin_probs"].append(prob)
                    state["bin_labels"].append(label)
                    state["bin_nll"].append(-(label * math.log(max(prob, eps)) + (1 - label) * math.log(max(1 - prob, eps))))
                    pred = 1.0 if prob >= 0.5 else 0.0
                    state["bin_correct"].append(1 if pred == label else 0)
                elif tcode == 2 and cat_logits is not None:
                    offset = cat_offsets[fid] if fid < len(cat_offsets) else -1
                    card = cat_cards[fid] if fid < len(cat_cards) else 0
                    if offset < 0 or card <= 0:
                        continue
                    logits = cat_logits[0, offset : offset + card]
                    probs = torch.softmax(logits, dim=-1)
                    klass = int(round(value))
                    klass = max(0, min(card - 1, klass))
                    prob_true = float(probs[klass].item())
                    eps = 1e-12
                    state["cat_true_prob"].append(prob_true)
                    state["cat_nll"].append(-math.log(max(prob_true, eps)))
                    pred_class = int(torch.argmax(probs).item())
                    state["cat_correct"].append(1 if pred_class == klass else 0)

            state["sets"] += 1

    save_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {}
    summary_curve: List[Tuple[float, float]] = []
    for r in ratios:
        state = ratio_state[r]
        if state["sets"] == 0:
            metrics[str(r)] = {"status": "insufficient_data"}
            continue
        cont_rmse = float(math.sqrt(np.mean(state["cont_sq_err"]))) if state["cont_sq_err"] else float("nan")
        cont_mae = float(np.mean(state["cont_abs_err"])) if state["cont_abs_err"] else float("nan")
        cont_nll = float(np.mean(state["cont_nll"])) if state["cont_nll"] else float("nan")
        cont_ci = float(np.mean(state["cont_ci_hits"])) if state["cont_ci_hits"] else float("nan")

        bin_acc = float(np.mean(state["bin_correct"])) if state["bin_correct"] else float("nan")
        bin_nll = float(np.mean(state["bin_nll"])) if state["bin_nll"] else float("nan")
        bin_ece = _expected_calibration_error(state["bin_probs"], state["bin_labels"]) if state["bin_probs"] else float("nan")

        cat_acc = float(np.mean(state["cat_correct"])) if state["cat_correct"] else float("nan")
        cat_nll = float(np.mean(state["cat_nll"])) if state["cat_nll"] else float("nan")

        metrics[str(r)] = {
            "sets_evaluated": state["sets"],
            "continuous": {
                "rmse": cont_rmse,
                "mae": cont_mae,
                "nll": cont_nll,
                "ci95_hit_rate": cont_ci,
            },
            "binary": {
                "accuracy": bin_acc,
                "nll": bin_nll,
                "ece": bin_ece,
            },
            "categorical": {
                "accuracy": cat_acc,
                "nll": cat_nll,
            },
        }

        summary_curve.append((r, cont_rmse if math.isfinite(cont_rmse) else float("nan")))

    if plt is not None and summary_curve:
        xs = [pt[0] for pt in summary_curve]
        ys = [pt[1] for pt in summary_curve]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o", color="#e76f51")
        ax.set_xlabel("mask ratio")
        ax.set_ylabel("continuous RMSE")
        ax.set_title("Conditional inference degradation curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "conditional_inference_rmse.png", dpi=220)
        plt.close(fig)

    return {
        "status": "ok",
        "mask_ratios": [float(r) for r in ratios],
        "metrics": metrics,
    }


def evaluate_information_gain_monotonicity(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    top_k: int,
    max_sets: int,
) -> Dict[str, Any]:
    if getattr(model, "feature_types", None) is None or getattr(model, "enable_prob_head", False) is False:
        return {"status": "skipped", "reason": "probabilistic head or feature schema unavailable"}

    curves: List[List[float]] = []
    metrics_per_set: List[Dict[str, Any]] = []
    processed = 0
    monotonic_violations: List[float] = []

    for set_dict in _iterate_sets(model, batches, device):
        feat = set_dict.get("feat_id")
        if feat is None:
            continue
        num_tokens = int(feat.size(1))
        if num_tokens < 2:
            continue
        with torch.no_grad():
            mu_ref, _ = _encode_set_latent(model, set_dict)
            head_ref = _predict_head_outputs(model, mu_ref)
            base_unc = _compute_uncertainty_scalar(model, head_ref)
        if not math.isfinite(base_unc):
            continue

        contributions: List[float] = []
        keep_all = list(range(num_tokens))
        with torch.no_grad():
            for idx in range(num_tokens):
                keep = [k for k in keep_all if k != idx]
                reduced = _slice_set_by_indices(set_dict, keep)
                if reduced is None:
                    contributions.append(float("nan"))
                    continue
                mu_i, _ = _encode_set_latent(model, reduced)
                head_i = _predict_head_outputs(model, mu_i)
                unc_i = _compute_uncertainty_scalar(model, head_i)
                if not math.isfinite(unc_i):
                    contributions.append(float("nan"))
                else:
                    contributions.append(float(unc_i - base_unc))

        valid_indices = [i for i, c in enumerate(contributions) if math.isfinite(c)]
        if not valid_indices:
            continue
        ranking = sorted(valid_indices, key=lambda i: contributions[i], reverse=True)
        steps = min(top_k, len(ranking))

        s_curve: List[float] = [1.0]
        current_keep = list(range(num_tokens))
        with torch.no_grad():
            for step in range(steps):
                remove_idx = ranking[step]
                if remove_idx not in current_keep:
                    continue
                current_keep.remove(remove_idx)
                reduced = _slice_set_by_indices(set_dict, current_keep)
                if reduced is None:
                    break
                mu_k, _ = _encode_set_latent(model, reduced)
                cos_sim = F.cosine_similarity(mu_ref.unsqueeze(0), mu_k.unsqueeze(0), dim=-1).item()
                cos_sim = float(max(min(cos_sim, 1.0), -1.0))
                s_curve.append(cos_sim)

        if len(s_curve) <= 1:
            continue

        k_values = list(range(len(s_curve)))
        spearman = _spearman_corr(k_values, s_curve)
        tau, tau_p = _kendall_tau_b(k_values, s_curve)
        auc_decay = float(np.mean(s_curve))

        if len(s_curve) > 1:
            violations = 0
            for idx in range(1, len(s_curve)):
                if s_curve[idx] > s_curve[idx - 1] + 1e-4:
                    violations += 1
            monotonic_violations.append(violations / max(1, len(s_curve) - 1))

        def _delta_at(k: int) -> float:
            if k < len(s_curve):
                return float(s_curve[0] - s_curve[k])
            return float("nan")

        metrics_per_set.append({
            "num_tokens": num_tokens,
            "S_curve": s_curve,
            "spearman": spearman,
            "kendall_tau": tau,
            "kendall_tau_p": tau_p,
            "AUC_decay": auc_decay,
            "delta@1": _delta_at(1),
            "delta@3": _delta_at(3),
            "delta@5": _delta_at(5),
            "top_indices": ranking[:steps],
            "top_contributions": [float(contributions[i]) for i in ranking[:steps]],
        })
        curves.append(s_curve)
        processed += 1
        if processed >= max_sets:
            break

    if not curves:
        return {"status": "skipped", "reason": "no eligible sets"}

    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(curves):
        padded[i, : len(curve)] = curve
    mean_curve = np.nanmean(padded, axis=0)
    std_curve = np.nanstd(padded, axis=0)

    tau_vals = [m["kendall_tau"] for m in metrics_per_set if math.isfinite(m["kendall_tau"])]
    spearman_vals = [m["spearman"] for m in metrics_per_set if math.isfinite(m["spearman"])]
    auc_vals = [m["AUC_decay"] for m in metrics_per_set if math.isfinite(m["AUC_decay"])]

    summary = {
        "sets_evaluated": processed,
        "mean_curve": mean_curve.tolist(),
        "std_curve": std_curve.tolist(),
        "avg_kendall_tau": float(np.nanmean(tau_vals) if tau_vals else float("nan")),
        "avg_spearman": float(np.nanmean(spearman_vals) if spearman_vals else float("nan")),
        "avg_auc_decay": float(np.nanmean(auc_vals) if auc_vals else float("nan")),
        "monotonic_violation_rate": float(np.mean(monotonic_violations)) if monotonic_violations else float("nan"),
        "kendall_tau_range": [
            float(np.nanmin(tau_vals)) if tau_vals else float("nan"),
            float(np.nanmax(tau_vals)) if tau_vals else float("nan"),
        ],
    }

    return {
        "status": "ok",
        "summary": summary,
        "per_set": metrics_per_set,
    }


def evaluate_active_measurement_value(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    max_sets: int = 256,
) -> Dict[str, Any]:
    if getattr(model, "enable_prob_head", False) is False or getattr(model, "prob_shared", None) is None:
        return {"status": "skipped", "reason": "probabilistic head unavailable"}

    feature_scores: Dict[int, List[float]] = defaultdict(list)
    set_summaries: List[Dict[str, Any]] = []

    processed = 0
    for set_idx, set_dict in enumerate(_iterate_sets(model, batches, device)):
        if processed >= max_sets:
            break

        feat = set_dict.get("feat_id")
        if feat is None:
            continue
        num_tokens = int(feat.size(1))
        if num_tokens == 0:
            continue

        with torch.no_grad():
            mu_ref, _ = _encode_set_latent(model, set_dict)
            head_ref = _predict_head_outputs(model, mu_ref)
        base_unc = _compute_uncertainty_scalar(model, head_ref)
        if not math.isfinite(base_unc):
            continue

        token_scores: List[Tuple[int, float]] = []
        keep_all = list(range(num_tokens))
        with torch.no_grad():
            for idx in range(num_tokens):
                fid = int(feat[0, idx, 0].item())
                keep = [k for k in keep_all if k != idx]
                reduced = _slice_set_by_indices(set_dict, keep)
                if reduced is None:
                    continue
                mu_i, _ = _encode_set_latent(model, reduced)
                head_i = _predict_head_outputs(model, mu_i)
                unc_i = _compute_uncertainty_scalar(model, head_i)
                if not math.isfinite(unc_i):
                    continue
                delta = float(unc_i - base_unc)
                feature_scores[fid].append(delta)
                token_scores.append((fid, delta))

        if token_scores:
            token_scores.sort(key=lambda item: item[1], reverse=True)
            set_summaries.append({
                "set_index": set_idx,
                "base_uncertainty": base_unc,
                "top_contributors": token_scores[: min(5, len(token_scores))],
                "all_contributors": token_scores,
            })
            processed += 1

    if processed == 0:
        return {"status": "skipped", "reason": "no eligible sets"}

    save_dir.mkdir(parents=True, exist_ok=True)

    feature_rankings = sorted(
        (
            {
                "feature_id": int(fid),
                "mean_delta": float(np.mean(vals)),
                "p90_delta": float(np.percentile(vals, 90)) if len(vals) > 0 else float("nan"),
                "count": len(vals),
            }
            for fid, vals in feature_scores.items()
            if vals
        ),
        key=lambda item: item["mean_delta"],
        reverse=True,
    )

    if plt is not None and feature_rankings:
        topk = feature_rankings[: min(15, len(feature_rankings))]
        xs = [str(item["feature_id"]) for item in topk]
        ys = [item["mean_delta"] for item in topk]
        fig, ax = plt.subplots(figsize=(max(6, len(xs) * 0.4), 4))
        ax.bar(xs, ys, color="#264653")
        ax.set_xlabel("feature id")
        ax.set_ylabel("delta uncertainty")
        ax.set_title("Active measurement value (top features)")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(save_dir / "active_measurement_top_features.png", dpi=220)
        plt.close(fig)

    return {
        "status": "ok",
        "sets_evaluated": processed,
        "feature_ranking": feature_rankings,
        "per_set": set_summaries,
    }


def evaluate_subsystem_discovery(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    max_sets: int = 512,
    corr_threshold: float = 0.6,
) -> Dict[str, Any]:
    num_features = getattr(model, "num_features", 0)
    feature_types_tensor = getattr(model, "feature_types", None)
    if num_features <= 0 or feature_types_tensor is None:
        return {"status": "skipped", "reason": "model missing feature schema"}

    feature_types_list = [int(x) for x in torch.flatten(feature_types_tensor).tolist()]
    cat_offsets_tensor = getattr(model, "cat_offsets", None)
    cat_cards_tensor = getattr(model, "cat_cardinalities", None)
    cat_offsets = [int(x) for x in torch.flatten(cat_offsets_tensor).tolist()] if cat_offsets_tensor is not None else []
    cat_cards = [int(x) for x in torch.flatten(cat_cards_tensor).tolist()] if cat_cards_tensor is not None else []

    responses: List[np.ndarray] = []
    latents: List[np.ndarray] = []

    for set_dict in _iterate_sets(model, batches, device):
        if len(responses) >= max_sets:
            break

        with torch.no_grad():
            mu, _ = _encode_set_latent(model, set_dict)
            head = _predict_head_outputs(model, mu)
        if head is None:
            continue

        cont_mu, cont_logvar, bin_logit, cat_logits = head
        resp = torch.zeros((num_features,), device=mu.device, dtype=torch.float32)
        for fid in range(num_features):
            tcode = feature_types_list[fid]
            if tcode == 0:
                resp[fid] = cont_mu[0, fid]
            elif tcode == 1:
                resp[fid] = torch.sigmoid(bin_logit[0, fid])
            elif tcode == 2 and cat_logits is not None:
                offset = cat_offsets[fid] if fid < len(cat_offsets) else -1
                card = cat_cards[fid] if fid < len(cat_cards) else 0
                if offset >= 0 and card > 0:
                    logits = cat_logits[0, offset : offset + card]
                    probs = torch.softmax(logits, dim=-1)
                    resp[fid] = probs.max()
                else:
                    resp[fid] = float("nan")
            else:
                resp[fid] = float("nan")

        responses.append(resp.detach().cpu().numpy())
        latents.append(mu.squeeze(0).detach().cpu().numpy())

    if not responses:
        return {"status": "skipped", "reason": "no evaluable sets"}

    responses_np = np.asarray(responses, dtype=np.float32)
    latents_np = np.asarray(latents, dtype=np.float32)

    col_means = np.nanmean(responses_np, axis=0)
    inds = np.where(np.isnan(responses_np))
    responses_np[inds] = np.take(col_means, inds[1])

    corr = np.corrcoef(responses_np, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    visited: set[int] = set()
    clusters: List[Dict[str, Any]] = []
    for fid in range(num_features):
        if fid in visited:
            continue
        component: List[int] = []
        queue = [fid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            neighbors = [j for j in range(num_features) if j != current and abs(corr[current, j]) >= corr_threshold]
            queue.extend([n for n in neighbors if n not in visited])
        if len(component) > 1:
            clusters.append({
                "feature_ids": component,
                "size": len(component),
                "mean_internal_corr": float(
                    np.mean([abs(corr[i, j]) for i in component for j in component if i != j])
                ) if len(component) > 1 else 0.0,
            })

    top_pairs: List[Tuple[int, int, float]] = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            top_pairs.append((i, j, float(abs(corr[i, j]))))
    top_pairs.sort(key=lambda item: item[2], reverse=True)

    latent_cov = np.cov(latents_np, rowvar=False) if latents_np.shape[0] > 1 else np.zeros((latents_np.shape[1], latents_np.shape[1]))
    eigvals = np.linalg.eigvalsh(latent_cov) if latent_cov.size > 0 else np.array([])

    save_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None and corr.size > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title("Feature response correlation heatmap")
        ax.set_xlabel("feature id")
        ax.set_ylabel("feature id")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_dir / "feature_correlation_heatmap.png", dpi=220)
        plt.close(fig)

    return {
        "status": "ok",
        "sets_evaluated": len(responses_np),
        "correlation_threshold": corr_threshold,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "top_pairs": top_pairs[:20],
        "latent_eigenvalues": eigvals.tolist() if eigvals.size > 0 else [],
    }


def evaluate_intraset_self_consistency(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    max_sets: int,
) -> Dict[str, Any]:
    if getattr(model, "feature_types", None) is None or getattr(model, "enable_prob_head", False) is False:
        return {"status": "skipped", "reason": "probabilistic head or feature schema unavailable"}

    cont_pit: List[float] = []
    cont_nll: List[float] = []
    cont_coverage: Dict[float, float] = {}
    cont_ci_hits: List[int] = []
    cont_abs_err: List[float] = []

    bin_probs: List[float] = []
    bin_labels: List[float] = []
    bin_pit: List[float] = []
    bin_nll: List[float] = []
    bin_pred_conf: List[float] = []
    bin_pred_correct: List[float] = []

    cat_true_prob: List[float] = []
    cat_pred_conf: List[float] = []
    cat_pred_correct: List[float] = []
    cat_pit: List[float] = []
    cat_nll: List[float] = []

    processed = 0

    for set_dict in _iterate_sets(model, batches, device):
        feat = set_dict.get("feat_id")
        if feat is None:
            continue
        num_tokens = int(feat.size(1))
        if num_tokens == 0:
            continue
        processed += 1
        if processed > max_sets:
            break

        indices = list(range(num_tokens))
        with torch.no_grad():
            for idx in indices:
                keep = [k for k in indices if k != idx]
                reduced = _slice_set_by_indices(set_dict, keep)
                if reduced is None:
                    continue
                mu_cond, _ = _encode_set_latent(model, reduced)
                head = _predict_head_outputs(model, mu_cond)
                if head is None:
                    continue
                cont_mu, cont_logvar, bin_logit, cat_logits = head

                fid = int(set_dict["feat_id"][0, idx, 0].item())
                tcode = int(model.feature_types[fid].item()) if model.feature_types is not None else 0
                value = float(set_dict["val"][0, idx, 0].item())

                if tcode == 0:
                    mu = float(cont_mu[0, fid].item())
                    logvar = float(cont_logvar[0, fid].item())
                    var = max(math.exp(logvar), 1e-6)
                    sigma = math.sqrt(var)
                    z_score = (value - mu) / sigma
                    pit = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
                    nll = 0.5 * (math.log(2 * math.pi) + logvar + ((value - mu) ** 2) / var)
                    cont_pit.append(float(pit))
                    cont_nll.append(float(nll))
                    cont_ci_hits.append(1 if abs(value - mu) <= 1.96 * sigma else 0)
                    cont_abs_err.append(abs(value - mu))
                elif tcode == 1:
                    p = float(torch.sigmoid(bin_logit[0, fid]).item())
                    label = 1.0 if value > 0.5 else 0.0
                    pit = (1 - p) + 0.5 * p if label > 0.5 else 0.5 * (1 - p)
                    eps = 1e-8
                    nll = -(label * math.log(max(p, eps)) + (1 - label) * math.log(max(1 - p, eps)))
                    pred_label = 1.0 if p >= 0.5 else 0.0
                    pred_conf = p if pred_label > 0.5 else (1 - p)
                    correct = 1.0 if pred_label == label else 0.0
                    bin_probs.append(p)
                    bin_labels.append(label)
                    bin_pit.append(float(pit))
                    bin_nll.append(float(nll))
                    bin_pred_conf.append(float(pred_conf))
                    bin_pred_correct.append(float(correct))
                elif tcode == 2 and cat_logits is not None:
                    offset = int(model.cat_offsets[fid].item()) if hasattr(model, "cat_offsets") else -1
                    card = int(model.cat_cardinalities[fid].item()) if hasattr(model, "cat_cardinalities") else 0
                    if offset < 0 or card <= 0:
                        continue
                    logits = cat_logits[0, offset : offset + card]
                    probs = torch.softmax(logits, dim=-1)
                    klass = int(round(value))
                    klass = max(0, min(card - 1, klass))
                    prob_true = float(probs[klass].item())
                    pit = float(torch.sum(probs[:klass]).item() + 0.5 * prob_true)
                    eps = 1e-12
                    nll = -math.log(max(prob_true, eps))
                    pred_class = int(torch.argmax(probs).item())
                    pred_conf = float(probs[pred_class].item())
                    correct = 1.0 if pred_class == klass else 0.0
                    cat_true_prob.append(prob_true)
                    cat_pred_conf.append(pred_conf)
                    cat_pred_correct.append(correct)
                    cat_pit.append(pit)
                    cat_nll.append(float(nll))

    coverage_levels = [0.5, 0.8, 0.9, 0.95]
    cont_coverage = {
        str(level): float(np.mean(np.asarray(cont_pit) <= level)) if cont_pit else float("nan")
        for level in coverage_levels
    }

    cont_ks, cont_p = _ks_test_uniform(cont_pit)
    bin_ks, bin_p = _ks_test_uniform(bin_pit)
    cat_ks, cat_p = _ks_test_uniform(cat_pit)

    results = {
        "status": "ok",
        "sets_evaluated": processed,
        "continuous": {
            "count": len(cont_pit),
            "ks_stat": cont_ks,
            "ks_pvalue": cont_p,
            "nll_mean": float(np.mean(cont_nll)) if cont_nll else float("nan"),
            "mae_mean": float(np.mean(cont_abs_err)) if cont_abs_err else float("nan"),
            "ci95_hit_rate": float(np.mean(cont_ci_hits)) if cont_ci_hits else float("nan"),
            "pit_quantiles": np.quantile(cont_pit, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if len(cont_pit) >= 5 else [],
            "coverage": cont_coverage,
        },
        "binary": {
            "count": len(bin_probs),
            "ks_stat": bin_ks,
            "ks_pvalue": bin_p,
            "nll_mean": float(np.mean(bin_nll)) if bin_nll else float("nan"),
            "ece": _expected_calibration_error(bin_pred_conf, bin_pred_correct) if bin_pred_conf else float("nan"),
            "accuracy": float(np.mean(bin_pred_correct)) if bin_pred_correct else float("nan"),
        },
        "categorical": {
            "count": len(cat_true_prob),
            "ks_stat": cat_ks,
            "ks_pvalue": cat_p,
            "nll_mean": float(np.mean(cat_nll)) if cat_nll else float("nan"),
            "ece": _expected_calibration_error(cat_pred_conf, cat_pred_correct) if cat_pred_conf else float("nan"),
            "accuracy": float(np.mean(cat_pred_correct)) if cat_pred_correct else float("nan"),
        },
    }

    return results


def evaluate_prior_alignment(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    sample_limit: int = 4096,
    seed: int = 0,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    mus_all: List[torch.Tensor] = []
    logvars_all: List[torch.Tensor] = []

    for batch in batches:
        batch_dev = _to_device(batch, device)
        with torch.no_grad():
            mus, logvars = _extract_set_posteriors(model, batch_dev)
        mus_all.extend(mus)
        logvars_all.extend(logvars)

    if not mus_all:
        return {"status": "skipped", "reason": "no posterior samples"}

    mu_stack = _stack_tensors(mus_all)
    logvar_stack = _stack_tensors(logvars_all)
    latent_dim = mu_stack.size(-1)

    std_stack = torch.exp(0.5 * logvar_stack)
    if mu_stack.size(0) > sample_limit:
        idx = torch.randperm(mu_stack.size(0))[:sample_limit]
        mu_stack = mu_stack[idx]
        std_stack = std_stack[idx]
    samples = mu_stack + torch.randn_like(mu_stack) * std_stack

    samples_np = samples.detach().cpu().numpy()
    mu_global = np.mean(samples_np, axis=0)
    cov_global = np.cov(samples_np, rowvar=False) if samples_np.shape[0] > 1 else np.eye(latent_dim)

    mean_norm = float(np.linalg.norm(mu_global))
    cov_deviation = float(np.linalg.norm(cov_global - np.eye(latent_dim)) / latent_dim)
    try:
        logdet_cov = float(np.linalg.slogdet(cov_global + 1e-6 * np.eye(latent_dim))[1])
    except Exception:
        logdet_cov = float("nan")

    try:
        prior_samples = torch.randn_like(samples)
        mmd = float(_mmd_rbf(samples, prior_samples, bandwidths=(0.2, 0.5, 1.0, 2.0, 5.0)).item())
    except Exception:
        mmd = float("nan")

    trace_cov = float(np.trace(cov_global))
    kl_gaussian = float(0.5 * (trace_cov + mean_norm ** 2 - latent_dim - logdet_cov)) if math.isfinite(logdet_cov) else float("nan")

    save_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None and latent_dim >= 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.4, s=8, label="aggregated posterior")
        prior_np = np.random.randn(min(len(samples_np), 2000), latent_dim)
        ax.scatter(prior_np[:, 0], prior_np[:, 1], alpha=0.2, s=8, label="prior N(0,I)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("Prior vs aggregated posterior (first two dims)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "prior_vs_posterior_scatter.png", dpi=220)
        plt.close(fig)

    return {
        "status": "ok",
        "latent_dim": latent_dim,
        "sample_count": int(samples_np.shape[0]),
        "mean_norm": mean_norm,
        "covariance_deviation": cov_deviation,
        "trace_covariance": trace_cov,
        "logdet_covariance": logdet_cov,
        "gaussian_kl_to_prior": kl_gaussian,
        "mmd_to_prior": mmd,
    }


def evaluate_noise_robustness(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    save_dir: Path,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.2),
    max_sets: int = 128,
    seed: int = 17,
) -> Dict[str, Any]:
    encoder = getattr(model, "set_encoder", None)
    if encoder is None:
        return {"status": "skipped", "reason": "model missing set encoder"}

    torch.manual_seed(seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    sets_cache: List[Dict[str, Any]] = []
    for set_dict in _iterate_sets(model, batches, device):
        sets_cache.append({k: v.clone() if torch.is_tensor(v) else v for k, v in set_dict.items()})
        if len(sets_cache) >= max_sets:
            break

    if not sets_cache:
        return {"status": "skipped", "reason": "no evaluable sets"}

    def _apply_noise(source: Dict[str, Any], level: float) -> Dict[str, Any]:
        noisy = _clone_set_dict(source)
        if level <= 0:
            return noisy
        if noisy.get("val") is not None:
            val = noisy["val"]
            noise = torch.randn(val.size(), device=val.device, generator=rng) * level
            val = val + noise
            dropout_p = min(0.3, level * 1.5)
            if dropout_p > 0:
                drop_mask = torch.rand(val.size(), device=val.device, generator=rng) < dropout_p
                val = val.masked_fill(drop_mask, 0.0)
            noisy["val"] = val
        if noisy.get("var") is not None:
            var = noisy["var"]
            dir_noise = torch.randn(var.size(), device=var.device, generator=rng) * (level * 0.1)
            noisy["var"] = var + dir_noise
        return noisy

    def _score_sets(sets: List[Dict[str, Any]]) -> Dict[str, float]:
        sinkhorn_scores: List[float] = []
        uncertainty_scores: List[float] = []
        head_available = False
        for s in sets:
            var = s.get("var")
            val = s.get("val")
            if var is None or val is None:
                continue
            var = var.to(device)
            val = val.to(device)
            try:
                score = _per_set_elbo_score(encoder, var, val)
            except Exception:
                score = float("nan")
            sinkhorn_scores.append(score)
            with torch.no_grad():
                mu, _ = _encode_set_latent(model, s)
                head = _predict_head_outputs(model, mu)
            unc = _compute_uncertainty_scalar(model, head)
            if math.isfinite(unc):
                uncertainty_scores.append(unc)
            if head is not None:
                head_available = True
        return {
            "sinkhorn_mean": float(np.nanmean(sinkhorn_scores)) if sinkhorn_scores else float("nan"),
            "sinkhorn_std": float(np.nanstd(sinkhorn_scores)) if sinkhorn_scores else float("nan"),
            "uncertainty_mean": float(np.nanmean(uncertainty_scores)) if uncertainty_scores else float("nan"),
            "uncertainty_std": float(np.nanstd(uncertainty_scores)) if uncertainty_scores else float("nan"),
            "head_available": head_available,
            "sample_count": len(sinkhorn_scores),
        }

    results: Dict[str, Any] = {}
    base_metrics: Optional[Dict[str, float]] = None

    for level in noise_levels:
        noisy_sets = [_apply_noise(s, float(level)) for s in sets_cache]
        metrics = _score_sets(noisy_sets)
        results[str(level)] = metrics
        if level == 0.0:
            base_metrics = metrics

    if base_metrics is not None:
        for level, metrics in results.items():
            if level == "0.0":
                continue
            metrics["sinkhorn_delta"] = metrics["sinkhorn_mean"] - base_metrics["sinkhorn_mean"]
            metrics["uncertainty_delta"] = metrics["uncertainty_mean"] - base_metrics["uncertainty_mean"]

    save_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None and len(results) > 1:
        xs = [float(k) for k in results.keys()]
        ys = [results[k]["sinkhorn_mean"] for k in results.keys()]
        ys2 = [results[k]["uncertainty_mean"] for k in results.keys()]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xs, ys, marker="o", color="#2a9d8f", label="sinkhorn score")
        ax1.set_xlabel("noise level")
        ax1.set_ylabel("ELBO-like score", color="#2a9d8f")
        ax1.tick_params(axis="y", labelcolor="#2a9d8f")
        ax2 = ax1.twinx()
        ax2.plot(xs, ys2, marker="s", color="#e76f51", label="uncertainty")
        ax2.set_ylabel("uncertainty", color="#e76f51")
        ax2.tick_params(axis="y", labelcolor="#e76f51")
        lines, labels = [], []
        for ax in (ax1, ax2):
            line, lab = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(lab)
        ax1.legend(lines, labels, loc="upper left")
        fig.tight_layout()
        fig.savefig(save_dir / "noise_robustness_curves.png", dpi=220)
        plt.close(fig)

    return {
        "status": "ok",
        "noise_levels": [float(l) for l in noise_levels],
        "metrics": results,
        "sets_cached": len(sets_cache),
    }


def summarize_dashboard(metrics: Dict[str, Any], save_dir: Path) -> Tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    flat_metrics: Dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _flatten(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flat_metrics[prefix] = json.dumps(value)
        else:
            flat_metrics[prefix] = value

    for key, value in metrics.items():
        _flatten(key, value)

    csv_path = save_dir / "eval_summary.csv"
    pd.DataFrame([flat_metrics]).to_csv(csv_path, index=False)

    json_path = save_dir / "eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return csv_path, json_path


def compare_checkpoints(*args, **kwargs) -> Path:
    raise NotImplementedError("Checkpoint leaderboard comparison is not yet implemented.")


# ---------------------------------------------------------------------------
# Orchestration


@dataclass
class EvalConfig:
    ckpt: str
    schema: str
    data_dir: str
    seed: int
    n_z_samples: int
    save_dir: Path
    device: torch.device
    batch_size: int
    info_top_k: int
    info_max_sets: int
    consistency_max_sets: int
    recon_max_sets: int
    conditional_mask_ratios: Tuple[float, ...]
    conditional_per_ratio_sets: int
    active_max_sets: int
    subsystem_max_sets: int
    prior_sample_limit: int
    noise_levels: Tuple[float, ...]
    noise_max_sets: int


def build_evaluation_config(args: argparse.Namespace) -> EvalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _parse_floats(text: str, default: Sequence[float]) -> Tuple[float, ...]:
        if text is None:
            return tuple(float(x) for x in default)
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if not parts:
            return tuple(float(x) for x in default)
        return tuple(float(p) for p in parts)

    return EvalConfig(
        ckpt=args.ckpt,
        schema=args.schema,
        data_dir=args.data_dir,
        seed=args.seed,
        n_z_samples=args.n_z_samples,
        save_dir=Path(args.save_dir),
        device=device,
        batch_size=args.batch_size,
        info_top_k=args.info_top_k,
        info_max_sets=args.info_max_sets,
        consistency_max_sets=args.consistency_max_sets,
        recon_max_sets=args.recon_max_sets,
        conditional_mask_ratios=_parse_floats(args.conditional_mask_ratios, (0.2, 0.5, 0.8)),
        conditional_per_ratio_sets=args.conditional_per_ratio_sets,
        active_max_sets=args.active_max_sets,
        subsystem_max_sets=args.subsystem_max_sets,
        prior_sample_limit=args.prior_sample_limit,
        noise_levels=_parse_floats(args.noise_levels, (0.0, 0.05, 0.1, 0.2)),
        noise_max_sets=args.noise_max_sets,
    )


def run_evaluation(cfg: EvalConfig) -> Dict[str, Any]:
    LOGGER.info("Loading schema: %s", cfg.schema)
    schema = load_schema(cfg.schema)

    LOGGER.info("Loading checkpoint: %s", cfg.ckpt)
    state, hparams = _load_ckpt_bundle(cfg.ckpt)
    model = _build_model(state, hparams, schema)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(cfg.device)

    LOGGER.info("Loading validation data from: %s", cfg.data_dir)
    batches = load_validation_batches(cfg.data_dir, schema, cfg.batch_size)

    metrics: Dict[str, Any] = {}

    LOGGER.info("Starting evaluation: posterior_collapse_detection")
    posterior_dir = _ensure_dir(cfg.save_dir / "posterior_collapse_detection")
    posterior_metrics = evaluate_posterior_health(
        model,
        batches,
        device=cfg.device,
        n_z_samples=cfg.n_z_samples,
        save_dir=posterior_dir,
        seed=cfg.seed,
    )
    metrics["posterior_collapse_detection"] = posterior_metrics
    _write_json(posterior_dir / "metrics.json", posterior_metrics)
    LOGGER.info("Completed evaluation: posterior_collapse_detection")

    LOGGER.info("Starting evaluation: latent_information_distribution")
    info_dist_dir = _ensure_dir(cfg.save_dir / "latent_information_distribution")
    info_dist_metrics = evaluate_information_distribution(posterior_metrics)
    metrics["latent_information_distribution"] = info_dist_metrics
    _write_json(info_dist_dir / "metrics.json", info_dist_metrics)
    LOGGER.info("Completed evaluation: latent_information_distribution")

    LOGGER.info("Starting evaluation: reconstruction_quality")
    recon_dir = _ensure_dir(cfg.save_dir / "reconstruction_quality")
    recon_metrics = evaluate_reconstruction_quality(
        model,
        batches,
        device=cfg.device,
        save_dir=recon_dir,
        max_sets=cfg.recon_max_sets,
    )
    metrics["reconstruction_quality"] = recon_metrics
    _write_json(recon_dir / "metrics.json", recon_metrics)
    LOGGER.info("Completed evaluation: reconstruction_quality")

    LOGGER.info("Starting evaluation: conditional_inference")
    conditional_dir = _ensure_dir(cfg.save_dir / "conditional_inference")
    conditional_metrics = evaluate_conditional_inference(
        model,
        batches,
        device=cfg.device,
        save_dir=conditional_dir,
        mask_ratios=cfg.conditional_mask_ratios,
        per_ratio_sets=cfg.conditional_per_ratio_sets,
        seed=cfg.seed,
    )
    metrics["conditional_inference"] = conditional_metrics
    _write_json(conditional_dir / "metrics.json", conditional_metrics)
    LOGGER.info("Completed evaluation: conditional_inference")

    LOGGER.info("Starting evaluation: active_measurement_value")
    active_dir = _ensure_dir(cfg.save_dir / "active_measurement_value")
    active_metrics = evaluate_active_measurement_value(
        model,
        batches,
        device=cfg.device,
        save_dir=active_dir,
        max_sets=cfg.active_max_sets,
    )
    metrics["active_measurement_value"] = active_metrics
    _write_json(active_dir / "metrics.json", active_metrics)
    LOGGER.info("Completed evaluation: active_measurement_value")

    LOGGER.info("Starting evaluation: information_gain_monotonicity")
    ig_dir = _ensure_dir(cfg.save_dir / "information_gain_monotonicity")
    info_gain_metrics = evaluate_information_gain_monotonicity(
        model,
        batches,
        device=cfg.device,
        top_k=cfg.info_top_k,
        max_sets=cfg.info_max_sets,
    )
    metrics["information_gain_monotonicity"] = info_gain_metrics
    _write_json(ig_dir / "metrics.json", info_gain_metrics)
    LOGGER.info("Completed evaluation: information_gain_monotonicity")

    LOGGER.info("Starting evaluation: intraset_self_consistency")
    consistency_dir = _ensure_dir(cfg.save_dir / "intraset_self_consistency")
    consistency_metrics = evaluate_intraset_self_consistency(
        model,
        batches,
        device=cfg.device,
        max_sets=cfg.consistency_max_sets,
    )
    metrics["intraset_self_consistency"] = consistency_metrics
    _write_json(consistency_dir / "metrics.json", consistency_metrics)
    LOGGER.info("Completed evaluation: intraset_self_consistency")

    LOGGER.info("Starting evaluation: subsystem_discovery")
    subsystem_dir = _ensure_dir(cfg.save_dir / "subsystem_discovery")
    subsystem_metrics = evaluate_subsystem_discovery(
        model,
        batches,
        device=cfg.device,
        save_dir=subsystem_dir,
        max_sets=cfg.subsystem_max_sets,
    )
    metrics["subsystem_discovery"] = subsystem_metrics
    _write_json(subsystem_dir / "metrics.json", subsystem_metrics)
    LOGGER.info("Completed evaluation: subsystem_discovery")

    LOGGER.info("Starting evaluation: prior_vs_posterior_alignment")
    prior_dir = _ensure_dir(cfg.save_dir / "prior_vs_posterior_alignment")
    prior_metrics = evaluate_prior_alignment(
        model,
        batches,
        device=cfg.device,
        save_dir=prior_dir,
        sample_limit=cfg.prior_sample_limit,
        seed=cfg.seed,
    )
    metrics["prior_vs_posterior_alignment"] = prior_metrics
    _write_json(prior_dir / "metrics.json", prior_metrics)
    LOGGER.info("Completed evaluation: prior_vs_posterior_alignment")

    LOGGER.info("Starting evaluation: noise_robustness")
    noise_dir = _ensure_dir(cfg.save_dir / "noise_robustness")
    noise_metrics = evaluate_noise_robustness(
        model,
        batches,
        device=cfg.device,
        save_dir=noise_dir,
        noise_levels=cfg.noise_levels,
        max_sets=cfg.noise_max_sets,
        seed=cfg.seed,
    )
    metrics["noise_robustness"] = noise_metrics
    _write_json(noise_dir / "metrics.json", noise_metrics)
    LOGGER.info("Completed evaluation: noise_robustness")

    summarize_dashboard(metrics, cfg.save_dir)

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate SeqSetVAE checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to trained SetVAE checkpoint")
    ap.add_argument("--schema", required=True, help="Schema file (CSV or JSON)")
    ap.add_argument("--data_dir", required=True, help="Directory containing train/valid/test parquet splits")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for validation loader collation")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n_z_samples", type=int, default=1)
    ap.add_argument("--save_dir", default="./eval_outputs", help="Directory to write metrics & plots")
    ap.add_argument("--info_top_k", type=int, default=20, help="Top-k variables to evaluate in information gain monotonicity")
    ap.add_argument("--info_max_sets", type=int, default=128, help="Maximum number of sets sampled for information gain analysis")
    ap.add_argument("--consistency_max_sets", type=int, default=128, help="Maximum sets sampled for intra-set consistency analysis")
    ap.add_argument("--recon_max_sets", type=int, default=128, help="Maximum sets used for reconstruction quality evaluation")
    ap.add_argument("--conditional_mask_ratios", default="0.2,0.5,0.8", help="Comma-separated mask ratios for conditional inference experiments")
    ap.add_argument("--conditional_per_ratio_sets", type=int, default=128, help="Maximum sets per mask ratio for conditional inference")
    ap.add_argument("--active_max_sets", type=int, default=256, help="Maximum sets used for active measurement scoring")
    ap.add_argument("--subsystem_max_sets", type=int, default=512, help="Maximum sets used for subsystem discovery")
    ap.add_argument("--prior_sample_limit", type=int, default=4096, help="Maximum posterior samples used for prior alignment")
    ap.add_argument("--noise_levels", default="0.0,0.05,0.1,0.2", help="Comma-separated noise levels for robustness evaluation")
    ap.add_argument("--noise_max_sets", type=int, default=128, help="Maximum sets used per noise configuration")
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    cfg = build_evaluation_config(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

