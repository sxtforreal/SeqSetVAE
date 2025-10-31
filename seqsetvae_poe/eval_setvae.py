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
import os
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


def load_val_sets(path: str) -> List[Dict[str, torch.Tensor]]:
    """Load validation sets from a torch.save()-ed file or JSON."""

    ext = Path(path).suffix.lower()
    if ext in {".pt", ".pth", ".bin"}:
        data = torch.load(path, map_location="cpu")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported val_sets format (expected .pt/.pth/.json): {path}")

    if isinstance(data, dict) and all(torch.is_tensor(v) for v in data.values()):
        return [data]  # single mega-batch
    if isinstance(data, list):
        # Expect list of dict batches
        batches: List[Dict[str, torch.Tensor]] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise ValueError("Each batch entry must be a mapping")
            tensor_batch = {k: torch.tensor(v) if not torch.is_tensor(v) else v for k, v in item.items()}
            batches.append(tensor_batch)  # type: ignore[arg-type]
        return batches
    if isinstance(data, Mapping) and "batches" in data:
        batches = data["batches"]
        assert isinstance(batches, list)
        out: List[Dict[str, torch.Tensor]] = []
        for item in batches:
            tensor_batch = {k: torch.tensor(v) if not torch.is_tensor(v) else v for k, v in item.items()}
            out.append(tensor_batch)  # type: ignore[arg-type]
        return out
    raise ValueError("Unsupported val_sets payload structure")


# ---------------------------------------------------------------------------
# Mask scenarios


def parse_mask_scenarios(spec: str) -> List[str]:
    if spec.strip() == "":
        return ["none"]
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("Mask scenario file must contain a list")
    return [part.strip() for part in spec.split(",") if part.strip()]


def apply_mask_scenario(batch: Mapping[str, torch.Tensor], scenario: str, seed: int) -> Mapping[str, torch.Tensor]:
    if scenario == "none":
        return batch
    rng = torch.Generator(device=batch["val"].device)
    rng.manual_seed(seed & 0xFFFFFFFF)
    var = batch["var"].clone()
    val = batch["val"].clone()
    carry = batch.get("carry_mask", torch.zeros_like(val))
    padding = batch.get("padding_mask", torch.zeros(var.size(0), var.size(1), dtype=torch.bool, device=var.device))

    mask_live = (carry.squeeze(-1) < 0.5) & (~padding)

    if scenario.startswith("mcAR_"):
        try:
            ratio = float(scenario.split("_", 1)[1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid mcAR scenario: {scenario}") from exc
        probs = torch.rand_like(val.squeeze(-1), generator=rng)
        mask = (probs < ratio) & (~padding)
        val = val.masked_fill(mask.unsqueeze(-1), 0.0)
    elif scenario == "carry_only":
        val = val.masked_fill(mask_live.unsqueeze(-1), 0.0)
    else:
        LOGGER.warning("Unknown mask scenario '%s'; returning batch unchanged", scenario)
    out = dict(batch)
    out["val"] = val
    return out


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

    # Mutual information proxy
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

    # Latent traversal table placeholder (requires decoding utilities)
    results["latent_traversals"] = [
        {
            "dim": int(idx),
            "status": "TODO",
            "note": "Implement probabilistic-head traversal once decoding helper is available.",
        }
        for idx in torch.argsort(kl_mean, descending=True)[: min(5, kl_mean.numel())].tolist()
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

    return {
        "entropy": entropy,
        "entropy_norm": float(entropy / max_entropy) if max_entropy > 0 else float("nan"),
        "gini": gini,
        "status": "partial",
        "note": "Linear probe predictability not yet implemented.",
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
    }

    return {
        "status": "ok",
        "summary": summary,
        "per_set": metrics_per_set,
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
            "pit_quantiles": np.quantile(cont_pit, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if len(cont_pit) >= 5 else [],
            "coverage": cont_coverage,
        },
        "binary": {
            "count": len(bin_probs),
            "ks_stat": bin_ks,
            "ks_pvalue": bin_p,
            "nll_mean": float(np.mean(bin_nll)) if bin_nll else float("nan"),
            "ece": _expected_calibration_error(bin_pred_conf, bin_pred_correct) if bin_pred_conf else float("nan"),
        },
        "categorical": {
            "count": len(cat_true_prob),
            "ks_stat": cat_ks,
            "ks_pvalue": cat_p,
            "nll_mean": float(np.mean(cat_nll)) if cat_nll else float("nan"),
            "ece": _expected_calibration_error(cat_pred_conf, cat_pred_correct) if cat_pred_conf else float("nan"),
        },
    }

    return results


def evaluate_reconstruction_setlevel(*args, **kwargs) -> Tuple[Dict[str, Any], List[str]]:
    return {
        "status": "TODO",
        "note": "Sinkhorn head evaluation requires reconstruction hooks not present in this snapshot.",
    }, []


def evaluate_feature_inference(*args, **kwargs) -> Tuple[Dict[str, Any], List[str]]:
    return {
        "status": "TODO",
        "note": "Feature-level metrics depend on probabilistic-head sampling utilities.",
    }, []


def rank_next_measurements(*args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(columns=["feature_id", "score", "method"])


def analyze_subsystems_weights(*args, **kwargs) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    return {"status": "TODO"}, None


def analyze_subsystems_sensitivity(*args, **kwargs) -> Dict[str, Any]:
    return {"status": "TODO"}


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
    val_sets: str
    mask_scenarios: List[str]
    seed: int
    n_z_samples: int
    save_dir: Path
    device: torch.device
    info_top_k: int
    info_max_sets: int
    consistency_max_sets: int


def build_evaluation_config(args: argparse.Namespace) -> EvalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return EvalConfig(
        ckpt=args.ckpt,
        schema=args.schema,
        val_sets=args.val_sets,
        mask_scenarios=parse_mask_scenarios(args.mask_scenarios),
        seed=args.seed,
        n_z_samples=args.n_z_samples,
        save_dir=Path(args.save_dir),
        device=device,
        info_top_k=args.info_top_k,
        info_max_sets=args.info_max_sets,
        consistency_max_sets=args.consistency_max_sets,
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

    LOGGER.info("Loading validation sets: %s", cfg.val_sets)
    batches = load_val_sets(cfg.val_sets)

    metrics: Dict[str, Any] = {}

    LOGGER.info("Starting evaluation: posterior_health")
    posterior_dir = _ensure_dir(cfg.save_dir / "posterior_health")
    metrics["posterior_health"] = evaluate_posterior_health(
        model,
        batches,
        device=cfg.device,
        n_z_samples=cfg.n_z_samples,
        save_dir=posterior_dir,
        seed=cfg.seed,
    )
    _write_json(posterior_dir / "metrics.json", metrics["posterior_health"])
    LOGGER.info("Completed evaluation: posterior_health")

    LOGGER.info("Starting evaluation: latent_information_distribution")
    info_dist_dir = _ensure_dir(cfg.save_dir / "latent_information_distribution")
    metrics["latent_information_distribution"] = evaluate_information_distribution(metrics["posterior_health"])
    _write_json(info_dist_dir / "metrics.json", metrics["latent_information_distribution"])
    LOGGER.info("Completed evaluation: latent_information_distribution")

    LOGGER.info("Starting evaluation: information_gain_monotonicity")
    ig_dir = _ensure_dir(cfg.save_dir / "information_gain_monotonicity")
    metrics["information_gain_monotonicity"] = evaluate_information_gain_monotonicity(
        model,
        batches,
        device=cfg.device,
        top_k=cfg.info_top_k,
        max_sets=cfg.info_max_sets,
    )
    _write_json(ig_dir / "metrics.json", metrics["information_gain_monotonicity"])
    LOGGER.info("Completed evaluation: information_gain_monotonicity")

    LOGGER.info("Starting evaluation: intraset_self_consistency")
    consistency_dir = _ensure_dir(cfg.save_dir / "intraset_self_consistency")
    metrics["intraset_self_consistency"] = evaluate_intraset_self_consistency(
        model,
        batches,
        device=cfg.device,
        max_sets=cfg.consistency_max_sets,
    )
    _write_json(consistency_dir / "metrics.json", metrics["intraset_self_consistency"])
    LOGGER.info("Completed evaluation: intraset_self_consistency")

    placeholder_sections: Dict[str, Any] = {
        "set_reconstruction": {"status": "pending"},
        "feature_inference": {scenario: {"status": "pending"} for scenario in cfg.mask_scenarios},
        "next_best_measurement": {"status": "pending"},
        "subsystem_weights": {"status": "pending"},
        "subsystem_sensitivity": {"status": "pending"},
        "prior_vs_posterior": {"status": "pending"},
        "stress_tests": {"status": "pending"},
        "heads_ablation": {"status": "pending"},
        "model_selection": {"status": "pending"},
    }

    for name, value in placeholder_sections.items():
        subdir = _ensure_dir(cfg.save_dir / name)
        _write_json(subdir / "metrics.json", value)
        metrics[name] = value

    summarize_dashboard(metrics, cfg.save_dir)

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate SeqSetVAE checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to trained SetVAE checkpoint")
    ap.add_argument("--schema", required=True, help="Schema file (CSV or JSON)")
    ap.add_argument("--val_sets", required=True, help="Torch file with validation batches")
    ap.add_argument("--mask_scenarios", default="none", help="Comma list or JSON file of mask scenarios")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n_z_samples", type=int, default=1)
    ap.add_argument("--save_dir", default="./eval_outputs", help="Directory to write metrics & plots")
    ap.add_argument("--info_top_k", type=int, default=20, help="Top-k variables to evaluate in information gain monotonicity")
    ap.add_argument("--info_max_sets", type=int, default=128, help="Maximum number of sets sampled for information gain analysis")
    ap.add_argument("--consistency_max_sets", type=int, default=128, help="Maximum sets sampled for intra-set consistency analysis")
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    cfg = build_evaluation_config(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

