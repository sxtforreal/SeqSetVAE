#!/usr/bin/env python3
"""Full-stack evaluation suite for SeqSetVAE pretraining models.

This script implements posterior diagnostics, reconstruction quality checks,
conditional inference stress tests, active observation ranking, latent
structure analysis, prior alignment and robustness probes.  The
implementation follows the 10-module specification used by the project so
that running the script once produces a complete report and intermediate
artifacts on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:  # optional plotting dependencies
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - keep optional
    plt = None  # type: ignore

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - keep optional
    umap = None  # type: ignore

from dataset import _collate_lvcf, _detect_vcols  # type: ignore


EPS = 1e-8
MASK_SCENARIOS_SUPPORTED = ["none", "mar_0.2", "mar_0.5", "mar_0.8", "carry_only"]
ROBUST_DEFAULT_MODES = [
    "train_noise_on_infer_off",
    "train_noise_on_infer_on",
    "train_noise_off_infer_on",
    "head_sinkhorn_only",
    "head_prob_only",
]


try:
    from torch.cuda.amp import autocast as _torch_autocast
except Exception:  # pragma: no cover
    @contextmanager
    def _amp_autocast():
        yield
else:
    @contextmanager
    def _amp_autocast():
        if torch.cuda.is_available():
            with _torch_autocast():
                yield
        else:
            yield


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


def _str2bool(val: str) -> bool:
    return val.lower() in {"1", "true", "yes", "y", "t"}


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
    def feature_types(self) -> torch.Tensor:
        return torch.tensor([f.type_code for f in self.features], dtype=torch.long)

    @property
    def categorical_ids(self) -> List[int]:
        return [f.feature_id for f in self.features if f.type_code == 2]

    @property
    def categorical_cards(self) -> List[int]:
        return [f.cardinality for f in self.features if f.type_code == 2]


@dataclass
class SetSample:
    uid: str
    tensors: Dict[str, torch.Tensor]
    target: torch.Tensor
    v_norm: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    size: int
    head_cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = field(default=None)


@dataclass
class DataCache:
    batches: List[Dict[str, torch.Tensor]]
    sets: List[SetSample]
    feature_types: torch.Tensor
    cat_offsets: Optional[torch.Tensor]
    cat_cardinalities: Optional[torch.Tensor]

    @property
    def latent_matrix(self) -> torch.Tensor:
        if not hasattr(self, "_latent_matrix"):
            self._latent_matrix = torch.stack([s.mu for s in self.sets], dim=0)
        return getattr(self, "_latent_matrix")


@dataclass
class MaskRegistry:
    records: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))

    def register(self, scenario: str, indices: np.ndarray) -> None:
        self.records[scenario].append(indices.astype(np.int64))

    def dump(self, base_dir: Path) -> None:
        for scenario, arrs in self.records.items():
            if not arrs:
                continue
            out_dir = _ensure_dir(base_dir / scenario)
            # variable-length arrays, use object dtype for reproducibility
            stacked = np.array([a for a in arrs], dtype=object)
            np.save(out_dir / "mask_index.npy", stacked, allow_pickle=True)


@dataclass
class EvalConfig:
    ckpt: str
    schema: str
    data_dir: str
    save_dir: Path
    batch_size: int
    seed: int
    n_probe_z: int
    plots: bool
    mask_scenarios: List[str]
    topk_ig: int
    sample_sets: int
    save_raw: bool
    compare_ckpts: List[str]
    robust_modes: List[str]
    tier: str
    eval_sinkhorn: bool
    eval_probes: bool
    eval_active: bool
    eval_ig: bool
    eval_prior: bool
    eval_subsystems: bool
    eval_robustness: bool
    mc_samples: int
    intra_samples: int
    active_topk: int


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------


def load_schema(path: str) -> SchemaBundle:
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
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Schema file missing required columns: {missing}")

    features: List[FeatureSchema] = []
    type_map = {"cont": 0, "continuous": 0, "real": 0, "bin": 1, "binary": 1, "cat": 2, "categorical": 2}

    for _, row in df.iterrows():
        raw_type = str(row["type"]).strip().lower()
        type_code = type_map.get(raw_type)
        if type_code is None:
            raise ValueError(f"Unrecognized feature type '{row['type']}'")
        name = None
        if "name" in row and not pd.isna(row["name"]):
            name = str(row["name"])
        cardinality = 0
        if "cardinality" in row and not pd.isna(row["cardinality"]):
            cardinality = int(row["cardinality"])
        features.append(
            FeatureSchema(
                feature_id=int(row["feature_id"]),
                type_code=type_code,
                name=name,
                cardinality=cardinality,
            )
        )

    features.sort(key=lambda f: f.feature_id)
    return SchemaBundle(features=features)


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _load_ckpt_bundle(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        state = {}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    hparams: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("hyper_parameters", "hparams"):
            hp = raw.get(key)
            if isinstance(hp, dict):
                hparams = hp
                break
    return state, hparams


def _infer_dims_from_state(state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    dims: Dict[str, int] = {}

    w = state.get("set_encoder.dim_reducer.weight")
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        dims["reduced_dim"], dims["input_dim"] = int(w.shape[0]), int(w.shape[1])

    ew = state.get("set_encoder.embed.0.weight")
    if isinstance(ew, torch.Tensor) and ew.dim() == 2:
        latent_dim, embed_in = int(ew.shape[0]), int(ew.shape[1])
        dims.setdefault("latent_dim", latent_dim)
        dims.setdefault("reduced_dim", embed_in)
        dims.setdefault("input_dim", embed_in)

    ow = state.get("set_encoder.out.weight")
    if isinstance(ow, torch.Tensor) and ow.dim() == 2:
        out_dim, latent_dim = int(ow.shape[0]), int(ow.shape[1])
        dims.setdefault("latent_dim", latent_dim)
        dims.setdefault("reduced_dim", out_dim)
        dims.setdefault("input_dim", out_dim)

    mu_w = state.get("set_encoder.mu_logvar.4.weight")
    if isinstance(mu_w, torch.Tensor) and mu_w.dim() == 2:
        dims["latent_dim"] = int(mu_w.shape[1])

    return dims


def _detect_num_flows(state: Mapping[str, torch.Tensor]) -> int:
    indices: set[int] = set()
    for key in state.keys():
        if ".flows." not in key:
            continue
        suffix = key.split(".flows.")[-1].split(".")[0]
        if suffix.isdigit():
            indices.add(int(suffix))
    return (max(indices) + 1) if indices else 0


def _build_model(state: Mapping[str, torch.Tensor], hparams: Mapping[str, Any], schema: SchemaBundle) -> torch.nn.Module:
    try:
        from seqsetvae_poe.model import SetVAEOnlyPretrain  # type: ignore
    except Exception:  # pragma: no cover
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
        lr=float(_hp("lr", 3e-4)),
        warmup_beta=bool(_hp("warmup_beta", True)),
        max_beta=float(_hp("max_beta", 0.2)),
        beta_warmup_steps=int(_hp("beta_warmup_steps", 8000)),
        free_bits=float(_hp("free_bits", 0.05)),
        use_kl_capacity=bool(_hp("use_kl_capacity", True)),
        capacity_per_dim_end=float(_hp("capacity_per_dim_end", 0.03)),
        capacity_warmup_steps=int(_hp("capacity_warmup_steps", 20000)),
        use_flows=bool(num_flows > 0),
        num_flows=num_flows,
        use_sinkhorn=bool(_hp("use_sinkhorn", True)),
        sinkhorn_eps=float(_hp("sinkhorn_eps", 0.1)),
        sinkhorn_iters=int(_hp("sinkhorn_iters", 100)),
        enable_prob_head=True,
        num_features=schema.num_features,
        feature_types=schema.feature_types.tolist(),
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
# Validation data caching utilities
# ---------------------------------------------------------------------------


def load_validation_batches(data_dir: str, schema: SchemaBundle, batch_size: int) -> List[Dict[str, torch.Tensor]]:
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
        current.append((df, path.stem))
        if len(current) >= batch_size:
            batches.append(_collate_lvcf(current, vcols, name_to_id_or_none))
            current = []

    if current:
        batches.append(_collate_lvcf(current, vcols, name_to_id_or_none))

    return batches


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _clone_set_to_device(sample: SetSample, device: torch.device, indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in sample.tensors.items():
        if tensor is None:
            out[key] = tensor
            continue
        t = tensor.to(device)
        if indices is not None and tensor.dim() >= 2:
            t = t.index_select(1, indices.to(device))
        out[key] = t
    return out


def _encode_set_latent(model: torch.nn.Module, s: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    with _amp_autocast():
        v_norm, _ = model._compute_target(s)
        if hasattr(model, "_build_token_inputs") and getattr(model, "num_features", 0) > 0:
            x_input = model._build_token_inputs(v_norm, s)
        else:
            x_input = v_norm * s["val"]
        z_list, _ = model.set_encoder.encode(x_input)
        mu = z_list[-1][1].squeeze(1)
        logvar = z_list[-1][2].squeeze(1)
    return mu, logvar


def build_data_cache(
    model: torch.nn.Module,
    batches: Sequence[Dict[str, torch.Tensor]],
    schema: SchemaBundle,
    device: torch.device,
    cache_dir: Path,
    save_raw: bool,
) -> DataCache:
    model.eval()
    sets: List[SetSample] = []
    mask_registry = MaskRegistry()  # placeholder for future population to satisfy typing
    _ = mask_registry  # avoid linter warnings until used later

    for batch_idx, batch in enumerate(batches):
        batch_dev = _to_device(batch, device)
        try:
            patient_sets = model._split_sets(
                batch_dev["var"],
                batch_dev["val"],
                batch_dev["minute"],
                batch_dev["set_id"],
                batch_dev.get("padding_mask"),
                batch_dev.get("carry_mask"),
                batch_dev.get("feat_id"),
            )
        except TypeError:  # compatibility with older checkpoints
            patient_sets = model._split_sets(
                batch_dev["var"],
                batch_dev["val"],
                batch_dev["minute"],
                batch_dev["set_id"],
                batch_dev.get("padding_mask"),
                batch_dev.get("carry_mask"),
            )

        for local_idx, s in enumerate(patient_sets):
            uid = f"set_{batch_idx:04d}_{local_idx:03d}"
            with torch.no_grad():
                mu, logvar = _encode_set_latent(model, s)
                v_norm, target = model._compute_target(s)
            cpu_tensors = {
                key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
                for key, tensor in s.items()
            }
            sample = SetSample(
                uid=uid,
                tensors=cpu_tensors,
                target=target.detach().cpu(),
                v_norm=v_norm.detach().cpu(),
                mu=mu.detach().cpu(),
                logvar=logvar.detach().cpu(),
                size=int(cpu_tensors["var"].shape[1]) if "var" in cpu_tensors else 0,
            )
            sets.append(sample)

    cache = DataCache(
        batches=list(batches),
        sets=sets,
        feature_types=schema.feature_types,
        cat_offsets=None,
        cat_cardinalities=None,
    )

    if schema.categorical_ids:
        cat_offsets = torch.full((schema.num_features,), fill_value=-1, dtype=torch.long)
        cat_cards = torch.zeros(schema.num_features, dtype=torch.long)
        running = 0
        for fid, card in zip(schema.categorical_ids, schema.categorical_cards):
            cat_offsets[fid] = running
            cat_cards[fid] = max(1, card)
            running += max(1, card)
        cache.cat_offsets = cat_offsets
        cache.cat_cardinalities = cat_cards

    if save_raw:
        raw_dir = _ensure_dir(cache_dir / "raw_sets")
        for sample in sets:
            np.savez_compressed(
                raw_dir / f"{sample.uid}.npz",
                **{k: v.numpy() for k, v in sample.tensors.items() if torch.is_tensor(v)},
                target=sample.target.numpy(),
                v_norm=sample.v_norm.numpy(),
                mu=sample.mu.numpy(),
                logvar=sample.logvar.numpy(),
            )

    LOGGER.info("Cached %d validation batches containing %d sets", len(batches), len(sets))
    return cache


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[float]) -> float:
    arr = [float(v) for v in values if math.isfinite(v)]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = float((y_true > 0.5).sum())
    neg = float((y_true <= 0.5).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum = float(ranks[y_true > 0.5].sum())
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(max(0.0, min(1.0, auc)))


def _auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = 0.0
    fp = 0.0
    pr_pairs: List[Tuple[float, float]] = []
    pos_total = float((y_true > 0.5).sum())
    if pos_total == 0:
        return float("nan")
    for label in y_true:
        if label > 0.5:
            tp += 1.0
        else:
            fp += 1.0
        precision = tp / max(EPS, tp + fp)
        recall = tp / pos_total
        pr_pairs.append((recall, precision))
    pr_pairs.insert(0, (0.0, pr_pairs[0][1] if pr_pairs else 1.0))
    pr_pairs.append((1.0, pr_pairs[-1][1] if pr_pairs else 0.0))
    pr_pairs = sorted(pr_pairs, key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(pr_pairs)):
        r0, p0 = pr_pairs[i - 1]
        r1, p1 = pr_pairs[i]
        area += (r1 - r0) * ((p0 + p1) / 2.0)
    return float(max(0.0, min(1.0, area)))


def _ece_score(probs: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    if not probs:
        return float("nan")
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(probs_arr)
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i == bins - 1:
            mask = (probs_arr >= left) & (probs_arr <= right)
        else:
            mask = (probs_arr >= left) & (probs_arr < right)
        if not np.any(mask):
            continue
        acc = labels_arr[mask].mean()
        conf = probs_arr[mask].mean()
        ece += abs(acc - conf) * (mask.sum() / total)
    return float(ece)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0:
            continue
        precision = tp / max(tp + fp, EPS)
        recall = tp / max(tp + fn, EPS)
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def _coverage_rate(pit: Sequence[float], nominal: float) -> float:
    arr = np.asarray(pit, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= nominal))


def compute_continuous_metrics(
    preds_mu: Sequence[float],
    preds_var: Sequence[float],
    targets: Sequence[float],
) -> Dict[str, Any]:
    mu = np.asarray(preds_mu, dtype=np.float64)
    var = np.asarray(preds_var, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.isfinite(mu) & np.isfinite(var) & np.isfinite(y) & (var > 0)
    if not np.any(mask):
        return {"count": 0}
    mu = mu[mask]
    var = var[mask]
    y = y[mask]
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    mae = float(np.mean(np.abs(mu - y)))
    nll = float(0.5 * np.mean(np.log(2 * math.pi * var) + ((y - mu) ** 2) / var))
    sigma = np.sqrt(var)
    z = (y - mu) / sigma
    pit = 0.5 * (1.0 + torch.erf(torch.tensor(z / math.sqrt(2.0))).numpy())

    # CRPS for Gaussian predictive distribution
    crps = float(np.mean(
        sigma * (
            1 / math.sqrt(math.pi)
            - 2 * torch.distributions.Normal(0, 1).log_prob(torch.tensor(z)).exp().numpy()
            - z * (2 * pit - 1)
        )
    ))

    ks_stat, ks_p = _ks_test_uniform(pit)
    coverage = {
        "0.5": _coverage_rate(pit, 0.5),
        "0.8": _coverage_rate(pit, 0.8),
        "0.9": _coverage_rate(pit, 0.9),
        "0.95": _coverage_rate(pit, 0.95),
    }

    return {
        "count": int(mu.size),
        "rmse": rmse,
        "mae": mae,
        "nll": nll,
        "crps": crps,
        "pit_ks_stat": ks_stat,
        "pit_ks_p": ks_p,
        "coverage": coverage,
        "pit_values": pit.tolist(),
    }


def compute_binary_metrics(probs: Sequence[float], labels: Sequence[float]) -> Dict[str, Any]:
    if not probs:
        return {"count": 0}
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    auc = _auc_score(labels_arr, probs_arr)
    auprc = _auprc_score(labels_arr, probs_arr)
    brier = float(np.mean((probs_arr - labels_arr) ** 2))
    nll = float(-np.mean(labels_arr * np.log(probs_arr + EPS) + (1 - labels_arr) * np.log(1 - probs_arr + EPS)))
    ece = _ece_score(probs, labels)
    ks_stat, ks_p = _ks_test_uniform(np.where(labels_arr > 0.5, 1 - probs_arr / 2, probs_arr / 2))
    return {
        "count": int(len(probs)),
        "auc": auc,
        "auprc": auprc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
        "pit_ks_stat": ks_stat,
        "pit_ks_p": ks_p,
    }


def compute_multiclass_metrics(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    cardinalities: Sequence[int],
) -> Dict[str, Any]:
    if not logits:
        return {"count": 0}
    probs = [torch.softmax(torch.tensor(row, dtype=torch.float32), dim=-1).numpy() for row in logits]
    probs_arr = np.stack(probs, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(probs_arr, axis=1)
    acc = float(np.mean(preds == labels_arr))
    f1 = _macro_f1(labels_arr, preds, int(max(cardinalities) if cardinalities else probs_arr.shape[1]))
    ece = _ece_score(np.max(probs_arr, axis=1), (preds == labels_arr).astype(float))
    nll = float(-np.mean(np.log(probs_arr[np.arange(len(labels_arr)), labels_arr] + EPS)))
    return {
        "count": int(len(labels_arr)),
        "accuracy": acc,
        "macro_f1": f1,
        "ece": ece,
        "nll": nll,
    }


def merge_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    merged: Dict[str, Any] = {}
    for item in metrics:
        for key, value in item.items():
            if isinstance(value, dict):
                merged.setdefault(key, [])
                merged[key].append(value)
            else:
                merged.setdefault(key, []).append(value)
    final: Dict[str, Any] = {}
    for key, values in merged.items():
        if isinstance(values[0], dict):
            final[key] = merge_metrics(values)  # type: ignore[arg-type]
        else:
            final[key] = _safe_mean(values)
    return final


# ---------------------------------------------------------------------------
# Scenario helpers & sampling utilities
# ---------------------------------------------------------------------------


def parse_mask_scenarios(raw: str) -> List[str]:
    if not raw:
        return ["none"]
    scenarios = [s.strip() for s in raw.split(",") if s.strip()]
    validated = []
    for s in scenarios:
        if s not in MASK_SCENARIOS_SUPPORTED:
            raise ValueError(f"Unsupported mask scenario '{s}'")
        validated.append(s)
    return validated or ["none"]


def select_random_sets(cache: DataCache, limit: int, seed: int) -> List[SetSample]:
    rng = random.Random(seed)
    sets = list(cache.sets)
    if not sets:
        return []
    if limit <= 0 or limit >= len(sets):
        return sets
    return rng.sample(sets, limit)


def apply_mask_scenario(
    sample: SetSample,
    scenario: str,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_id = sample.tensors.get("feat_id")
    carry = sample.tensors.get("carry") or sample.tensors.get("carry_mask")
    size = sample.size
    indices = torch.arange(size, dtype=torch.long)
    if scenario == "none" or size == 0:
        return indices, torch.tensor([], dtype=torch.long)

    if scenario.startswith("mar_"):
        rate = float(scenario.split("_")[1])
        candidate = indices
        k = max(0, min(size, int(math.ceil(rate * size))))
        if k <= 0:
            return indices, torch.tensor([], dtype=torch.long)
        mask_idx = torch.tensor(rng.sample(candidate.tolist(), k), dtype=torch.long)
        keep = torch.tensor(sorted(set(candidate.tolist()) - set(mask_idx.tolist())), dtype=torch.long)
        if keep.numel() == 0:
            keep = mask_idx[:1]
            mask_idx = mask_idx[1:]
        return keep, mask_idx

    if scenario == "carry_only" and carry is not None:
        carry_vec = (carry.squeeze(0).squeeze(-1) > 0.5).nonzero(as_tuple=False).squeeze(-1)
        keep = carry_vec
        mask_idx = torch.tensor(sorted(set(indices.tolist()) - set(keep.tolist())), dtype=torch.long)
        if keep.numel() == 0:
            keep = indices[:1]
            mask_idx = indices[1:]
        return keep, mask_idx

    # fallback: no masking
    return indices, torch.tensor([], dtype=torch.long)


def compute_token_uncertainty(
    model: torch.nn.Module,
    head_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    fid: int,
    feature_types: torch.Tensor,
) -> float:
    cont_mu, cont_logvar, bin_logit, cat_logits = head_outputs
    tcode = int(feature_types[fid].item()) if feature_types.numel() > fid else 0
    if tcode == 0:
        var = float(torch.exp(cont_logvar[0, fid]).item())
        return var
    if tcode == 1:
        prob = float(torch.sigmoid(bin_logit[0, fid]).item())
        return prob * (1.0 - prob)
    if tcode == 2 and cat_logits is not None and getattr(model, "cat_offsets", None) is not None:
        offset = int(model.cat_offsets[fid].item()) if model.cat_offsets is not None else -1
        card = int(model.cat_cardinalities[fid].item()) if model.cat_cardinalities is not None else 0
        if offset >= 0 and card > 0:
            probs = torch.softmax(cat_logits[0, offset : offset + card], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + EPS))
            return float(entropy.item())
    return 0.0


def _predict_head_outputs(
    model: torch.nn.Module,
    z_mu: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if not getattr(model, "enable_prob_head", False):
        return None
    if getattr(model, "prob_shared", None) is None:
        return None
    h = model.prob_shared(z_mu.unsqueeze(0))
    cont_mu = model.prob_cont_mu(h)
    cont_logvar = model.prob_cont_logvar(h)
    bin_logit = model.prob_bin_logit(h)
    cat_logits = model.prob_cat_logits(h) if getattr(model, "prob_cat_logits", None) is not None else None
    return cont_mu, cont_logvar, bin_logit, cat_logits


def _aggregate_feature_targets(sample: SetSample, feature_types: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_features = int(feature_types.numel())
    cont = np.full(num_features, np.nan, dtype=np.float64)
    binaria = np.full(num_features, np.nan, dtype=np.float64)
    categorical = np.full(num_features, -1, dtype=np.int64)
    feat_id = sample.tensors.get("feat_id")
    values = sample.tensors.get("val")
    if feat_id is None or values is None:
        return cont, binaria, categorical
    fid = feat_id.squeeze(0).squeeze(-1).numpy().astype(int)
    val = values.squeeze(0).squeeze(-1).numpy()
    for idx, f in enumerate(fid):
        if f < 0 or f >= num_features:
            continue
        tcode = int(feature_types[f].item())
        if tcode == 0:
            cont[f] = float(val[idx])
        elif tcode == 1:
            binaria[f] = 1.0 if val[idx] > 0.5 else 0.0
        elif tcode == 2:
            categorical[f] = int(round(float(val[idx])))
    return cont, binaria, categorical


def _get_head_outputs(
    model: torch.nn.Module,
    sample: SetSample,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if sample.head_cache is not None:
        cached: List[Optional[torch.Tensor]] = []
        for item in sample.head_cache:
            if item is None:
                cached.append(None)
            else:
                cached.append(item.to(device))
        return tuple(cached)  # type: ignore[return-value]

    z_mu = sample.mu.to(device)
    head = _predict_head_outputs(model, z_mu)
    if head is None:
        sample.head_cache = None
        return None
    cached = tuple(t.detach().cpu() if t is not None else None for t in head)
    sample.head_cache = cached
    return _get_head_outputs(model, sample, device)


# ---------------------------------------------------------------------------
# Evaluation modules
# ---------------------------------------------------------------------------


def evaluate_posterior_health(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    save_dir = _ensure_dir(cfg.save_dir / "posterior")

    mu_stack = cache.latent_matrix
    logvar_stack = torch.stack([s.logvar for s in cache.sets], dim=0)
    kl_dim = _kl_diag_gaussian(mu_stack, logvar_stack)
    kl_mean = kl_dim.mean(dim=0)
    kl_std = kl_dim.std(dim=0, unbiased=False)
    kl_total = float(kl_mean.sum().item())

    active_dims = {
        "@0.1": int((kl_mean > 0.1).sum().item()),
        "@0.05": int((kl_mean > 0.05).sum().item()),
        "ratio@0.1": float((kl_mean > 0.1).sum().item()) / max(1, kl_mean.numel()),
    }

    coverage_curve = []
    sorted_vals, _ = torch.sort(kl_mean, descending=True)
    cumulative = torch.cumsum(sorted_vals, dim=0)
    total = float(sorted_vals.sum().item() + EPS)
    for k in range(sorted_vals.numel()):
        coverage_curve.append((k + 1, float(cumulative[k].item() / total)))

    pd.DataFrame(
        {
            "dimension": np.arange(kl_mean.numel()),
            "kl_mean": kl_mean.cpu().numpy(),
            "kl_std": kl_std.cpu().numpy(),
        }
    ).to_csv(save_dir / "kl_per_dim.csv", index=False)

    pd.DataFrame(coverage_curve, columns=["top_k", "coverage"]).to_csv(save_dir / "coverage_curve.csv", index=False)
    _write_json(save_dir / "active_dims.json", active_dims)

    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device=mu_stack.device)
    generator.manual_seed(cfg.seed)
    samples_per_set = max(1, cfg.n_probe_z)
    q_samples = _sample_diag_gaussians(mu_stack, logvar_stack, samples_per_set, generator=generator)
    q_samples_flat = q_samples.reshape(-1, mu_stack.shape[-1])
    p_samples = torch.randn_like(q_samples_flat, generator=generator)
    mmd = _mmd_rbf(q_samples_flat, p_samples, bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0))
    mi_proxy = float(kl_dim.sum(dim=1).mean().item() - mmd.item())

    compare_rows = []
    if cfg.compare_ckpts:
        for ckpt_path in cfg.compare_ckpts:
            try:
                state, hparams = _load_ckpt_bundle(ckpt_path)
                tmp_model = _build_model(state, hparams, load_schema(cfg.schema))
                tmp_model.to(mu_stack.device)
                tmp_model.eval()
                with torch.no_grad():
                    tmp_mus = []
                    tmp_logvars = []
                    for sample in select_random_sets(cache, cfg.sample_sets, cfg.seed):
                        s_dev = _clone_set_to_device(sample, mu_stack.device)
                        mu, logvar = _encode_set_latent(tmp_model, s_dev)
                        tmp_mus.append(mu.cpu())
                        tmp_logvars.append(logvar.cpu())
                if tmp_mus:
                    tmp_mu_stack = torch.stack(tmp_mus, dim=0)
                    tmp_logvar_stack = torch.stack(tmp_logvars, dim=0)
                    mmd_tmp = _mmd_rbf(
                        _sample_diag_gaussians(tmp_mu_stack, tmp_logvar_stack, samples_per_set).reshape(-1, tmp_mu_stack.shape[-1]),
                        torch.randn_like(tmp_mu_stack.expand(samples_per_set, -1, -1)).reshape(-1, tmp_mu_stack.shape[-1]),
                        bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0),
                    )
                    mi_tmp = float(_kl_diag_gaussian(tmp_mu_stack, tmp_logvar_stack).sum(dim=1).mean().item() - mmd_tmp.item())
                    compare_rows.append({"ckpt": Path(ckpt_path).name, "mi_proxy": mi_tmp})
            except Exception as err:  # pragma: no cover
                LOGGER.warning("Failed to compare checkpoint %s: %s", ckpt_path, err)

    if compare_rows:
        pd.DataFrame(compare_rows).to_csv(save_dir / "mi_over_ckpts.csv", index=False)

    if cfg.plots and plt is not None:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        dims = np.arange(kl_mean.numel())
        ax1.bar(dims, kl_mean.cpu().numpy(), color="#2a9d8f")
        ax1.set_xlabel("Latent dimension")
        ax1.set_ylabel("KL mean (nats)")
        ax2 = ax1.twinx()
        ax2.plot(dims, np.array([c[1] for c in coverage_curve]), color="#e76f51", linewidth=2)
        ax2.set_ylabel("Cumulative KL share")
        ax2.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(save_dir / "kl_coverage.png", dpi=200)
        plt.close(fig)

    metrics = {
        "KL_total": kl_total,
        "KL_per_dim_mean": kl_mean.cpu().tolist(),
        "KL_per_dim_std": kl_std.cpu().tolist(),
        "ActiveDims": active_dims,
        "coverage_curve": coverage_curve,
        "MI_proxy": mi_proxy,
    }
    _write_json(save_dir / "metrics.json", metrics)
    return metrics


def evaluate_latent_information_distribution(
    model: torch.nn.Module,
    cache: DataCache,
    posterior_metrics: Mapping[str, Any],
    cfg: EvalConfig,
    run_probes: bool = True,
) -> Dict[str, Any]:
    save_dir = _ensure_dir(cfg.save_dir / "info_dist")
    probe_dir = _ensure_dir(save_dir / "probe")

    kl_mean = torch.tensor(posterior_metrics.get("KL_per_dim_mean", cache.latent_matrix.mean(dim=0).tolist()))
    kl_total = float(kl_mean.sum().item())
    eps = 1e-8
    p = (kl_mean / (kl_total + eps)).cpu().numpy()
    entropy = float(-np.sum(p * np.log(p + eps))) if kl_total > 0 else float("nan")
    max_entropy = math.log(len(p)) if len(p) > 0 else float("nan")
    gini = float(1.0 - np.sum(np.minimum.outer(p, p)) * 2.0)
    _write_json(save_dir / "gini.json", {"entropy": entropy, "entropy_norm": entropy / max_entropy if max_entropy > 0 else float("nan"), "gini": gini})
    pd.DataFrame({"dim": np.arange(len(p)), "kl_share": p}).to_csv(save_dir / "kl_share.csv", index=False)

    probe_metrics = {
        "ridge_r2_mean": float("nan"),
        "binary_auc_mean": float("nan"),
        "multiclass_f1_mean": float("nan"),
    }

    if run_probes:
        latents = cache.latent_matrix
        feature_types = cache.feature_types
        cont_targets = []
        bin_targets = []
        cat_targets = []
        for sample in select_random_sets(cache, min(cfg.sample_sets, 300), cfg.seed):
            cont, binaria, categorical = _aggregate_feature_targets(sample, feature_types)
            cont_targets.append(cont)
            bin_targets.append(binaria)
            cat_targets.append(categorical)
        cont_targets = np.stack(cont_targets, axis=0)
        bin_targets = np.stack(bin_targets, axis=0)
        cat_targets = np.stack(cat_targets, axis=0)

        num_features = int(feature_types.numel())
        rng = np.random.default_rng(cfg.seed)
        idx = np.arange(latents.shape[0])
        rng.shuffle(idx)
        split = max(1, int(0.8 * len(idx)))
        train_idx = idx[:split]
        test_idx = idx[split:]
        if len(test_idx) == 0:
            test_idx = train_idx

        feature_names = []
        for schema_feat in load_schema(cfg.schema).features:
            feature_names.append(schema_feat.name or f"feat_{schema_feat.feature_id}")
        while len(feature_names) < num_features:
            feature_names.append(f"feat_{len(feature_names)}")

        ridge_rows: List[Dict[str, Any]] = []
        auc_rows: List[Dict[str, Any]] = []
        f1_rows: List[Dict[str, Any]] = []

        X_train = latents[train_idx].numpy()
        X_test = latents[test_idx].numpy()
        alpha = 1e-3
        I = np.eye(X_train.shape[1])

        for fid in range(num_features):
            tcode = int(feature_types[fid].item())
            name = feature_names[fid] if fid < len(feature_names) else f"feat_{fid}"
            if tcode == 0:
                y_train = cont_targets[train_idx, fid]
                y_test = cont_targets[test_idx, fid]
                if np.isfinite(y_train).sum() < 5 or np.isfinite(y_test).sum() < 5:
                    continue
                A = X_train.T @ X_train + alpha * I
                b = X_train.T @ y_train
                w = np.linalg.solve(A, b)
                y_pred = X_test @ w
                ss_res = np.sum((y_pred - y_test) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2) + EPS
                r2 = 1 - ss_res / ss_tot
                ridge_rows.append({"feature_id": fid, "name": name, "r2": float(r2)})
            elif tcode == 1:
                y_train = bin_targets[train_idx, fid]
                y_test = bin_targets[test_idx, fid]
                mask_train = np.isfinite(y_train)
                mask_test = np.isfinite(y_test)
                if mask_train.sum() < 10 or mask_test.sum() < 5:
                    continue
                X_tr = torch.tensor(X_train[mask_train], dtype=torch.float32)
                y_tr = torch.tensor(y_train[mask_train], dtype=torch.float32)
                X_te = torch.tensor(X_test[mask_test], dtype=torch.float32)
                y_te = torch.tensor(y_test[mask_test], dtype=torch.float32)
                w = torch.zeros(X_tr.shape[1], requires_grad=True)
                b = torch.zeros(1, requires_grad=True)
                optimizer = torch.optim.SGD([w, b], lr=0.1)
                pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / (y_tr.sum() + EPS)])
                for _ in range(100):
                    optimizer.zero_grad()
                    logits = X_tr @ w + b
                    loss = F.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=pos_weight)
                    loss.backward()
                    optimizer.step()
                probs = torch.sigmoid(X_te @ w + b).detach().cpu().numpy()
                labels = y_te.cpu().numpy()
                metrics = compute_binary_metrics(probs.tolist(), labels.tolist())
                metrics.update({"feature_id": fid, "name": name})
                auc_rows.append(metrics)
            elif tcode == 2:
                y_train = cat_targets[train_idx, fid]
                y_test = cat_targets[test_idx, fid]
                mask_train = y_train >= 0
                mask_test = y_test >= 0
                if mask_train.sum() < 10 or mask_test.sum() < 5:
                    continue
                cardinality = int(cache.cat_cardinalities[fid].item()) if cache.cat_cardinalities is not None else int(np.max(y_train[mask_train]) + 1)
                X_tr = torch.tensor(X_train[mask_train], dtype=torch.float32)
                y_tr = torch.tensor(y_train[mask_train], dtype=torch.long)
                X_te = torch.tensor(X_test[mask_test], dtype=torch.float32)
                y_te = torch.tensor(y_test[mask_test], dtype=torch.long)
                head = torch.nn.Linear(X_tr.shape[1], cardinality)
                optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
                for _ in range(150):
                    optimizer.zero_grad()
                    logits = head(X_tr)
                    loss = F.cross_entropy(logits, y_tr)
                    loss.backward()
                    optimizer.step()
                logits = head(X_te).detach().cpu().numpy()
                metrics = compute_multiclass_metrics(logits.tolist(), y_te.cpu().numpy().tolist(), [cardinality])
                metrics.update({"feature_id": fid, "name": name})
                f1_rows.append(metrics)

        if ridge_rows:
            pd.DataFrame(ridge_rows).to_csv(probe_dir / "R2_per_feature.csv", index=False)
        if auc_rows:
            pd.DataFrame(auc_rows).to_csv(probe_dir / "AUC_per_feature.csv", index=False)
        if f1_rows:
            pd.DataFrame(f1_rows).to_csv(probe_dir / "macroF1_per_feature.csv", index=False)

        probe_metrics = {
            "ridge_r2_mean": float(np.mean([row["r2"] for row in ridge_rows])) if ridge_rows else float("nan"),
            "binary_auc_mean": float(np.mean([row.get("auc", np.nan) for row in auc_rows])) if auc_rows else float("nan"),
            "multiclass_f1_mean": float(np.mean([row.get("macro_f1", np.nan) for row in f1_rows])) if f1_rows else float("nan"),
        }

    aggregate = {
        "entropy": entropy,
        "entropy_norm": entropy / max_entropy if max_entropy > 0 else float("nan"),
        "gini": gini,
        "ridge_r2_mean": probe_metrics["ridge_r2_mean"],
        "binary_auc_mean": probe_metrics["binary_auc_mean"],
        "multiclass_f1_mean": probe_metrics["multiclass_f1_mean"],
    }

    _write_json(save_dir / "metrics.json", aggregate)
    return aggregate


def _sinkhorn_plan(
    recon: torch.Tensor,
    target: torch.Tensor,
    eps: float,
    iters: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # recon/target: [1,N,D]
    recon_unit = recon / (torch.norm(recon, dim=-1, keepdim=True) + EPS)
    target_unit = target / (torch.norm(target, dim=-1, keepdim=True) + EPS)
    sim = torch.bmm(recon_unit, target_unit.transpose(1, 2)).abs()
    c_dir = 1.0 - sim
    mag_r = torch.norm(recon, dim=-1)
    mag_t = torch.norm(target, dim=-1)
    c_mag = (mag_r.unsqueeze(-1) - mag_t.unsqueeze(1)).abs()
    r_exp = recon.unsqueeze(2)
    t_exp = target.unsqueeze(1)
    d_pos = ((r_exp - t_exp) ** 2).sum(dim=-1)
    d_neg = ((r_exp + t_exp) ** 2).sum(dim=-1)
    c_vec = torch.minimum(d_pos, d_neg)
    C = c_dir + c_mag + c_vec
    K = torch.exp(-C / max(eps, EPS)).clamp_min(1e-12)
    N = K.shape[1]
    a = torch.full((1, N), 1.0 / N, device=K.device, dtype=K.dtype)
    b = torch.full((1, N), 1.0 / N, device=K.device, dtype=K.dtype)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max(1, iters)):
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    P = torch.diag_embed(u.squeeze(0)) @ K @ torch.diag_embed(v.squeeze(0))
    total_cost = float((P * C).sum().item())
    dir_cost = float((P * c_dir).sum().item())
    mag_cost = float((P * c_mag).sum().item())
    vec_cost = float((P * c_vec).sum().item())
    return P.squeeze(0), {
        "total": total_cost,
        "direction": dir_cost,
        "magnitude": mag_cost,
        "vector": vec_cost,
    }


def evaluate_reconstruction(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
    eval_prob: bool = True,
    eval_sinkhorn: bool = True,
) -> Dict[str, Any]:
    sink_dir = _ensure_dir(cfg.save_dir / "recon" / "sinkhorn") if eval_sinkhorn else None
    prob_dir = _ensure_dir(cfg.save_dir / "recon" / "probabilistic") if eval_prob else None

    rows_sink: List[Dict[str, Any]] = []
    rows_plan: List[Dict[str, Any]] = []
    cont_preds: List[float] = []
    cont_vars: List[float] = []
    cont_targets: List[float] = []
    bin_probs: List[float] = []
    bin_labels: List[float] = []
    cat_logits: List[List[float]] = []
    cat_labels: List[int] = []
    cat_cards: List[int] = []

    device = next(model.parameters()).device
    sampled_sets = select_random_sets(cache, cfg.sample_sets, cfg.seed)
    for sample in sampled_sets:
        if eval_sinkhorn:
            s_dev = _clone_set_to_device(sample, device)
            with torch.no_grad():
                with _amp_autocast():
                    v_norm, target = model._compute_target(s_dev)
                    if hasattr(model, "_build_token_inputs") and getattr(model, "num_features", 0) > 0:
                        x_input = model._build_token_inputs(v_norm, s_dev)
                    else:
                        x_input = v_norm * s_dev["val"]
                    z_list, _ = model.set_encoder.encode(x_input)
                    recon = model.set_encoder.decode(z_list, target_n=target.size(1), use_mean=True, noise_std=0.0)

            plan, costs = _sinkhorn_plan(recon.detach(), target.detach(), getattr(model, "sinkhorn_eps", 0.1), getattr(model, "sinkhorn_iters", 100))
            topk = torch.topk(plan.flatten(), min(10, plan.numel())).values.sum().item()
            rows_sink.append({"uid": sample.uid, **costs})
            rows_plan.append({"uid": sample.uid, "top10_mass": float(topk), "total_mass": float(plan.sum().item())})

        if not eval_prob:
            continue

        head = _get_head_outputs(model, sample, device)
        if head is None:
            continue
        cont_mu, cont_logvar, bin_logit, cat_logits_head = head
        feat_id = sample.tensors.get("feat_id")
        values = sample.tensors.get("val")
        if feat_id is None or values is None:
            continue
        fid = feat_id.squeeze(0).squeeze(-1).numpy().astype(int)
        val = values.squeeze(0).squeeze(-1).numpy()
        for idx, f in enumerate(fid):
            if f < 0 or f >= cache.feature_types.numel():
                continue
            tcode = int(cache.feature_types[f].item())
            if tcode == 0:
                cont_preds.append(float(cont_mu[0, f].item()))
                cont_vars.append(float(torch.exp(cont_logvar[0, f]).item()))
                cont_targets.append(float(val[idx]))
            elif tcode == 1:
                prob = float(torch.sigmoid(bin_logit[0, f]).item())
                bin_probs.append(prob)
                bin_labels.append(1.0 if val[idx] > 0.5 else 0.0)
            elif tcode == 2 and cat_logits_head is not None and cache.cat_offsets is not None:
                offset = int(cache.cat_offsets[f].item())
                card = int(cache.cat_cardinalities[f].item()) if cache.cat_cardinalities is not None else 0
                if offset >= 0 and card > 0:
                    logits = cat_logits_head[0, offset : offset + card].detach().cpu().numpy().tolist()
                    cat_logits.append(logits)
                    cat_labels.append(int(max(0, min(card - 1, int(round(val[idx]))))))
                    cat_cards.append(card)

    sink_metrics: Dict[str, Any] = {}
    if eval_sinkhorn and rows_sink:
        if sink_dir is not None:
            pd.DataFrame(rows_sink).to_csv(sink_dir / "ot_components.csv", index=False)
            if rows_plan:
                pd.DataFrame(rows_plan).to_csv(sink_dir / "plan_peakedness.csv", index=False)
        sink_metrics = merge_metrics([{k: v for k, v in row.items() if k != "uid"} for row in rows_sink])
        if rows_plan:
            sink_metrics["plan_top10_share"] = _safe_mean([row["top10_mass"] / max(row["total_mass"], EPS) for row in rows_plan])

    prob_metrics: Dict[str, Any] = {}
    if eval_prob:
        if cont_preds:
            cont_metric = compute_continuous_metrics(cont_preds, cont_vars, cont_targets)
            prob_metrics["continuous"] = {k: v for k, v in cont_metric.items() if k != "pit_values"}
            if cfg.plots and plt is not None and prob_dir is not None and cfg.tier == "C":
                plt.figure(figsize=(4, 3))
                plt.hist(cont_metric.get("pit_values", []), bins=20, range=(0, 1))
                plt.title("Continuous PIT")
                plt.savefig(prob_dir / "continuous_pit.png", dpi=150)
                plt.close()
        if bin_probs:
            prob_metrics["binary"] = compute_binary_metrics(bin_probs, bin_labels)
        if cat_logits:
            prob_metrics["categorical"] = compute_multiclass_metrics(cat_logits, cat_labels, cat_cards)
        if prob_dir is not None:
            _write_json(prob_dir / "metrics.json", prob_metrics)

    if eval_sinkhorn and sink_dir is not None:
        _write_json(sink_dir / "metrics.json", sink_metrics)

    return {
        "sinkhorn": sink_metrics,
        "probabilistic": prob_metrics,
    }


def evaluate_conditional_inference(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    scenarios = cfg.mask_scenarios or ["none"]
    base_dir = _ensure_dir(cfg.save_dir / "cond_infer")
    curves_dir = _ensure_dir(base_dir / "curves")
    registry = MaskRegistry()
    device = next(model.parameters()).device

    scenario_metrics: Dict[str, Any] = {}
    rng = random.Random(cfg.seed)

    for scenario in scenarios:
        scen_dir = _ensure_dir(base_dir / scenario.replace(".", "_"))
        rows = []
        cont_preds: List[float] = []
        cont_vars: List[float] = []
        cont_targets: List[float] = []
        bin_probs: List[float] = []
        bin_labels: List[float] = []
        cat_logits: List[List[float]] = []
        cat_labels: List[int] = []
        cat_cards: List[int] = []

        for sample in select_random_sets(cache, cfg.sample_sets, cfg.seed + scenarios.index(scenario)):
            keep_idx, mask_idx = apply_mask_scenario(sample, scenario, rng)
            values = sample.tensors.get("val")
            feat_id_full = sample.tensors.get("feat_id")
            if values is None or feat_id_full is None:
                continue

            if scenario == "none":
                head_full = _get_head_outputs(model, sample, device)
                if head_full is None:
                    continue
                cont_mu, cont_logvar, bin_logit, cat_logits_head = head_full
                fid_full = feat_id_full.squeeze(0).squeeze(-1).cpu().numpy().astype(int)
                val_full = values.squeeze(0).squeeze(-1).numpy()
                for local, f in enumerate(fid_full):
                    if f < 0 or f >= cache.feature_types.numel():
                        continue
                    tcode = int(cache.feature_types[f].item())
                    if tcode == 0:
                        cont_preds.append(float(cont_mu[0, f].item()))
                        cont_vars.append(float(torch.exp(cont_logvar[0, f]).item()))
                        cont_targets.append(float(val_full[local]))
                    elif tcode == 1:
                        prob = float(torch.sigmoid(bin_logit[0, f]).item())
                        bin_probs.append(prob)
                        bin_labels.append(1.0 if val_full[local] > 0.5 else 0.0)
                    elif tcode == 2 and cat_logits_head is not None and cache.cat_offsets is not None:
                        offset = int(cache.cat_offsets[f].item())
                        card = int(cache.cat_cardinalities[f].item()) if cache.cat_cardinalities is not None else 0
                        if offset >= 0 and card > 0:
                            logits = cat_logits_head[0, offset : offset + card].detach().cpu().numpy().tolist()
                            cat_logits.append(logits)
                            cat_labels.append(int(max(0, min(card - 1, int(round(val_full[local]))))))
                            cat_cards.append(card)
                continue

            if mask_idx.numel() == 0:
                continue
            registry.register(scenario, mask_idx.numpy())
            observed = _clone_set_to_device(sample, device, keep_idx)
            masked = _clone_set_to_device(sample, device, mask_idx)
            with torch.no_grad():
                mu_cond, logvar_cond = _encode_set_latent(model, observed)
            head = _predict_head_outputs(model, mu_cond)
            if head is None:
                continue
            cont_mu, cont_logvar, bin_logit, cat_logits_head = head
            fid = masked.get("feat_id")
            if fid is None:
                continue
            fid_np = fid.squeeze(0).squeeze(-1).cpu().numpy().astype(int)
            val = values.squeeze(0).squeeze(-1).numpy()[mask_idx.numpy()]
            for local, f in enumerate(fid_np):
                if f < 0 or f >= cache.feature_types.numel():
                    continue
                tcode = int(cache.feature_types[f].item())
                if tcode == 0:
                    cont_preds.append(float(cont_mu[0, f].item()))
                    cont_vars.append(float(torch.exp(cont_logvar[0, f]).item()))
                    cont_targets.append(float(val[local]))
                elif tcode == 1:
                    prob = float(torch.sigmoid(bin_logit[0, f]).item())
                    bin_probs.append(prob)
                    bin_labels.append(1.0 if val[local] > 0.5 else 0.0)
                elif tcode == 2 and cat_logits_head is not None and cache.cat_offsets is not None:
                    offset = int(cache.cat_offsets[f].item())
                    card = int(cache.cat_cardinalities[f].item()) if cache.cat_cardinalities is not None else 0
                    if offset >= 0 and card > 0:
                        logits = cat_logits_head[0, offset : offset + card].detach().cpu().numpy().tolist()
                        cat_logits.append(logits)
                        cat_labels.append(int(max(0, min(card - 1, int(round(val[local]))))))
                        cat_cards.append(card)

        metrics: Dict[str, Any] = {}
        if cont_preds:
            metrics["continuous"] = {k: v for k, v in compute_continuous_metrics(cont_preds, cont_vars, cont_targets).items() if k != "pit_values"}
        if bin_probs:
            metrics["binary"] = compute_binary_metrics(bin_probs, bin_labels)
        if cat_logits:
            metrics["categorical"] = compute_multiclass_metrics(cat_logits, cat_labels, cat_cards)
        _write_json(scen_dir / "metrics.json", metrics)
        scenario_metrics[scenario] = metrics

    registry.dump(base_dir)

    # Degradation curves (example: binary AUC vs scenario)
    curve_rows = []
    for scenario, metrics in scenario_metrics.items():
        entry = {"scenario": scenario}
        entry["auc_bin"] = metrics.get("binary", {}).get("auc")
        entry["rmse_cont"] = metrics.get("continuous", {}).get("rmse")
        entry["accuracy_cat"] = metrics.get("categorical", {}).get("accuracy")
        curve_rows.append(entry)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(curves_dir / "degradation.csv", index=False)

    return scenario_metrics


def evaluate_active_observation(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
    mc_samples: int = 16,
) -> Dict[str, Any]:
    base_dir = _ensure_dir(cfg.save_dir / "active_obs")
    proxy_path = base_dir / "proxy_rank.csv"
    delta_path = base_dir / "delta_unc_rank.csv"

    feature_types = cache.feature_types
    proxy_scores: Dict[int, List[float]] = defaultdict(list)
    delta_scores: Dict[int, List[float]] = defaultdict(list)

    device = next(model.parameters()).device
    mask_rng = random.Random(cfg.seed)

    for sample in select_random_sets(cache, cfg.sample_sets, cfg.seed):
        keep_idx, mask_idx = apply_mask_scenario(sample, "carry_only", mask_rng)
        observed = _clone_set_to_device(sample, device, keep_idx)
        with torch.no_grad():
            mu_base, logvar_base = _encode_set_latent(model, observed)
        head_base = _get_head_outputs(model, sample, device)
        if head_base is None:
            continue
        feat_id = sample.tensors.get("feat_id")
        values = sample.tensors.get("val")
        if feat_id is None or values is None:
            continue
        fid = feat_id.squeeze(0).squeeze(-1).numpy().astype(int)
        val = values.squeeze(0).squeeze(-1).numpy()
        full_set = _clone_set_to_device(sample, device)

        ranked = []
        for idx, f in enumerate(fid):
            if f < 0 or f >= feature_types.numel():
                continue
            if idx in keep_idx.tolist():
                continue
            score = compute_token_uncertainty(model, head_base, f, feature_types)
            ranked.append((idx, f, score))

        if not ranked:
            continue
        ranked.sort(key=lambda x: x[2], reverse=True)
        top_features = ranked[: cfg.active_topk]
        for idx, f, score in top_features:
            proxy_scores[f].append(score)

            # Monte Carlo delta uncertainty
            tcode = int(feature_types[f].item())
            mu_samples = []
            for _ in range(min(mc_samples, cfg.mc_samples)):
                if tcode == 0:
                    pred_mu = float(head_base[0][0, f].item())  # type: ignore[index]
                    pred_sigma = math.sqrt(float(torch.exp(head_base[1][0, f]).item()))  # type: ignore[index]
                    sampled_val = random.gauss(pred_mu, pred_sigma)
                elif tcode == 1:
                    prob = float(torch.sigmoid(head_base[2][0, f]).item())  # type: ignore[index]
                    sampled_val = 1.0 if random.random() < prob else 0.0
                else:
                    if head_base[3] is None or cache.cat_offsets is None:
                        continue
                    offset = int(cache.cat_offsets[f].item())
                    card = int(cache.cat_cardinalities[f].item()) if cache.cat_cardinalities is not None else 0
                    if offset < 0 or card <= 0:
                        continue
                    probs = torch.softmax(head_base[3][0, offset : offset + card], dim=-1).cpu().numpy()  # type: ignore[index]
                    sampled_val = int(np.random.choice(np.arange(card), p=probs))

                augmented = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in full_set.items()}
                augmented["val"] = augmented["val"].clone()
                augmented["val"][:, idx : idx + 1, :] = torch.tensor(sampled_val, dtype=augmented["val"].dtype, device=device).view(1, 1, 1)
                with torch.no_grad():
                    mu_aug, logvar_aug = _encode_set_latent(model, augmented)
                mu_samples.append((mu_aug, logvar_aug))

            if not mu_samples:
                continue
            base_unc = float(torch.exp(logvar_base).sum().item())
            aug_unc = []
            for mu_aug, logvar_aug in mu_samples:
                aug_unc.append(float(torch.exp(logvar_aug).sum().item()))
            delta_scores[f].append(base_unc - float(np.mean(aug_unc)))

    proxy_rank = []
    delta_rank = []
    for fid, scores in proxy_scores.items():
        proxy_rank.append({"feature_id": fid, "score": float(np.mean(scores))})
    for fid, scores in delta_scores.items():
        delta_rank.append({"feature_id": fid, "score": float(np.mean(scores))})

    if proxy_rank:
        pd.DataFrame(proxy_rank).sort_values("score", ascending=False).to_csv(proxy_path, index=False)
    if delta_rank:
        pd.DataFrame(delta_rank).sort_values("score", ascending=False).to_csv(delta_path, index=False)

    joined = []
    for fid in set(list(proxy_scores.keys()) + list(delta_scores.keys())):
        proxy_mean = float(np.mean(proxy_scores.get(fid, [float("nan")])))
        delta_mean = float(np.mean(delta_scores.get(fid, [float("nan")])))
        joined.append((proxy_mean, delta_mean))
    if joined:
        a = np.array([j[0] for j in joined])
        b = np.array([j[1] for j in joined])
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            rho = _spearman_corr(a[mask].tolist(), b[mask].tolist())
        else:
            rho = float("nan")
    else:
        rho = float("nan")

    corr = {"spearman": rho, "count": int(len(joined))}
    _write_json(base_dir / "corr.json", corr)

    return {
        "proxy": proxy_rank,
        "delta_unc": delta_rank,
        "correlation": corr,
    }


def evaluate_ig_monotonicity(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
    topk: int,
) -> Dict[str, Any]:
    save_dir = _ensure_dir(cfg.save_dir / "ig_monotonic")
    curves_path = save_dir / "curves.csv"
    stats_path = save_dir / "stats.json"
    device = next(model.parameters()).device

    curves: List[List[float]] = []
    w2_values: List[List[float]] = []
    stats_rows: List[Dict[str, Any]] = []

    for sample in select_random_sets(cache, cfg.sample_sets, cfg.seed):
        full = _clone_set_to_device(sample, device)
        with torch.no_grad():
            mu_full, logvar_full = _encode_set_latent(model, full)
        head_full = _predict_head_outputs(model, mu_full)
        if head_full is None:
            continue

        feat_id = sample.tensors.get("feat_id")
        if feat_id is None:
            continue
        fid = feat_id.squeeze(0).squeeze(-1)
        tokens = list(range(fid.shape[0]))
        scores = []
        for idx, f in enumerate(fid.cpu().numpy().astype(int)):
            if f < 0 or f >= cache.feature_types.numel():
                continue
            scores.append((idx, compute_token_uncertainty(model, head_full, f, cache.feature_types)))
        if not scores:
            continue
        scores.sort(key=lambda x: x[1], reverse=True)
        steps = min(topk, len(scores))

        keep_tokens = tokens.copy()
        curve = [1.0]
        w2_curve = [0.0]
        for step in range(steps):
            remove_idx = scores[step][0]
            if remove_idx in keep_tokens:
                keep_tokens.remove(remove_idx)
            if not keep_tokens:
                break
            subset = _clone_set_to_device(sample, device, torch.tensor(keep_tokens, dtype=torch.long))
            with torch.no_grad():
                mu_sub, logvar_sub = _encode_set_latent(model, subset)
            cos = F.cosine_similarity(mu_full.unsqueeze(0), mu_sub.unsqueeze(0), dim=-1).item()
            curve.append(float(max(min(cos, 1.0), -1.0)))
            w2 = torch.norm(mu_full - mu_sub).item() + float(torch.abs(torch.exp(logvar_full) - torch.exp(logvar_sub)).mean().item())
            w2_curve.append(w2)

        if len(curve) <= 1:
            continue
        curves.append(curve)
        w2_values.append(w2_curve)
        k_vals = list(range(len(curve)))
        spearman = _spearman_corr(k_vals, curve)
        tau, _ = _kendall_tau_b(k_vals, curve)
        auc_decay = float(np.mean(curve))
        stats_rows.append({
            "uid": sample.uid,
            "spearman": spearman,
            "kendall_tau": tau,
            "auc_decay": auc_decay,
            "delta@1": float(curve[0] - curve[1]) if len(curve) > 1 else float("nan"),
            "delta@3": float(curve[0] - curve[min(3, len(curve) - 1)]),
            "delta@5": float(curve[0] - curve[min(5, len(curve) - 1)]),
        })

    if curves:
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, curve in enumerate(curves):
            padded[i, : len(curve)] = curve
        mean_curve = np.nanmean(padded, axis=0)
        pd.DataFrame({"step": np.arange(len(mean_curve)), "mean_cos": mean_curve}).to_csv(curves_path, index=False)
    stats = {
        "sets_evaluated": len(curves),
        "avg_kendall_tau": _safe_mean([row["kendall_tau"] for row in stats_rows]) if stats_rows else float("nan"),
        "avg_spearman": _safe_mean([row["spearman"] for row in stats_rows]) if stats_rows else float("nan"),
        "avg_auc_decay": _safe_mean([row["auc_decay"] for row in stats_rows]) if stats_rows else float("nan"),
    }
    _write_json(stats_path, stats)
    return {"curves": curves, "stats": stats}


def evaluate_intra_set_consistency(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    base_dir = _ensure_dir(cfg.save_dir / "intraconsistency")
    pit_path = base_dir / "pit.csv"
    ks_path = base_dir / "ks.json"
    ece_path = base_dir / "ece.csv"
    coverage_path = base_dir / "coverage.csv"

    device = next(model.parameters()).device

    pit_values: List[float] = []
    ks_rows = []
    ece_rows = []
    coverage_rows = []

    for sample in select_random_sets(cache, cfg.sample_sets, cfg.seed):
        feat_id = sample.tensors.get("feat_id")
        values = sample.tensors.get("val")
        if feat_id is None or values is None:
            continue
        fid = feat_id.squeeze(0).squeeze(-1)
        val = values.squeeze(0).squeeze(-1)
        for idx in range(fid.shape[0]):
            keep = torch.tensor([i for i in range(fid.shape[0]) if i != idx], dtype=torch.long)
            masked = torch.tensor([idx], dtype=torch.long)
            observed = _clone_set_to_device(sample, device, keep)
            with torch.no_grad():
                mu_cond, logvar_cond = _encode_set_latent(model, observed)
            head = _predict_head_outputs(model, mu_cond)
            if head is None:
                continue
            f = int(fid[idx].item())
            if f < 0 or f >= cache.feature_types.numel():
                continue
            tcode = int(cache.feature_types[f].item())
            target_val = float(val[idx].item())
            if tcode == 0:
                mu = float(head[0][0, f].item())
                sigma = math.sqrt(float(torch.exp(head[1][0, f]).item()))
                z_score = (target_val - mu) / max(sigma, EPS)
                pit = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
                pit_values.append(pit)
            elif tcode == 1:
                prob = float(torch.sigmoid(head[2][0, f]).item())
                ece_rows.append({"feature_id": f, "prob": prob, "label": 1.0 if target_val > 0.5 else 0.0})
            elif tcode == 2 and head[3] is not None and cache.cat_offsets is not None:
                offset = int(cache.cat_offsets[f].item())
                card = int(cache.cat_cardinalities[f].item()) if cache.cat_cardinalities is not None else 0
                if offset >= 0 and card > 0:
                    probs = torch.softmax(head[3][0, offset : offset + card], dim=-1).cpu().numpy()
                    one_hot = np.zeros(card)
                    klass = int(max(0, min(card - 1, int(round(target_val)))))
                    one_hot[klass] = 1.0
                    ece_rows.append({"feature_id": f, "prob": float(probs.max()), "label": 1.0 if probs.argmax() == klass else 0.0})

    if pit_values:
        df_pit = pd.DataFrame({"pit": pit_values})
        df_pit.to_csv(pit_path, index=False)
        ks_stat, ks_p = _ks_test_uniform(pit_values)
    else:
        df_pit = pd.DataFrame()
        ks_stat, ks_p = float("nan"), float("nan")

    if cfg.tier != "A":
        coverage_rows.append({
            "level": 0.5,
            "coverage": _coverage_rate(pit_values, 0.5),
        })
        coverage_rows.append({
            "level": 0.8,
            "coverage": _coverage_rate(pit_values, 0.8),
        })
        coverage_rows.append({
            "level": 0.9,
            "coverage": _coverage_rate(pit_values, 0.9),
        })
        coverage_rows.append({
            "level": 0.95,
            "coverage": _coverage_rate(pit_values, 0.95),
        })
        pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)

    if ece_rows:
        pd.DataFrame(ece_rows).to_csv(ece_path, index=False)

    ks_summary = {"ks_stat": ks_stat, "ks_p": ks_p}
    _write_json(ks_path, ks_summary)
    return {"pit": pit_values, "ks": ks_summary, "ece_rows": len(ece_rows)}


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + EPS
    return (matrix @ matrix.T) / (norm * norm.T)


def _greedy_cluster(sim_matrix: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim_matrix.shape[0]
    unassigned = set(range(n))
    clusters: List[List[int]] = []
    while unassigned:
        idx = unassigned.pop()
        cluster = [idx]
        similar = [j for j in list(unassigned) if sim_matrix[idx, j] >= threshold]
        for j in similar:
            cluster.append(j)
            unassigned.remove(j)
        clusters.append(sorted(cluster))
    return clusters


def _ari_score(labels_true: List[int], labels_pred: List[int]) -> float:
    from collections import Counter

    n = len(labels_true)
    if n <= 1:
        return float("nan")
    label_pair = Counter()
    for t, p in zip(labels_true, labels_pred):
        label_pair[(t, p)] += 1
    sum_comb_c = sum(v * (v - 1) / 2 for v in Counter(labels_true).values())
    sum_comb_k = sum(v * (v - 1) / 2 for v in Counter(labels_pred).values())
    sum_comb = sum(v * (v - 1) / 2 for v in label_pair.values())
    expected = sum_comb_c * sum_comb_k / (n * (n - 1) / 2)
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if denom == 0:
        return float("nan")
    return float((sum_comb - expected) / denom)


def _nmi_score(labels_true: List[int], labels_pred: List[int]) -> float:
    from collections import Counter

    n = len(labels_true)
    if n == 0:
        return float("nan")
    counts_true = Counter(labels_true)
    counts_pred = Counter(labels_pred)
    joint = Counter(zip(labels_true, labels_pred))

    def entropy(counts: Counter) -> float:
        total = float(sum(counts.values()))
        if total <= 0:
            return 0.0
        return -sum((c / total) * math.log((c + EPS) / total) for c in counts.values())

    h_true = entropy(counts_true)
    h_pred = entropy(counts_pred)
    if h_true <= 0 or h_pred <= 0:
        return float("nan")

    mi = 0.0
    for (t, p), c in joint.items():
        p_tp = c / n
        mi += p_tp * math.log((p_tp + EPS) / ((counts_true[t] / n) * (counts_pred[p] / n) + EPS))
    return float(mi / math.sqrt(h_true * h_pred))


def analyze_subsystems(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    save_dir = _ensure_dir(cfg.save_dir / "subsystems")
    weights_path = save_dir / "weights_cluster.json"
    sens_path = save_dir / "sens_cluster.json"
    heatmap_path = save_dir / "weights_heatmap.png"
    consistency_path = save_dir / "cluster_consistency.json"

    feature_types = cache.feature_types
    num_features = int(feature_types.numel())
    vectors = []
    for fid in range(num_features):
        parts = []
        if getattr(model, "prob_cont_mu", None) is not None:
            parts.append(model.prob_cont_mu.weight[:, fid].detach().cpu().numpy())
            parts.append(model.prob_cont_logvar.weight[:, fid].detach().cpu().numpy())
        if getattr(model, "prob_bin_logit", None) is not None:
            parts.append(model.prob_bin_logit.weight[:, fid].detach().cpu().numpy())
        if getattr(model, "prob_cat_logits", None) is not None and cache.cat_offsets is not None:
            offset = int(cache.cat_offsets[fid].item())
            card = int(cache.cat_cardinalities[fid].item()) if cache.cat_cardinalities is not None else 0
            if offset >= 0 and card > 0:
                parts.append(model.prob_cat_logits.weight[:, offset : offset + card].mean(dim=1).detach().cpu().numpy())
        if parts:
            vectors.append(np.concatenate(parts))
        else:
            vectors.append(np.zeros(8))
    vectors_np = np.stack(vectors, axis=0)
    sim_matrix = _cosine_similarity_matrix(vectors_np)
    clusters = _greedy_cluster(sim_matrix, threshold=0.7)
    _write_json(weights_path, {"clusters": clusters})

    if cfg.plots and plt is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(sim_matrix, cmap="viridis")
        plt.colorbar(label="Cosine similarity")
        plt.title("Probability head weight similarity")
        plt.savefig(heatmap_path, dpi=200)
        plt.close()

    # Sensitivity clustering
    eps = 0.1
    sens_vectors = np.zeros((num_features, cache.latent_matrix.shape[1]))
    device = next(model.parameters()).device
    count = 0
    for sample in select_random_sets(cache, min(cfg.sample_sets, 200), cfg.seed):
        mu = sample.mu.to(device)
        head = _predict_head_outputs(model, mu)
        if head is None:
            continue
        count += 1
        for dim in range(mu.shape[0]):
            mu_plus = mu.clone()
            mu_minus = mu.clone()
            mu_plus[dim] += eps
            mu_minus[dim] -= eps
            head_plus = _predict_head_outputs(model, mu_plus)
            head_minus = _predict_head_outputs(model, mu_minus)
            if head_plus is None or head_minus is None:
                continue
            diff = (head_plus[0] - head_minus[0]).abs().detach().cpu().numpy()  # type: ignore[index]
            sens_vectors[:, dim] += diff.squeeze()
    if count > 0:
        sens_vectors /= count
    sens_sim = _cosine_similarity_matrix(sens_vectors)
    sens_clusters = _greedy_cluster(sens_sim, threshold=0.6)
    _write_json(sens_path, {"clusters": sens_clusters})

    # Consistency metrics
    weight_labels = {}
    for idx, cluster in enumerate(clusters):
        for fid in cluster:
            weight_labels[fid] = idx
    sens_labels = {}
    for idx, cluster in enumerate(sens_clusters):
        for fid in cluster:
            sens_labels[fid] = idx
    labels_true = [weight_labels.get(fid, -1) for fid in range(num_features)]
    labels_pred = [sens_labels.get(fid, -1) for fid in range(num_features)]
    ari = _ari_score(labels_true, labels_pred)
    nmi = _nmi_score(labels_true, labels_pred)
    corr = {"ARI": ari, "NMI": nmi}
    _write_json(consistency_path, corr)

    return {
        "weights_clusters": clusters,
        "sensitivity_clusters": sens_clusters,
        "consistency": corr,
    }


def evaluate_prior_alignment(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    save_dir = _ensure_dir(cfg.save_dir / "prior_vs_post")
    mmd_path = save_dir / "mmd.csv"
    umap_path = save_dir / "z_umap.png"

    mu_stack = cache.latent_matrix
    logvar_stack = torch.stack([s.logvar for s in cache.sets], dim=0)
    generator = torch.Generator(device=mu_stack.device)
    generator.manual_seed(cfg.seed)
    q_samples = _sample_diag_gaussians(mu_stack, logvar_stack, max(1, cfg.n_probe_z), generator=generator).reshape(-1, mu_stack.shape[-1])
    p_samples = torch.randn_like(q_samples, generator=generator)
    mmd = float(_mmd_rbf(q_samples, p_samples, bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0)).item())
    pd.DataFrame({"mmd": [mmd]}).to_csv(mmd_path, index=False)

    # 2D embedding for visualization
    q_np = q_samples.cpu().numpy()
    p_np = p_samples.cpu().numpy()
    combined = np.concatenate([q_np, p_np], axis=0)
    labels = np.array(["posterior"] * len(q_np) + ["prior"] * len(p_np))
    if umap is not None:
        reducer = umap.UMAP(random_state=cfg.seed)
        embedding = reducer.fit_transform(combined)
    else:
        # Fallback to PCA via SVD
        centered = combined - combined.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        embedding = centered @ Vt[:2].T
    if cfg.plots and plt is not None:
        plt.figure(figsize=(5, 4))
        mask_post = labels == "posterior"
        mask_prior = labels == "prior"
        plt.scatter(embedding[mask_post, 0], embedding[mask_post, 1], s=8, alpha=0.6, label="posterior")
        plt.scatter(embedding[mask_prior, 0], embedding[mask_prior, 1], s=8, alpha=0.6, label="prior")
        plt.legend()
        plt.title("Prior vs Posterior Latent Embedding")
        plt.savefig(umap_path, dpi=200)
        plt.close()

    return {"mmd": mmd}


def evaluate_robustness_ablation(
    model: torch.nn.Module,
    cache: DataCache,
    cfg: EvalConfig,
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    modes = modes or ROBUST_DEFAULT_MODES
    save_dir = _ensure_dir(cfg.save_dir / "robustness")
    ablation_path = save_dir / "ablations.csv"
    delta_path = save_dir / "delta_metrics.csv"

    baseline = evaluate_reconstruction(model, cache, cfg)
    baseline_prob = baseline.get("probabilistic", {})
    baseline_sink = baseline.get("sinkhorn", {})

    rows = []
    delta_rows = []
    for mode in modes:
        cloned = model  # reuse same model; modify evaluation behaviour only
        metrics = {}
        if mode == "train_noise_on_infer_on":
            original = getattr(cloned, "eval_decoder_noise_std", 0.0)
            setattr(cloned, "eval_decoder_noise_std", max(0.05, float(original)))
            metrics = evaluate_reconstruction(cloned, cache, cfg)
            setattr(cloned, "eval_decoder_noise_std", original)
        elif mode == "train_noise_off_infer_on":
            original = getattr(cloned, "train_decoder_noise_std", 0.0)
            setattr(cloned, "train_decoder_noise_std", 0.0)
            setattr(cloned, "eval_decoder_noise_std", 0.1)
            metrics = evaluate_reconstruction(cloned, cache, cfg)
            setattr(cloned, "train_decoder_noise_std", original)
            setattr(cloned, "eval_decoder_noise_std", 0.0)
        elif mode == "head_sinkhorn_only":
            metrics = {"sinkhorn": evaluate_reconstruction(cloned, cache, cfg)["sinkhorn"]}
        elif mode == "head_prob_only":
            metrics = {"probabilistic": evaluate_reconstruction(cloned, cache, cfg)["probabilistic"]}
        else:
            metrics = baseline

        rows.append({
            "mode": mode,
            "sinkhorn_total": metrics.get("sinkhorn", {}).get("total"),
            "prob_nll": metrics.get("probabilistic", {}).get("continuous", {}).get("nll"),
            "prob_auc": metrics.get("probabilistic", {}).get("binary", {}).get("auc"),
            "prob_ece": metrics.get("probabilistic", {}).get("binary", {}).get("ece"),
        })
        delta_rows.append({
            "mode": mode,
            "delta_sinkhorn": (metrics.get("sinkhorn", {}).get("total") or 0.0) - (baseline_sink.get("total") or 0.0),
            "delta_nll": (metrics.get("probabilistic", {}).get("continuous", {}).get("nll") or 0.0) - (baseline_prob.get("continuous", {}).get("nll") or 0.0),
            "delta_auc": (metrics.get("probabilistic", {}).get("binary", {}).get("auc") or 0.0) - (baseline_prob.get("binary", {}).get("auc") or 0.0),
        })

    if rows:
        pd.DataFrame(rows).to_csv(ablation_path, index=False)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(delta_path, index=False)

    delta_mean = _safe_mean([abs(row.get("delta_sinkhorn", 0.0)) for row in delta_rows]) if delta_rows else float("nan")

    return {"ablations": rows, "deltas": delta_rows, "delta_mean": delta_mean}


# ---------------------------------------------------------------------------
# Summaries & CLI
# ---------------------------------------------------------------------------


SUMMARY_THRESHOLDS = {
    "posterior.KL_total": ("<", 800.0),
    "posterior.ActiveDims.@0.1": (">", 64),
    "reconstruction.probabilistic.binary.auc": (">", 0.7),
    "reconstruction.probabilistic.binary.ece": ("<", 0.15),
    "reconstruction.sinkhorn.total": ("<", 50.0),
    "conditional.none.binary.auc": (">", 0.65),
    "ig.stats.avg_kendall_tau": (">", 0.0),
    "IG_AUCdecay": (">", 0.0),
    "ActiveObs_corr": (">", 0.0),
    "prior.mmd": ("<", 0.5),
    "Subsystem_ARI": (">", 0.2),
    "Robust_delta": ("<", 5.0),
}


def _flatten_metrics(prefix: str, data: Any, out: Dict[str, Any]) -> None:
    if isinstance(data, Mapping):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(new_prefix, value, out)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        out[prefix] = list(data)
    else:
        out[prefix] = data


def summarize_dashboard(results: Dict[str, Any], cfg: EvalConfig) -> None:
    summary_dir = _ensure_dir(cfg.save_dir / "summary")
    flat: Dict[str, Any] = {}
    for key, value in results.items():
        _flatten_metrics(key, value, flat)
    pd.DataFrame([flat]).to_csv(summary_dir / "eval_summary.csv", index=False)
    _write_json(summary_dir / "eval_summary.json", flat)

    pass_fail: Dict[str, bool] = {}
    for metric, (op, threshold) in SUMMARY_THRESHOLDS.items():
        if metric.startswith("reconstruction.sinkhorn") and not cfg.eval_sinkhorn:
            continue
        if (metric.startswith("ig.") or metric.startswith("IG_")) and not cfg.eval_ig:
            continue
        if (metric.startswith("ActiveObs") or metric.startswith("active_observation")) and not cfg.eval_active:
            continue
        if (metric.startswith("prior") or metric.startswith("MMD")) and not cfg.eval_prior:
            continue
        if metric.startswith("Subsystem") and not cfg.eval_subsystems:
            continue
        if metric.startswith("Robust") and not cfg.eval_robustness:
            continue
        value = flat.get(metric)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            pass_fail[metric] = False
            continue
        if op == "<":
            pass_fail[metric] = value < threshold
        elif op == ">":
            pass_fail[metric] = value > threshold
        else:
            pass_fail[metric] = False
    _write_json(summary_dir / "pass_fail.json", pass_fail)


def build_evaluation_config(args: argparse.Namespace) -> EvalConfig:
    save_dir = Path(args.save_dir)
    tier = str(args.tier).upper()
    if tier not in {"A", "B", "C"}:
        raise ValueError(f"Unsupported tier: {tier}")

    plots_flag = _str2bool(args.plots)
    save_raw = _str2bool(args.save_raw)

    sample_sets = args.sample_sets if args.sample_sets > 0 else {"A": 150, "B": 300, "C": 400}[tier]
    intra_samples = args.intra_samples if args.intra_samples > 0 else {"A": 100, "B": 200, "C": 300}[tier]
    mc_samples = args.mc_samples if args.mc_samples > 0 else {"A": 8, "B": 12, "C": 16}[tier]

    if args.mask_scenarios:
        mask_scenarios = parse_mask_scenarios(args.mask_scenarios)
    else:
        defaults = ["none", "mar_0.2", "mar_0.5"]
        if tier == "C":
            defaults.append("mar_0.8")
        mask_scenarios = defaults

    compare_ckpts = [s.strip() for s in args.compare_ckpts.split(",") if s.strip()] if args.compare_ckpts else []
    if tier == "A":
        compare_ckpts = []

    robust_modes = [s.strip() for s in args.robust_modes.split(",") if s.strip()] if args.robust_modes else []
    if tier != "C":
        robust_modes = []
    elif not robust_modes:
        robust_modes = ROBUST_DEFAULT_MODES

    eval_sinkhorn = tier in {"B", "C"}
    eval_probes = tier in {"B", "C"}
    eval_active = tier in {"B", "C"}
    eval_ig = tier in {"B", "C"}
    eval_prior = tier in {"B", "C"}
    eval_subsystems = tier == "C"
    eval_robustness = tier == "C"

    active_topk = args.active_topk if args.active_topk > 0 else 10

    return EvalConfig(
        ckpt=args.ckpt,
        schema=args.schema,
        data_dir=args.data_dir,
        save_dir=save_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        n_probe_z=args.n_probe_z,
        plots=plots_flag and tier == "C",
        mask_scenarios=mask_scenarios,
        topk_ig=args.topk_IG,
        sample_sets=sample_sets,
        save_raw=save_raw,
        compare_ckpts=compare_ckpts,
        robust_modes=robust_modes,
        tier=tier,
        eval_sinkhorn=eval_sinkhorn,
        eval_probes=eval_probes,
        eval_active=eval_active,
        eval_ig=eval_ig,
        eval_prior=eval_prior,
        eval_subsystems=eval_subsystems,
        eval_robustness=eval_robustness,
        mc_samples=mc_samples,
        intra_samples=intra_samples,
        active_topk=active_topk,
    )


def run_evaluation(cfg: EvalConfig) -> Dict[str, Any]:
    LOGGER.info("Loading schema: %s", cfg.schema)
    schema = load_schema(cfg.schema)

    LOGGER.info("Loading checkpoint: %s", cfg.ckpt)
    state, hparams = _load_ckpt_bundle(cfg.ckpt)
    model = _build_model(state, hparams, schema)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    LOGGER.info("Loading validation data cache")
    batches = load_validation_batches(cfg.data_dir, schema, cfg.batch_size)
    cache = build_data_cache(model, batches, schema, device, cfg.save_dir, cfg.save_raw)
    if cache.cat_offsets is not None:
        setattr(model, "cat_offsets", cache.cat_offsets.to(device))
        setattr(model, "cat_cardinalities", cache.cat_cardinalities.to(device) if cache.cat_cardinalities is not None else None)

    results: Dict[str, Any] = {"meta": {"tier": cfg.tier, "sample_sets": cfg.sample_sets}}

    LOGGER.info("Evaluating posterior health")
    posterior = evaluate_posterior_health(model, cache, cfg)
    results["posterior"] = posterior

    LOGGER.info("Evaluating latent information distribution")
    info_dist = evaluate_latent_information_distribution(model, cache, posterior, cfg, run_probes=cfg.eval_probes)
    results["info_dist"] = info_dist

    LOGGER.info("Evaluating reconstruction quality")
    recon = evaluate_reconstruction(model, cache, cfg, eval_prob=True, eval_sinkhorn=cfg.eval_sinkhorn)
    results["reconstruction"] = recon

    LOGGER.info("Evaluating conditional inference scenarios")
    cond = evaluate_conditional_inference(model, cache, cfg)
    results["conditional"] = cond

    LOGGER.info("Evaluating intra-set consistency")
    intra = evaluate_intra_set_consistency(model, cache, cfg)
    results["intra_consistency"] = intra

    # Core summary metrics for tier A
    results["KL_entropy"] = info_dist.get("entropy")
    results["Gini"] = info_dist.get("gini")
    results["PIT_KS_p"] = intra.get("ks", {}).get("ks_p")

    if cfg.eval_sinkhorn:
        results["OT_total"] = recon.get("sinkhorn", {}).get("total")
        results["plan_peaked"] = recon.get("sinkhorn", {}).get("plan_top10_share")

    if cfg.eval_probes:
        results["Probe_R2_mean"] = info_dist.get("ridge_r2_mean")
        results["Probe_AUC_mean"] = info_dist.get("binary_auc_mean")
        results["Probe_F1_mean"] = info_dist.get("multiclass_f1_mean")

    latent_dim = int(getattr(model, "latent_dim", len(posterior.get("KL_per_dim_mean", [])) or 1))
    active_dims = posterior.get("ActiveDims", {}).get("@0.1")
    collapse_detected = bool(active_dims is not None and latent_dim > 0 and active_dims < 0.3 * latent_dim)

    # Scenario-specific summary for conditional metrics
    for scenario_name, metrics in cond.items():
        binary = metrics.get("binary", {})
        if "nll" in binary:
            results[f"NLL_{scenario_name}"] = binary.get("nll")
        if "ece" in binary:
            results[f"ECE_{scenario_name}"] = binary.get("ece")

    if not collapse_detected and cfg.eval_active:
        LOGGER.info("Evaluating active observation strategies")
        active = evaluate_active_observation(model, cache, cfg, mc_samples=cfg.mc_samples)
        results["active_observation"] = active
        results["ActiveObs_corr"] = active.get("correlation", {}).get("spearman")
    else:
        results["active_observation"] = {}

    if not collapse_detected and cfg.eval_ig:
        LOGGER.info("Evaluating information gain monotonicity")
        ig = evaluate_ig_monotonicity(model, cache, cfg, topk=cfg.topk_ig)
        results["ig"] = ig
        results["IG_tau"] = ig.get("stats", {}).get("avg_kendall_tau")
        results["IG_AUCdecay"] = ig.get("stats", {}).get("avg_auc_decay")
    else:
        results["ig"] = {}

    if not collapse_detected and cfg.eval_prior:
        LOGGER.info("Evaluating prior alignment")
        prior = evaluate_prior_alignment(model, cache, cfg)
        results["prior"] = prior
        results["MMD_qp"] = prior.get("mmd")
    else:
        results["prior"] = {}

    if not collapse_detected and cfg.eval_subsystems:
        LOGGER.info("Analyzing subsystems")
        subsys = analyze_subsystems(model, cache, cfg)
        results["subsystems"] = subsys
        results["Subsystem_ARI"] = subsys.get("consistency", {}).get("ARI")
        results["Subsystem_NMI"] = subsys.get("consistency", {}).get("NMI")
    else:
        results["subsystems"] = {}

    if not collapse_detected and cfg.eval_robustness:
        LOGGER.info("Evaluating robustness modes")
        robust = evaluate_robustness_ablation(model, cache, cfg, cfg.robust_modes)
        results["robustness"] = robust
        results["Robust_delta"] = robust.get("delta_mean")
    else:
        results["robustness"] = {}

    if collapse_detected:
        LOGGER.warning(
            "Active dimensions low (%.1f of %.1f); skipping higher-tier modules.",
            active_dims or float("nan"),
            0.3 * latent_dim,
        )

    summarize_dashboard(results, cfg)
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate SeqSetVAE checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tier", choices=["A", "B", "C", "a", "b", "c"], default="A")
    ap.add_argument("--n_probe_z", type=int, default=1)
    ap.add_argument("--save_dir", default="./eval_outputs")
    ap.add_argument("--plots", default="false")
    ap.add_argument("--mask_scenarios", default="")
    ap.add_argument("--topk_IG", type=int, default=10)
    ap.add_argument("--sample_sets", type=int, default=0)
    ap.add_argument("--save_raw", default="false")
    ap.add_argument("--compare_ckpts", default="")
    ap.add_argument("--robust_modes", default="")
    ap.add_argument("--mc_samples", type=int, default=0)
    ap.add_argument("--intra_samples", type=int, default=0)
    ap.add_argument("--active_topk", type=int, default=10)
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    cfg = build_evaluation_config(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

