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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

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

    # Section 1: posterior health
    metrics["posterior_health"] = evaluate_posterior_health(
        model,
        batches,
        device=cfg.device,
        n_z_samples=cfg.n_z_samples,
        save_dir=cfg.save_dir / "figures",
        seed=cfg.seed,
    )

    # Section 2: information distribution
    metrics["latent_information_distribution"] = evaluate_information_distribution(metrics["posterior_health"])

    # Placeholder evaluations for sections 3-8
    metrics["set_reconstruction"] = {"status": "pending"}
    per_scenario_feature_metrics: Dict[str, Any] = {}
    for scenario in cfg.mask_scenarios:
        per_scenario_feature_metrics[scenario] = {"status": "pending"}
    metrics["feature_inference"] = per_scenario_feature_metrics
    metrics["next_best_measurement"] = {"status": "pending"}
    metrics["subsystem_weights"] = {"status": "pending"}
    metrics["subsystem_sensitivity"] = {"status": "pending"}
    metrics["prior_vs_posterior"] = {"status": "pending"}
    metrics["stress_tests"] = {"status": "pending"}
    metrics["heads_ablation"] = {"status": "pending"}
    metrics["model_selection"] = {"status": "pending"}

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
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    cfg = build_evaluation_config(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

