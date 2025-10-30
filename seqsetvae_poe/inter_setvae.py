#!/usr/bin/env python3
"""
Set-level analysis utilities for SetVAE (inter-set VAE) models.

Functions:
1) information_gain_monotonicity: Rank variables by their contribution to overall
   predictive uncertainty; progressively mask in that order and check whether
   similarity to the original set monotonically decreases.
2) intraset_self_consistency: Leave-one-out per variable; condition on the rest
   to obtain the masked variable's marginal predictive distribution and compute
   the p-value of the true observation. If none are significant, the set is
   self-consistent.

Note:
- This module reuses the schema-driven tokenization and predictive distribution
  utilities from seqsetvae_poe.infer_setvae.SetVAEInference.
- Similarity is computed in latent space by default (cosine between z means).

中文摘要：
- 信息增益：按“去掉该变量导致总体不确定性上升的幅度”排序；按此顺序依次遮盖，检查与原集的相似性（默认用潜在表征余弦相似度）是否单调递减。
- 集内自洽：循环遮盖单个变量，基于其余变量得到该变量的边际分布，计算真实观测值的 p-value；若全部不显著（p≥α），则认为自洽。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    # Prefer explicit import when used as a package
    from seqsetvae_poe.infer_setvae import SetVAEInference, Schema  # type: ignore
except Exception:  # pragma: no cover - fallback for local runs
    from infer_setvae import SetVAEInference, Schema  # type: ignore


__all__ = [
    "information_gain_monotonicity",
    "intraset_self_consistency",
]


def _aggregate_uncertainty(
    schema: Schema,
    dist_pred: Dict[str, Any],
    agg: str = "mean",
) -> float:
    """Aggregate predictive uncertainty across all features.

    - Continuous: variance
    - Binary: p*(1-p)
    - Categorical: entropy
    """
    # Build per-feature uncertainty using the same rules as SetVAEInference
    scores: List[float] = []
    # cont
    for fid, entry in dist_pred.get("cont", {}).items():
        scores.append(float(max(0.0, entry.get("var", 0.0))))
    # bin
    for fid, entry in dist_pred.get("bin", {}).items():
        p = float(entry.get("p", 0.0))
        scores.append(float(max(0.0, p * (1.0 - p))))
    # cat
    for fid, entry in dist_pred.get("cat", {}).items():
        probs = np.asarray(entry.get("probs", []), dtype=np.float64)
        if probs.size == 0:
            continue
        eps = 1e-12
        ent = float(-np.sum(probs * np.log(probs + eps)))
        scores.append(ent)

    if not scores:
        return 0.0
    if agg == "sum":
        return float(np.sum(scores))
    # default: mean
    return float(np.mean(scores))


def _latent_similarity(z_ref: torch.Tensor, z_cur: torch.Tensor, kind: str = "cosine") -> float:
    """Compute similarity between two latent means z in shape [1,1,D] or [1,D]."""
    # Normalize shapes
    if z_ref.dim() == 3:
        z_ref = z_ref.squeeze(1)
    if z_cur.dim() == 3:
        z_cur = z_cur.squeeze(1)
    if kind == "euclidean":
        # Convert distance to similarity in (0, 1]; larger is more similar
        dist = torch.linalg.vector_norm(z_ref - z_cur, ord=2).item()
        return float(1.0 / (1.0 + dist))
    # default: cosine
    sim = F.cosine_similarity(z_ref, z_cur, dim=-1).mean().item()
    return float(sim)


def information_gain_monotonicity(
    infer: SetVAEInference,
    observations: List[Dict[str, Any]],
    *,
    agg: str = "mean",
    similarity: str = "cosine",
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Information-gain based progressive masking test.

    Steps:
    1) Rank variables by contribution to overall predictive uncertainty:
       contribution_i = AggregateUncertainty(without i) - AggregateUncertainty(full)
    2) Progressively remove variables in that order and measure similarity of
       latent means to the original set; report whether it is monotonically
       non-increasing.

    Args:
      infer: Initialized SetVAEInference
      observations: list of {feature or feature_id, value}
      agg: aggregator for overall uncertainty, one of {"mean", "sum"}
      similarity: similarity metric in latent space, {"cosine", "euclidean"}
      max_steps: optional maximum number of removals (defaults to all - 1)

    Returns: dict with keys
      - ranking: list of indices (into observations) sorted by contribution desc
      - ranked_feature_ids: list of feature_ids in the same order
      - contribution: list of contribution scores aligned with ranking
      - similarities: similarity sequence after 0..k removals
      - monotonic_nonincreasing: bool
    """
    device = infer.device

    # Baseline latent and uncertainty
    x_full, _, _ = infer._build_tokens_from_partial(observations)
    z_full, _ = infer._encode_tokens_to_z(x_full.to(device))
    dist_full = infer._predict_distributions(z_full)
    base_unc = _aggregate_uncertainty(infer.schema, dist_full, agg=agg)

    # Compute contribution per variable by leave-one-out increase in uncertainty
    contrib: List[float] = []
    for i in range(len(observations)):
        obs_wo = [observations[j] for j in range(len(observations)) if j != i]
        x_i, _, _ = infer._build_tokens_from_partial(obs_wo)
        z_i, _ = infer._encode_tokens_to_z(x_i.to(device))
        dist_i = infer._predict_distributions(z_i)
        unc_i = _aggregate_uncertainty(infer.schema, dist_i, agg=agg)
        contrib.append(float(unc_i - base_unc))

    # Build ranking (descending contribution)
    ranking = sorted(range(len(contrib)), key=lambda i: contrib[i], reverse=True)
    ranked_feature_ids: List[int] = []
    for idx in ranking:
        obs = observations[idx]
        fid = obs.get("feature_id")
        if fid is None and "feature" in obs:
            fid = infer.schema.get_feature_id(obs["feature"])  # type: ignore
        ranked_feature_ids.append(int(fid))  # type: ignore[arg-type]

    # Progressive masking in ranked order
    steps = len(observations) - 1  # do not remove everything
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    sims: List[float] = []

    # s0: no removal
    sims.append(_latent_similarity(z_full, z_full, kind=similarity))

    current_obs = list(observations)
    for k in range(1, steps + 1):
        # Remove the k-th ranked variable
        to_remove = ranking[k - 1]
        # Remove by index in the current list: map original index to current position
        # Compute current index
        cur_idx = None
        for j, obs in enumerate(current_obs):
            fid = obs.get("feature_id")
            if fid is None and "feature" in obs:
                fid = infer.schema.get_feature_id(obs["feature"])  # type: ignore
            if int(fid) == ranked_feature_ids[k - 1]:  # type: ignore[arg-type]
                cur_idx = j
                break
        if cur_idx is None:
            # fallback: skip if not found (should not happen)
            continue
        del current_obs[cur_idx]
        # Re-encode and measure similarity
        x_k, _, _ = infer._build_tokens_from_partial(current_obs)
        z_k, _ = infer._encode_tokens_to_z(x_k.to(device))
        sims.append(_latent_similarity(z_full, z_k, kind=similarity))

    # Check monotonic non-increasing
    mono = True
    for i in range(1, len(sims)):
        if sims[i] > sims[i - 1] + 1e-8:
            mono = False
            break

    return {
        "ranking": ranking,
        "ranked_feature_ids": ranked_feature_ids,
        "contribution": [float(c) for c in contrib],
        "baseline_uncertainty": float(base_unc),
        "similarities": sims,
        "monotonic_nonincreasing": bool(mono),
        "similarity_metric": similarity,
        "uncertainty_agg": agg,
    }


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_normal_pvalue(x: float, mu: float, var: float) -> float:
    if var <= 0.0:
        # Degenerate: if x==mu then p=1, otherwise ~0
        return 1.0 if abs(x - mu) < 1e-8 else 0.0
    z = (x - mu) / max(1e-12, math.sqrt(var))
    cdf = _std_normal_cdf(z)
    return float(2.0 * min(cdf, 1.0 - cdf))


def _bernoulli_pvalue(p: float, v: int) -> float:
    # Two-sided style: probability of values with probability <= P(X=v)
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    pv = p if v == 1 else (1.0 - p)
    return float(min(pv, 1.0 - pv))


def _categorical_pvalue(probs: List[float], v: int) -> float:
    if not probs:
        return 1.0
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    p = p / np.sum(p)
    v = int(max(0, min(v, len(p) - 1)))
    pv = float(p[v])
    return float(np.sum(p[p <= pv]))


def intraset_self_consistency(
    infer: SetVAEInference,
    observations: List[Dict[str, Any]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Intra-set self-consistency via leave-one-out p-values.

    For each variable i:
      - Mask i, condition on others, predict marginal distribution of i
      - Compute p-value of the true observed value under that distribution

    Args:
      infer: Initialized SetVAEInference
      observations: list of {feature or feature_id, value}
      alpha: significance threshold

    Returns a dict with:
      - results: list of {feature_id, name, type, p_value, significant}
      - self_consistent: bool (True if all p>=alpha)
      - alpha: the threshold used
    """
    device = infer.device

    results: List[Dict[str, Any]] = []
    for i, ob in enumerate(observations):
        # Resolve feature id and observed value
        fid = ob.get("feature_id")
        if fid is None and "feature" in ob:
            fid = infer.schema.get_feature_id(ob["feature"])  # type: ignore
        if fid is None:
            raise ValueError(f"Observation {i} missing feature id")
        fid = int(fid)
        info = infer.schema.id_to_info.get(fid, None)
        if info is None:
            raise ValueError(f"feature_id {fid} missing from schema")
        val = float(ob.get("value", 0.0))

        # Mask i and predict
        obs_wo = [observations[j] for j in range(len(observations)) if j != i]
        x_i, _, _ = infer._build_tokens_from_partial(obs_wo)
        z_i, _ = infer._encode_tokens_to_z(x_i.to(device))
        dist_i = infer._predict_distributions(z_i)

        # Compute p-value by type
        pval: float = 1.0
        if info.type_code == 0:  # continuous
            entry = dist_i.get("cont", {}).get(fid, None)
            if entry is not None:
                mu = float(entry.get("mu", 0.0))
                var = float(entry.get("var", 1.0))
                pval = _two_sided_normal_pvalue(val, mu, var)
        elif info.type_code == 1:  # binary
            entry = dist_i.get("bin", {}).get(fid, None)
            if entry is not None:
                p = float(entry.get("p", 0.5))
                v = 1 if val > 0.5 else 0
                pval = _bernoulli_pvalue(p, v)
        elif info.type_code == 2:  # categorical
            entry = dist_i.get("cat", {}).get(fid, None)
            if entry is not None:
                probs = entry.get("probs", [])
                # ensure int class index within [0, card-1]
                v = int(round(val))
                if info.cardinality > 0:
                    v = max(0, min(v, int(info.cardinality) - 1))
                pval = _categorical_pvalue(probs, v)
        else:
            pval = 1.0

        results.append({
            "feature_id": fid,
            "name": info.name,
            "type": int(info.type_code),
            "p_value": float(pval),
            "significant": bool(pval < alpha),
        })

    self_consistent = all(not r["significant"] for r in results)

    return {
        "results": results,
        "self_consistent": bool(self_consistent),
        "alpha": float(alpha),
    }
