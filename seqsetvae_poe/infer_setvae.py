#!/usr/bin/env python3
"""
SetVAE inference utilities for:
  1) Conditional completion (infer_full_state)
  2) Active measurement value ranking (rank_measurement_value)
  3) Subsystem (feature grouping) analysis (analyze_subsystems)

Assumptions:
- Trained SetVAE checkpoint (.ckpt) that saved hyperparameters
- Feature schema CSV with columns: feature_id, type, [optional: name, cardinality]
- Using the schema-driven tokenization path from SetVAEOnlyPretrain

Minimal CLI examples:

1) Infer full state from a JSON file of observations (feature or feature_id with value):
   python -m seqsetvae_poe.infer_setvae \
     --ckpt /path/to/setvae_PT.ckpt \
     --schema_csv /path/to/schema.csv \
     --mode infer \
     --input_json partial.json \
     --output_json out_infer.json

   partial.json example:
   {
     "observations": [
       {"feature": "heart_rate", "value": 110},
       {"feature": "icd_implanted", "value": 1},
       {"feature_id": 42, "value": 3}
     ]
   }

2) Rank next measurements:
   python -m seqsetvae_poe.infer_setvae \
     --ckpt /path/to/setvae_PT.ckpt \
     --schema_csv /path/to/schema.csv \
     --mode rank \
     --input_json partial.json \
     --top_k 20 \
     --output_json out_rank.json

3) Analyze subsystems from saved z samples (.npy of shape [M, D]):
   python -m seqsetvae_poe.infer_setvae \
     --ckpt /path/to/setvae_PT.ckpt \
     --schema_csv /path/to/schema.csv \
     --mode analyze \
     --z_samples_npy zs.npy \
     --num_clusters 12 \
     --output_json out_clusters.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from seqsetvae_poe.model import SetVAEOnlyPretrain  # type: ignore
except Exception:
    from model import SetVAEOnlyPretrain  # type: ignore


@dataclass
class FeatureInfo:
    feature_id: int
    name: str
    type_code: int  # 0=cont, 1=bin, 2=cat
    cardinality: int = 0


class Schema:
    def __init__(self, schema_csv: str):
        import pandas as pd
        if not os.path.isfile(schema_csv):
            raise FileNotFoundError(f"Schema CSV not found: {schema_csv}")
        df = pd.read_csv(schema_csv)
        if "feature_id" not in df.columns or "type" not in df.columns:
            raise ValueError("schema.csv must contain columns: feature_id,type [optional: cardinality,name]")
        # normalize
        def _type_to_code(x: Any) -> int:
            if isinstance(x, str):
                s = x.strip().lower()
                if s in {"cont", "continuous", "real"}: return 0
                if s in {"bin", "binary", "bern", "bernoulli"}: return 1
                if s in {"cat", "categorical", "multi"}: return 2
                if s.isdigit(): return int(s)
                raise ValueError(f"Unknown type string: {x}")
            try:
                iv = int(x)
                if iv in (0, 1, 2):
                    return iv
            except Exception:
                pass
            raise ValueError(f"Unrecognized type value: {x}")

        df = df.copy()
        df["feature_id"] = df["feature_id"].astype(int)
        if "name" not in df.columns:
            df["name"] = df["feature_id"].astype(str)
        if "cardinality" not in df.columns:
            df["cardinality"] = 0
        df["type_code"] = df["type"].apply(_type_to_code).astype(int)

        self._id_to_info: Dict[int, FeatureInfo] = {}
        self._name_to_id: Dict[str, int] = {}
        for _, row in df.iterrows():
            fid = int(row["feature_id"])  # noqa: N806
            name = str(row.get("name", fid))
            tcode = int(row["type_code"])  # noqa: N806
            card = int(row.get("cardinality", 0) or 0)
            self._id_to_info[fid] = FeatureInfo(feature_id=fid, name=name, type_code=tcode, cardinality=card)
            # prefer lower-case names for lookups
            self._name_to_id[name.strip().lower()] = fid

    @property
    def id_to_info(self) -> Dict[int, FeatureInfo]:
        return self._id_to_info

    @property
    def name_to_id(self) -> Dict[str, int]:
        return self._name_to_id

    def get_feature_id(self, name_or_id: Any) -> Optional[int]:
        try:
            if isinstance(name_or_id, str):
                key = name_or_id.strip().lower()
                return self._name_to_id.get(key, None)
            iv = int(name_or_id)
            return iv
        except Exception:
            return None


class SetVAEInference:
    def __init__(self, ckpt_path: str, schema_csv: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.schema = Schema(schema_csv)
        # Best-effort load with Lightning's load_from_checkpoint to restore exact hparams
        self.model = self._load_model(ckpt_path).to(self.device).eval()
        if not getattr(self.model, "enable_prob_head", False):
            raise RuntimeError("Loaded SetVAE model does not have probabilistic head enabled (schema-driven).")

    def _load_model(self, ckpt_path: str) -> SetVAEOnlyPretrain:
        # Prefer Lightning load, which reconstructs with saved hyperparameters
        try:
            model = SetVAEOnlyPretrain.load_from_checkpoint(ckpt_path, map_location="cpu")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint with Lightning: {e}")

    # -------------------- Tokenization --------------------
    def _build_tokens_from_partial(self, observations: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build token inputs [1,N,R] using the model's schema-driven embedding stack from a list of observations.
        Each observation item should contain either {"feature": name, "value": v} or {"feature_id": id, "value": v}.
        Returns:
          x_tokens: [1, N, reduced_dim]
          feat_ids: [1, N, 1] (long)
          values: [1, N, 1] (float)
        """
        model = self.model
        assert model.num_features > 0, "Model was not initialized with schema (num_features=0)."
        if len(observations) == 0:
            raise ValueError("Empty observations.")

        feat_ids: List[int] = []
        values: List[float] = []
        for obs in observations:
            if "feature_id" in obs:
                fid = int(obs["feature_id"])  # noqa: N806
            elif "feature" in obs:
                fid = self.schema.get_feature_id(obs["feature"])  # type: ignore
                if fid is None:
                    raise ValueError(f"Unknown feature name: {obs['feature']}")
            else:
                raise ValueError("Each observation must include 'feature' or 'feature_id'.")
            if fid not in self.schema.id_to_info:
                raise ValueError(f"feature_id {fid} missing from schema")
            val = float(obs.get("value", 0.0))
            feat_ids.append(fid)
            values.append(val)

        device = self.device
        fid_t = torch.tensor(feat_ids, dtype=torch.long, device=device).view(1, -1, 1)
        val_t = torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1)

        # Build state embeddings by type
        # id embedding per token
        id_emb = model.id_embedding(fid_t.squeeze(-1))  # [1,N,Eid]

        # Prepare state_emb with the same shape as cont_value_embed out_dim
        state_dim: int = int(model.cont_value_embed[0].out_features)  # type: ignore[index]
        state_emb = torch.zeros(id_emb.shape[0], id_emb.shape[1], state_dim, device=device, dtype=id_emb.dtype)

        # Types per token
        tcode = torch.zeros_like(fid_t)
        if model.feature_types is not None:
            tcode = model.feature_types.to(device)[fid_t.squeeze(-1)].view(1, -1, 1)

        # Continuous
        cont_mask = (tcode == 0).squeeze(0).squeeze(-1)
        if cont_mask.any():
            cont_vals = val_t.squeeze(0).squeeze(-1)[cont_mask].view(-1, 1)
            cont_emb = model.cont_value_embed(cont_vals)
            state_emb[0, cont_mask, :] = cont_emb

        # Binary
        bin_mask = (tcode == 1).squeeze(0).squeeze(-1)
        if bin_mask.any():
            bin_vals = (val_t.squeeze(0).squeeze(-1)[bin_mask] > 0.5).long()
            bin_fid = fid_t.squeeze(0).squeeze(-1)[bin_mask]
            flat_idx = bin_fid * 2 + bin_vals
            bin_emb = model.bin_state_embed(flat_idx)
            state_emb[0, bin_mask, :] = bin_emb

        # Categorical
        cat_mask = (tcode == 2).squeeze(0).squeeze(-1)
        if cat_mask.any() and getattr(model, "cat_state_embed", None) is not None:
            cat_vals = val_t.squeeze(0).squeeze(-1)[cat_mask].long().clamp(min=0)
            cat_fid = fid_t.squeeze(0).squeeze(-1)[cat_mask]
            offs = model.cat_offsets.to(device)[cat_fid]
            max_class = (model.cat_cardinalities.to(device)[cat_fid] - 1).clamp(min=0)
            cat_vals = torch.minimum(cat_vals, max_class)
            flat_idx = offs + cat_vals
            cat_emb = model.cat_state_embed(flat_idx)
            state_emb[0, cat_mask, :] = cat_emb

        tok = torch.cat([id_emb, state_emb], dim=-1)
        x_tokens = model.token_mlp(tok)
        return x_tokens, fid_t, val_t

    # -------------------- Core inference --------------------
    @torch.inference_mode()
    def _encode_tokens_to_z(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Encode tokens with the SetVAE encoder and return posterior mean of last layer z.
        Returns:
          z_mu: [1, 1, latent_dim]
          z_list: list of tuples (z_sample, mu, logvar)
        """
        z_list, _ = self.model.set_encoder.encode(x_tokens)
        _, mu, _ = z_list[-1]
        return mu, z_list

    @torch.inference_mode()
    def _predict_distributions(self, z_mu: torch.Tensor) -> Dict[str, Any]:
        """
        Use the probabilistic head to predict distributions for all features.
        Returns a dictionary with keys: cont, bin, cat.
        - cont: {feature_id: {"mu": float, "var": float}}
        - bin:  {feature_id: {"p": float}}
        - cat:  {feature_id: {"probs": [float]}}
        """
        model = self.model
        assert getattr(model, "prob_shared", None) is not None, "Model missing probabilistic head"
        h = model.prob_shared(z_mu.squeeze(1))  # [1,H]

        out: Dict[str, Any] = {"cont": {}, "bin": {}, "cat": {}}
        F_total = model.num_features

        # Continuous
        cont_mu = model.prob_cont_mu(h)  # [1,F]
        cont_logvar = model.prob_cont_logvar(h).clamp(min=-8.0, max=8.0)
        cont_var = cont_logvar.exp()
        for fid in range(F_total):
            info = self.schema.id_to_info.get(fid, None)
            if info is None or info.type_code != 0:
                continue
            mu = float(cont_mu[0, fid].detach().cpu().item())
            var = float(cont_var[0, fid].detach().cpu().item())
            out["cont"][fid] = {"mu": mu, "var": var}

        # Binary
        bin_logit = model.prob_bin_logit(h)  # [1,F]
        bin_p = torch.sigmoid(bin_logit)
        for fid in range(F_total):
            info = self.schema.id_to_info.get(fid, None)
            if info is None or info.type_code != 1:
                continue
            p = float(bin_p[0, fid].detach().cpu().item())
            out["bin"][fid] = {"p": p}

        # Categorical
        if getattr(model, "prob_cat_logits", None) is not None and model.total_cat_states > 0:
            cat_logits = model.prob_cat_logits(h)  # [1,total_states]
            logits = cat_logits[0]
            for fid, info in self.schema.id_to_info.items():
                if info.type_code != 2 or info.cardinality <= 0:
                    continue
                off = int(model.cat_offsets[fid].item())
                card = int(model.cat_cardinalities[fid].item())
                if off < 0 or card <= 0:
                    continue
                sl = slice(off, off + card)
                probs = F.softmax(logits[sl], dim=-1)
                out["cat"][fid] = {"probs": probs.detach().cpu().tolist()}
        return out

    @staticmethod
    def _uncertainty_metrics(schema: Schema, dist_pred: Dict[str, Any]) -> Dict[int, float]:
        """
        Compute per-feature uncertainty: variance (cont), p*(1-p) (bin), entropy (cat).
        Returns a mapping: feature_id -> uncertainty score.
        """
        scores: Dict[int, float] = {}
        # cont
        for fid, entry in dist_pred.get("cont", {}).items():
            scores[int(fid)] = float(max(0.0, entry.get("var", 0.0)))
        # bin
        for fid, entry in dist_pred.get("bin", {}).items():
            p = float(entry.get("p", 0.0))
            scores[int(fid)] = float(max(0.0, p * (1.0 - p)))
        # cat
        for fid, entry in dist_pred.get("cat", {}).items():
            probs = np.asarray(entry.get("probs", []), dtype=np.float64)
            if probs.size == 0:
                continue
            eps = 1e-12
            ent = float(-np.sum(probs * np.log(probs + eps)))
            scores[int(fid)] = ent
        return scores

    @torch.inference_mode()
    def infer_full_state(self, observations: List[Dict[str, Any]], target_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Run conditional completion from partial observations.
        Returns a structured dictionary with predictions and optional reconstruction tokens.
        """
        x_tokens, fid_t, val_t = self._build_tokens_from_partial(observations)
        x_tokens = x_tokens.to(self.device)
        z_mu, z_list = self._encode_tokens_to_z(x_tokens)
        dist_pred = self._predict_distributions(z_mu)
        unc = self._uncertainty_metrics(self.schema, dist_pred)

        # Optional: reconstruct a self-consistent token set with the decoder
        recon_tokens: Optional[List[List[float]]] = None
        if target_n is None:
            target_n = int(x_tokens.shape[1])
        try:
            # Decode using mean latent; fabricate minimal z_list containing only last layer mean
            zlast = z_mu  # [1,1,D]
            dec_in: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [(zlast, zlast, torch.zeros_like(zlast))]
            recon = self.model.set_encoder.decode(dec_in, target_n=target_n, use_mean=True, noise_std=0.0)
            recon_tokens = recon[0].detach().cpu().tolist()
        except Exception:
            recon_tokens = None

        # Build human-friendly table
        feature_rows: List[Dict[str, Any]] = []
        for fid, info in self.schema.id_to_info.items():
            row: Dict[str, Any] = {
                "feature_id": fid,
                "name": info.name,
                "type": info.type_code,
            }
            if info.type_code == 0 and fid in dist_pred["cont"]:
                mu = dist_pred["cont"][fid]["mu"]
                var = dist_pred["cont"][fid]["var"]
                row.update({"estimate": mu, "uncertainty": var})
            elif info.type_code == 1 and fid in dist_pred["bin"]:
                p = dist_pred["bin"][fid]["p"]
                row.update({"estimate": float(p >= 0.5), "p": p, "uncertainty": p * (1.0 - p)})
            elif info.type_code == 2 and fid in dist_pred["cat"]:
                probs = dist_pred["cat"][fid]["probs"]
                top_idx = int(np.argmax(probs)) if len(probs) > 0 else -1
                ent = 0.0
                if len(probs) > 0:
                    arr = np.asarray(probs, dtype=np.float64)
                    ent = float(-np.sum(arr * np.log(arr + 1e-12)))
                row.update({"estimate": top_idx, "probs": probs, "uncertainty": ent})
            else:
                row.update({"estimate": None, "uncertainty": None})
            feature_rows.append(row)

        return {
            "z": z_mu[0, 0].detach().cpu().tolist(),
            "predictions": feature_rows,
            "recon_tokens": recon_tokens,  # vectors in reduced_dim space
        }

    @torch.inference_mode()
    def rank_measurement_value(self, observations: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Rank unobserved features by their measurement value score (uncertainty proxy).
        Returns a list of dicts sorted descending by score.
        """
        x_tokens, fid_t, _ = self._build_tokens_from_partial(observations)
        z_mu, _ = self._encode_tokens_to_z(x_tokens.to(self.device))
        dist_pred = self._predict_distributions(z_mu)
        uncertainty = self._uncertainty_metrics(self.schema, dist_pred)

        observed_ids = set(int(fid) for fid in fid_t.view(-1).tolist())
        rows: List[Dict[str, Any]] = []
        for fid, info in self.schema.id_to_info.items():
            if fid in observed_ids:
                continue
            score = float(uncertainty.get(fid, 0.0))
            row: Dict[str, Any] = {"feature_id": fid, "name": info.name, "score": score}
            # attach current prediction (point estimate) and its uncertainty
            if info.type_code == 0 and fid in dist_pred["cont"]:
                mu = dist_pred["cont"][fid]["mu"]
                var = dist_pred["cont"][fid]["var"]
                row.update({"pred": mu, "uncertainty": var})
            elif info.type_code == 1 and fid in dist_pred["bin"]:
                p = dist_pred["bin"][fid]["p"]
                row.update({"pred": float(p >= 0.5), "p": p, "uncertainty": p * (1.0 - p)})
            elif info.type_code == 2 and fid in dist_pred["cat"]:
                probs = dist_pred["cat"][fid]["probs"]
                top_idx = int(np.argmax(probs)) if len(probs) > 0 else -1
                ent = 0.0
                if len(probs) > 0:
                    arr = np.asarray(probs, dtype=np.float64)
                    ent = float(-np.sum(arr * np.log(arr + 1e-12)))
                row.update({"pred": top_idx, "probs": probs, "uncertainty": ent})
            rows.append(row)

        rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        if top_k > 0:
            rows = rows[:top_k]
        return rows

    def analyze_subsystems(
        self,
        z_samples: torch.Tensor,
        num_clusters: int = 12,
        method: str = "weights",
    ) -> Dict[str, Any]:
        """
        Analyze feature groupings (subsystems) using one of two methods:
        - method="weights": cluster by final head weights in prob_head_hidden space (cheap, default)
        - method="grad": average gradients of head outputs w.r.t latent z over z_samples (costly)
        Returns a dict with clusters and per-cluster members.
        """
        model = self.model.eval()
        device = self.device
        if not isinstance(z_samples, torch.Tensor):
            z_samples = torch.tensor(z_samples, dtype=torch.float32)
        if z_samples.dim() == 1:
            z_samples = z_samples.view(1, -1)
        z_samples = z_samples.to(device)

        # Build a feature vector per feature in a common space
        feat_vecs: Dict[int, np.ndarray] = {}
        hidden_dim = int(model.prob_cont_mu.in_features)  # head input size

        if method == "weights":
            # Use final linear layer weights in prob_head_hidden space
            # Continuous features: use mu head weights
            W_cont = model.prob_cont_mu.weight.detach().cpu().numpy()  # [F, H]
            # Binary features: logits head weights
            W_bin = model.prob_bin_logit.weight.detach().cpu().numpy()  # [F, H]
            # Categorical: aggregate slice-wise weights per feature by averaging
            W_cat_full = None
            if getattr(model, "prob_cat_logits", None) is not None and model.total_cat_states > 0:
                W_cat_full = model.prob_cat_logits.weight.detach().cpu().numpy()  # [total_states, H]

            for fid, info in self.schema.id_to_info.items():
                vec = None
                if info.type_code == 0:
                    vec = W_cont[fid]
                elif info.type_code == 1:
                    vec = W_bin[fid]
                elif info.type_code == 2 and W_cat_full is not None and info.cardinality > 0:
                    off = int(model.cat_offsets[fid].item())
                    card = int(model.cat_cardinalities[fid].item())
                    if off >= 0 and card > 0:
                        vec = W_cat_full[off : off + card].mean(axis=0)
                if vec is None:
                    vec = np.zeros((hidden_dim,), dtype=np.float32)
                # normalize for cosine-based clustering
                norm = float(np.linalg.norm(vec) + 1e-12)
                feat_vecs[fid] = (vec / norm).astype(np.float32)
        else:
            # Gradient-based: average |grad output| wrt z over samples
            # We'll compute gradient of scalar summary per feature:
            #  - cont: mu_f
            #  - bin: logit_f
            #  - cat: L2 norm of logits slice for feature f
            for fid, info in self.schema.id_to_info.items():
                grads = []
                for m in range(z_samples.shape[0]):
                    z = z_samples[m : m + 1].detach().clone().requires_grad_(True)  # [1,D]
                    h = model.prob_shared(z)
                    if info.type_code == 0:
                        scalar = model.prob_cont_mu(h)[0, fid]
                    elif info.type_code == 1:
                        scalar = model.prob_bin_logit(h)[0, fid]
                    else:
                        scalar = torch.tensor(0.0, device=device)
                        if getattr(model, "prob_cat_logits", None) is not None and info.cardinality > 0:
                            off = int(model.cat_offsets[fid].item())
                            card = int(model.cat_cardinalities[fid].item())
                            if off >= 0 and card > 0:
                                sl = slice(off, off + card)
                                logits = model.prob_cat_logits(h)[0, sl]
                                scalar = torch.linalg.vector_norm(logits, ord=2)
                    g = torch.autograd.grad(scalar, z, retain_graph=False, create_graph=False)[0]
                    grads.append(g.detach().cpu().numpy().reshape(-1))
                vec = np.mean(np.stack(grads, axis=0), axis=0)
                norm = float(np.linalg.norm(vec) + 1e-12)
                feat_vecs[fid] = (vec / norm).astype(np.float32)

        # Simple KMeans (if sklearn available), else greedy cosine thresholding
        feature_ids = sorted(self.schema.id_to_info.keys())
        X = np.stack([feat_vecs[fid] for fid in feature_ids], axis=0)

        clusters: List[List[int]] = []
        try:
            from sklearn.cluster import KMeans  # type: ignore
            kmeans = KMeans(n_clusters=max(2, int(num_clusters)), n_init=10, random_state=0)
            labels = kmeans.fit_predict(X)
            clusters_map: Dict[int, List[int]] = {}
            for fid, lab in zip(feature_ids, labels.tolist()):
                clusters_map.setdefault(int(lab), []).append(int(fid))
            clusters = [sorted(v) for _, v in sorted(clusters_map.items(), key=lambda kv: kv[0])]
        except Exception:
            # fallback: greedy grouping by cosine similarity threshold
            thr = 0.8
            taken = set()
            for i, fid in enumerate(feature_ids):
                if fid in taken:
                    continue
                center = X[i]
                group = [fid]
                taken.add(fid)
                for j, fj in enumerate(feature_ids):
                    if fj in taken:
                        continue
                    sim = float(np.dot(center, X[j]) / (np.linalg.norm(center) * np.linalg.norm(X[j]) + 1e-12))
                    if sim >= thr:
                        group.append(fj)
                        taken.add(fj)
                clusters.append(sorted(group))

        # Prepare output
        out_clusters: List[Dict[str, Any]] = []
        for cid, members in enumerate(clusters):
            out_clusters.append({
                "subsystem_id": cid,
                "features": [
                    {
                        "feature_id": fid,
                        "name": self.schema.id_to_info[fid].name,
                        "type": self.schema.id_to_info[fid].type_code,
                    }
                    for fid in members
                ],
            })
        return {"clusters": out_clusters}


# -------------------- CLI --------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="SetVAE inference utilities")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--schema_csv", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--mode", type=str, choices=["infer", "rank", "analyze"], required=True)
    ap.add_argument("--input_json", type=str, default=None, help="partial observations JSON for infer/rank")
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--z_samples_npy", type=str, default=None, help=".npy file of shape [M,D] for analyze")
    ap.add_argument("--num_clusters", type=int, default=12)
    args = ap.parse_args()

    infer = SetVAEInference(args.ckpt, args.schema_csv, device=args.device)

    if args.mode in {"infer", "rank"}:
        if not args.input_json:
            raise ValueError("--input_json is required for infer/rank modes")
        data = _load_json(args.input_json)
        observations = data.get("observations", [])
        if not isinstance(observations, list):
            raise ValueError("input_json must contain a list under 'observations'")
        if args.mode == "infer":
            result = infer.infer_full_state(observations)
        else:
            result = {"ranked": infer.rank_measurement_value(observations, top_k=args.top_k)}
        if args.output_json:
            _save_json(args.output_json, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "analyze":
        if not args.z_samples_npy:
            raise ValueError("--z_samples_npy is required for analyze mode")
        zs = np.load(args.z_samples_npy)
        result = infer.analyze_subsystems(torch.tensor(zs, dtype=torch.float32), num_clusters=args.num_clusters)
        if args.output_json:
            _save_json(args.output_json, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
