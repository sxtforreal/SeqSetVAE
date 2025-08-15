#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from comprehensive_visualizer import Config, load_model
from dataset import SeqSetVAEDataModule


def _to_numpy(t: torch.Tensor) -> np.ndarray:
	return t.detach().cpu().numpy()


def _stack_events(tensors: List[torch.Tensor]) -> np.ndarray:
	"""Concatenate a list of [B, N, D] tensors (B assumed 1) along N -> [N_total, D]."""
	arrs = []
	for tsr in tensors:
		if tsr is None:
			continue
		if tsr.ndim != 3:
			raise ValueError(f"Expected 3D tensor [B,N,D], got shape {tuple(tsr.shape)}")
		arrs.append(_to_numpy(tsr[0]))
	if not arrs:
		return np.zeros((0, 0), dtype=np.float32)
	return np.concatenate(arrs, axis=0)


def _nn_from_a_to_b(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (dists, indices) from each point in a to nearest in b using L2."""
	if len(b) == 0:
		return np.array([]), np.array([])
	nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
	dists, idxs = nn.kneighbors(a)
	return dists.squeeze(1), idxs.squeeze(1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
	if len(x) < 2 or len(y) < 2:
		return float("nan")
	if np.std(x) < 1e-8 or np.std(y) < 1e-8:
		return float("nan")
	return float(np.corrcoef(x, y)[0, 1])


def compute_recon_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict:
	"""Compute robust metrics comparing recon to orig in original feature space.
	Both inputs are [N, D].
	"""
	metrics: Dict[str, float] = {}
	if orig.size == 0 or recon.size == 0:
		return {"error": "empty inputs"}

	# Nearest neighbor from recon->orig and orig->recon
	d_ro, idx_ro = _nn_from_a_to_b(recon, orig)
	d_or, idx_or = _nn_from_a_to_b(orig, recon)

	# Chamfer (symmetric mean of squared L2)
	chamfer = float(np.mean(d_ro ** 2) + np.mean(d_or ** 2)) / 2.0
	metrics["chamfer_l2_mean"] = chamfer

	# L2 distances stats (recon->orig)
	metrics.update({
		"nn_l2_mean": float(np.mean(d_ro)),
		"nn_l2_median": float(np.median(d_ro)),
		"nn_l2_p95": float(np.percentile(d_ro, 95)),
	})

	# Directional cosine similarity using matched pairs (recon->orig)
	def _norm(x):
		return np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
	u_recon = recon / _norm(recon)
	u_orig = orig / _norm(orig)
	paired_orig = u_orig[idx_ro]
	cos_sim = np.sum(u_recon * paired_orig, axis=1)
	metrics["dir_cosine_mean"] = float(np.mean(cos_sim))
	metrics["dir_cosine_median"] = float(np.median(cos_sim))

	# Magnitude agreement for matched pairs
	norm_recon = np.linalg.norm(recon, axis=1)
	norm_orig = np.linalg.norm(orig, axis=1)
	norm_orig_matched = norm_orig[idx_ro]
	metrics["mag_mae"] = float(np.mean(np.abs(norm_recon - norm_orig_matched)))
	metrics["mag_rmse"] = float(np.sqrt(np.mean((norm_recon - norm_orig_matched) ** 2)))
	metrics["mag_corr"] = _safe_corr(norm_recon, norm_orig_matched)
	metrics["scale_ratio"] = float((np.mean(norm_recon) + 1e-8) / (np.mean(norm_orig) + 1e-8))

	# Coverage at absolute L2 thresholds
	for th in [0.25, 0.5, 1.0, 2.0]:
		metrics[f"coverage@{th}"] = float(np.mean(d_ro <= th))

	# Global distributional checks
	metrics["orig_norm_mean"] = float(np.mean(norm_orig))
	metrics["recon_norm_mean"] = float(np.mean(norm_recon))
	metrics["orig_norm_std"] = float(np.std(norm_orig))
	metrics["recon_norm_std"] = float(np.std(norm_recon))

	return metrics


def evaluate_single_sample(model, batch: Dict) -> Dict:
	"""Run a forward pass to populate model._last_* and compute metrics.
	Returns a dict with global and per-set metrics.
	"""
	device = next(model.parameters()).device
	for k, v in list(batch.items()):
		if torch.is_tensor(v):
			batch[k] = v.to(device)

	with torch.no_grad():
		_ = model(batch)

	# Per-set lists if available
	recon_list: List[torch.Tensor] = getattr(model, "_last_recon_list", None)
	orig_list: List[torch.Tensor] = getattr(model, "_last_target_list", None)

	results: Dict[str, object] = {"per_set": []}

	if recon_list and orig_list and len(recon_list) == len(orig_list):
		# Global (concatenated)
		recon_cat = _stack_events(recon_list)
		orig_cat = _stack_events(orig_list)
		results["global"] = compute_recon_metrics(orig_cat, recon_cat)

		# Per-set metrics
		for i, (r, o) in enumerate(zip(recon_list, orig_list)):
			r_np = _to_numpy(r[0])
			o_np = _to_numpy(o[0])
			res = compute_recon_metrics(o_np, r_np)
			res["set_index"] = i
			res["n_recon"] = int(r_np.shape[0])
			res["n_orig"] = int(o_np.shape[0])
			results["per_set"].append(res)
	else:
		# Fallback to concatenated attributes
		recon_cat = getattr(model, "_last_recon_cat", None)
		orig_cat = getattr(model, "_last_target_cat", None)
		if recon_cat is not None and orig_cat is not None:
			results["global"] = compute_recon_metrics(_to_numpy(orig_cat[0]), _to_numpy(recon_cat[0]))
		else:
			results["error"] = "Reconstruction tensors not found on model"

	return results


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Evaluate SeqSetVAE reconstruction quality")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--checkpoint_type", type=str, choices=["auto", "pretrain", "finetune"], default="auto")
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--params_map_path", type=str, required=True)
	parser.add_argument("--label_path", type=str, required=True)
	parser.add_argument("--num_samples", type=int, default=3)
	parser.add_argument("--save_dir", type=str, default="./recon_eval")
	args = parser.parse_args()

	# Prepare model
	cfg = Config()
	model = load_model(args.checkpoint, cfg, checkpoint_type=args.checkpoint_type)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	# Data
	dm = SeqSetVAEDataModule(
		saved_dir=args.data_dir,
		params_map_path=args.params_map_path,
		label_path=args.label_path,
		batch_size=1,
		num_workers=0,
		pin_memory=False,
	)
	dm.setup()
	dl = dm.train_dataloader()

	# Iterate first K samples deterministically
	all_results: List[Dict] = []
	for i, batch in enumerate(dl):
		if i >= args.num_samples:
			break
		print(f"Evaluating sample {i}...")
		res = evaluate_single_sample(model, batch)
		res["sample_index"] = i
		all_results.append(res)

	# Aggregate summary
	def _agg(key: str) -> float:
		vals = []
		for r in all_results:
			g = r.get("global", {})
			if key in g and np.isfinite(g[key]):
				vals.append(g[key])
		return float(np.mean(vals)) if vals else float("nan")

	summary = {
		"num_samples": len(all_results),
		"nn_l2_mean": _agg("nn_l2_mean"),
		"nn_l2_median": _agg("nn_l2_median"),
		"nn_l2_p95": _agg("nn_l2_p95"),
		"dir_cosine_mean": _agg("dir_cosine_mean"),
		"mag_mae": _agg("mag_mae"),
		"mag_rmse": _agg("mag_rmse"),
		"mag_corr": _agg("mag_corr"),
		"scale_ratio": _agg("scale_ratio"),
		"chamfer_l2_mean": _agg("chamfer_l2_mean"),
	}

	os.makedirs(args.save_dir, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = os.path.join(args.save_dir, f"reconstruction_eval_{ts}.json")
	with open(out_path, "w") as f:
		json.dump({"summary": summary, "details": all_results}, f, indent=2)

	# Pretty print key metrics
	print("\n===== Reconstruction Quality (mean over samples) =====")
	for k, v in summary.items():
		print(f"{k}: {v}")
	print(f"\nSaved detailed report to: {out_path}")


if __name__ == "__main__":
	main()