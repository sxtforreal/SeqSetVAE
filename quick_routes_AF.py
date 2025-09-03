#!/usr/bin/env python3
"""
Quick A–F Routes Evaluator

README
- Required files in --data_dir:
  - mu_train.npy, mu_val.npy, mu_test.npy                 # shapes [N,S,D]
  - logvar_train.npy, logvar_val.npy, logvar_test.npy     # optional, same shapes
  - y_train.npy, y_val.npy, y_test.npy                    # shapes [N]

- Example run:
  python quick_routes_AF.py \
    --data_dir /path/to/data \
    --seeds 0 1 2 \
    --k_list 16 32 64 \
    --l1_C 0.02 0.05 0.1 0.2 0.5 \
    --epochs 20 \
    --out_csv routes_results.csv
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay

import torch
import torch.nn as nn


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npy_or_none(path: str) -> Optional[np.ndarray]:
    return np.load(path) if os.path.isfile(path) else None


def mean_pool(mu: np.ndarray) -> np.ndarray:
    # mu: [N,S,D] -> [N,D]
    return mu.mean(axis=1)


def stats_pool(arr: np.ndarray) -> np.ndarray:
    # arr: [N,S,D] -> concat mean,std,max,q25,q75 along last dim -> [N,5D]
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    maxv = arr.max(axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    return np.concatenate([mean, std, maxv, q25, q75], axis=1)


def var_from_logvar(logvar: np.ndarray) -> np.ndarray:
    return np.exp(logvar)


def au_scores(mu_train: np.ndarray) -> np.ndarray:
    # AU_j = variance across samples and sets for each dim j
    # mu_train: [N,S,D]
    N, S, D = mu_train.shape
    flat = mu_train.reshape(N * S, D)
    return flat.var(axis=0)


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y[mask])) if np.any(mask) else 0.0
        ece += abs(acc - conf) * (np.sum(mask) / n)
    return float(ece)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def eval_linear(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    penalty: str = 'l2',
    C: Optional[float] = None,
    class_weight: str = 'balanced',
) -> Dict[str, float]:
    C_val = 1.0 if C is None else C
    lr = LogisticRegression(
        penalty=penalty,
        C=C_val,
        solver='liblinear' if penalty == 'l1' else 'lbfgs',
        max_iter=2000,
        class_weight=class_weight,
    )
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', lr),
    ])
    pipe.fit(Xtr, ytr)
    val_probs = pipe.predict_proba(Xval)[:, 1]
    test_probs = pipe.predict_proba(Xte)[:, 1]
    out = {
        'Val_AUPRC': float(average_precision_score(yval, val_probs)),
        'Val_AUROC': float(roc_auc_score(yval, val_probs)) if len(np.unique(yval)) > 1 else float('nan'),
        'Test_AUPRC': float(average_precision_score(yte, test_probs)),
        'Test_AUROC': float(roc_auc_score(yte, test_probs)) if len(np.unique(yte)) > 1 else float('nan'),
        'Test_ECE': ece_score(test_probs, yte, n_bins=10),
        'Test_Brier': brier_score(test_probs, yte),
    }
    return out


# -----------------------------
# Torch models for A and F
# -----------------------------

class AttentionPoolNet(nn.Module):
    def __init__(self, dim: int, att_dim: int = 64):
        super().__init__()
        self.Wq = nn.Linear(dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.out = nn.Linear(dim, 1)

    def forward(self, mu: torch.Tensor):
        # mu: [N,S,D]
        N, S, D = mu.shape
        q = torch.tanh(self.Wq(mu))          # [N,S,att]
        a = torch.softmax(self.v(q).squeeze(-1), dim=1)  # [N,S]
        z = torch.sum(a.unsqueeze(-1) * mu, dim=1)       # [N,D]
        logit = self.out(z).squeeze(-1)                  # [N]
        return logit


class SmallMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)


def make_class_weights(y: np.ndarray) -> float:
    # return pos_weight for BCEWithLogitsLoss
    y = y.astype(np.int64)
    c = np.bincount(y, minlength=2)
    pos_w = (c[0] + c[1]) / (2.0 * max(1, c[1]))
    return float(pos_w)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        xb, = batch
        xb = xb.to(device)
        logits = model(xb) if xb.ndim == 2 else model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def train_torch_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    device: str,
    is_attention: bool,
    patience: int = 5,
) -> Tuple[nn.Module, Dict[str, float]]:
    Xtr, ytr = train_data
    Xval, yval = val_data
    model = model.to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    yval_t = torch.tensor(yval, dtype=torch.float32, device=device)
    pos_w = make_class_weights(ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if is_attention:
        tr_tensor = torch.tensor(Xtr, dtype=torch.float32)
        val_tensor = torch.tensor(Xval, dtype=torch.float32)
    else:
        tr_tensor = torch.tensor(Xtr, dtype=torch.float32)
        val_tensor = torch.tensor(Xval, dtype=torch.float32)

    tr_loader = torch.utils.data.DataLoader([(tr_tensor[i],) for i in range(len(tr_tensor))], batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader([(val_tensor[i],) for i in range(len(val_tensor))], batch_size=256, shuffle=False)

    best_val = -np.inf
    best_state = None
    no_imp = 0
    for _ in range(epochs):
        model.train()
        for (xb,) in tr_loader:
            xb = xb.to(device)
            logits = model(xb)
            loss = criterion(logits, ytr_t[:xb.shape[0]]) if is_attention else criterion(logits, ytr_t[:xb.shape[0]])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # val
        with torch.no_grad():
            model.eval()
            probs_val = []
            for (xb,) in val_loader:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).detach().cpu().numpy()
                probs_val.append(p)
            probs_val = np.concatenate(probs_val)
            val_auprc = float(average_precision_score(yval, probs_val))
        if val_auprc > best_val + 1e-6:
            best_val = val_auprc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'Val_AUPRC': best_val}


def refit_and_eval_torch(
    make_model_fn,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    epochs: int,
    device: str,
    is_attention: bool,
) -> Dict[str, float]:
    Xtv = np.concatenate([Xtr, Xval], axis=0)
    ytv = np.concatenate([ytr, yval], axis=0)
    model = make_model_fn()
    model, _ = train_torch_model(model, (Xtv, ytv), (Xval, yval), epochs=epochs, device=device, is_attention=is_attention)
    # Evaluate on test
    with torch.no_grad():
        model.eval()
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        logits = model(Xte_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return {
        'Test_AUPRC': float(average_precision_score(yte, probs)),
        'Test_AUROC': float(roc_auc_score(yte, probs)) if len(np.unique(yte)) > 1 else float('nan'),
        'Test_ECE': ece_score(probs, yte, n_bins=10),
        'Test_Brier': brier_score(probs, yte),
    }


# -----------------------------
# Routes A–F
# -----------------------------

def route_A_attention(mu_tr, ytr, mu_val, yval, mu_te, yte, epochs: int, device: str) -> Tuple[Dict[str, float], int]:
    D = mu_tr.shape[-1]
    def make_model():
        return AttentionPoolNet(dim=D, att_dim=64)
    model = make_model()
    model, info = train_torch_model(model, (mu_tr, ytr), (mu_val, yval), epochs=epochs, device=device, is_attention=True)
    # Refit on TRAIN+VAL and eval TEST
    metrics = refit_and_eval_torch(lambda: AttentionPoolNet(dim=D, att_dim=64), mu_tr, ytr, mu_val, yval, mu_te, yte, epochs=epochs, device=device, is_attention=True)
    metrics.update({'Val_AUPRC': info['Val_AUPRC']})
    return metrics, D


def route_B1_au_topk(mu_tr, ytr, mu_val, yval, mu_te, yte, k_list: List[int]) -> Tuple[Dict[str, float], int, int]:
    Xtr_full = mean_pool(mu_tr)
    Xval_full = mean_pool(mu_val)
    Xte_full = mean_pool(mu_te)
    au = au_scores(mu_tr)
    best = None
    best_k = None
    for k in k_list:
        idx = np.argsort(-au)[:k]
        res = eval_linear(Xtr_full[:, idx], ytr, Xval_full[:, idx], yval, Xte_full[:, idx], yte, penalty='l2', C=1.0)
        if (best is None) or (res['Val_AUPRC'] > best['Val_AUPRC']):
            best = res
            best_k = int(k)
    # Refit with best k on TRAIN+VAL, then eval TEST
    idx = np.argsort(-au)[:best_k]
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')),
    ])
    Xtv = np.concatenate([Xtr_full[:, idx], Xval_full[:, idx]], axis=0)
    ytv = np.concatenate([ytr, yval], axis=0)
    pipe.fit(Xtv, ytv)
    test_probs = pipe.predict_proba(Xte_full[:, idx])[:, 1]
    best.update({
        'Test_AUPRC': float(average_precision_score(yte, test_probs)),
        'Test_AUROC': float(roc_auc_score(yte, test_probs)) if len(np.unique(yte)) > 1 else float('nan'),
        'Test_ECE': ece_score(test_probs, yte, n_bins=10),
        'Test_Brier': brier_score(test_probs, yte),
    })
    return best, best_k, int(best_k)


def route_B2_l1_sparse(mu_tr, ytr, mu_val, yval, mu_te, yte, C_list: List[float]) -> Tuple[Dict[str, float], float, int]:
    Xtr = mean_pool(mu_tr)
    Xval = mean_pool(mu_val)
    Xte = mean_pool(mu_te)
    best = None
    best_C = None
    best_idx = None
    for C in C_list:
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', LogisticRegression(penalty='l1', C=float(C), solver='liblinear', max_iter=2000, class_weight='balanced')),
        ])
        pipe.fit(Xtr, ytr)
        val_probs = pipe.predict_proba(Xval)[:, 1]
        res = {
            'Val_AUPRC': float(average_precision_score(yval, val_probs)),
            'Val_AUROC': float(roc_auc_score(yval, val_probs)) if len(np.unique(yval)) > 1 else float('nan'),
        }
        coef = pipe.named_steps['clf'].coef_[0]
        idx = np.where(np.abs(coef) > 1e-8)[0]
        if (best is None) or (res['Val_AUPRC'] > best['Val_AUPRC']):
            best = res
            best_C = float(C)
            best_idx = idx
    if best_idx is None or len(best_idx) == 0:
        best_idx = np.arange(Xtr.shape[1])
    # Refit on selected dims
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')),
    ])
    Xtv = np.concatenate([Xtr[:, best_idx], Xval[:, best_idx]], axis=0)
    ytv = np.concatenate([ytr, yval], axis=0)
    pipe.fit(Xtv, ytv)
    test_probs = pipe.predict_proba(Xte[:, best_idx])[:, 1]
    best.update({
        'Test_AUPRC': float(average_precision_score(yte, test_probs)),
        'Test_AUROC': float(roc_auc_score(yte, test_probs)) if len(np.unique(yte)) > 1 else float('nan'),
        'Test_ECE': ece_score(test_probs, yte, n_bins=10),
        'Test_Brier': brier_score(test_probs, yte),
    })
    return best, float(best_C), int(len(best_idx))


def route_C_baseline(mu_tr, ytr, mu_val, yval, mu_te, yte) -> Tuple[Dict[str, float], int]:
    Xtr = mean_pool(mu_tr)
    Xval = mean_pool(mu_val)
    Xte = mean_pool(mu_te)
    res = eval_linear(Xtr, ytr, Xval, yval, Xte, yte, penalty='l2', C=1.0)
    return res, Xtr.shape[1]


def route_D_stats_uncert(mu_tr, ytr, mu_val, yval, mu_te, yte, logvar_tr=None, logvar_val=None, logvar_te=None) -> Tuple[Dict[str, float], int]:
    X_mu_tr = stats_pool(mu_tr)
    X_mu_val = stats_pool(mu_val)
    X_mu_te = stats_pool(mu_te)
    if logvar_tr is not None and logvar_val is not None and logvar_te is not None:
        v_tr = var_from_logvar(logvar_tr)
        v_val = var_from_logvar(logvar_val)
        v_te = var_from_logvar(logvar_te)
        X_v_tr = stats_pool(v_tr)
        X_v_val = stats_pool(v_val)
        X_v_te = stats_pool(v_te)
        Xtr = np.concatenate([X_mu_tr, X_v_tr], axis=1)
        Xval = np.concatenate([X_mu_val, X_v_val], axis=1)
        Xte = np.concatenate([X_mu_te, X_v_te], axis=1)
    else:
        Xtr, Xval, Xte = X_mu_tr, X_mu_val, X_mu_te
    res = eval_linear(Xtr, ytr, Xval, yval, Xte, yte, penalty='l2', C=1.0)
    return res, Xtr.shape[1]


def icp_eval(best_method: str, probs_val: np.ndarray, yval: np.ndarray, probs_te: np.ndarray, yte: np.ndarray) -> pd.DataFrame:
    coverages = [0.5, 0.6, 0.7, 0.8, 0.9]
    order = np.argsort(-probs_val)
    out_rows = []
    for g in coverages:
        k = int(np.ceil(len(order) * g))
        idx = order[:k]
        # Apply same thresholding style to test by top-g fraction
        thr = np.sort(probs_val)[-k] if k > 0 else 1.0
        mask_te = probs_te >= thr
        if not np.any(mask_te):
            test_auprc = 0.0
        else:
            test_auprc = float(average_precision_score(yte[mask_te], probs_te[mask_te]))
        out_rows.append({'coverage': g, 'test_auprc': test_auprc, 'method': f'ICP_{best_method}'})
    return pd.DataFrame(out_rows)


def route_F_small_mlp(mu_tr, ytr, mu_val, yval, mu_te, yte, epochs: int, device: str) -> Tuple[Dict[str, float], int]:
    D = mu_tr.shape[-1]
    Xtr = mean_pool(mu_tr)
    Xval = mean_pool(mu_val)
    Xte = mean_pool(mu_te)
    def make_model():
        return SmallMLP(dim=D)
    model = make_model()
    model, info = train_torch_model(model, (Xtr, ytr), (Xval, yval), epochs=epochs, device=device, is_attention=False)
    # Refit and eval
    metrics = refit_and_eval_torch(lambda: SmallMLP(dim=D), Xtr, ytr, Xval, yval, Xte, yte, epochs=epochs, device=device, is_attention=False)
    metrics.update({'Val_AUPRC': info['Val_AUPRC']})
    return metrics, D


# -----------------------------
# Main orchestration
# -----------------------------

def run_for_seed(args, seed: int) -> List[Dict]:
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = args.data_dir
    mu_tr = np.load(os.path.join(data_dir, 'mu_train.npy'))
    mu_val = np.load(os.path.join(data_dir, 'mu_val.npy'))
    mu_te = np.load(os.path.join(data_dir, 'mu_test.npy'))
    ytr = np.load(os.path.join(data_dir, 'y_train.npy')).astype(np.int64)
    yval = np.load(os.path.join(data_dir, 'y_val.npy')).astype(np.int64)
    yte = np.load(os.path.join(data_dir, 'y_test.npy')).astype(np.int64)
    logvar_tr = load_npy_or_none(os.path.join(data_dir, 'logvar_train.npy'))
    logvar_val = load_npy_or_none(os.path.join(data_dir, 'logvar_val.npy'))
    logvar_te = load_npy_or_none(os.path.join(data_dir, 'logvar_test.npy'))

    records = []

    # Route C baseline first (reference A0)
    resC, nC = route_C_baseline(mu_tr, ytr, mu_val, yval, mu_te, yte)
    records.append({
        'seed': seed, 'method': 'C_baseline', 'chosen_k': None, 'chosen_C': None, 'params': json.dumps({}),
        'Test_AUPRC': resC['Test_AUPRC'], 'Test_AUROC': resC['Test_AUROC'], 'Test_ECE': resC['Test_ECE'], 'Test_Brier': resC['Test_Brier'],
        'n_features_used': nC, 'notes': ''
    })

    # Route B1
    resB1, kB1, _ = route_B1_au_topk(mu_tr, ytr, mu_val, yval, mu_te, yte, args.k_list)
    records.append({
        'seed': seed, 'method': 'B1_au_topk', 'chosen_k': kB1, 'chosen_C': None, 'params': json.dumps({'k_list': args.k_list}),
        'Test_AUPRC': resB1['Test_AUPRC'], 'Test_AUROC': resB1['Test_AUROC'], 'Test_ECE': resB1['Test_ECE'], 'Test_Brier': resB1['Test_Brier'],
        'n_features_used': kB1, 'notes': ''
    })

    # Route B2
    resB2, CB2, kB2 = route_B2_l1_sparse(mu_tr, ytr, mu_val, yval, mu_te, yte, args.l1_C)
    records.append({
        'seed': seed, 'method': 'B2_l1_sparse', 'chosen_k': kB2, 'chosen_C': CB2, 'params': json.dumps({'C_list': args.l1_C}),
        'Test_AUPRC': resB2['Test_AUPRC'], 'Test_AUROC': resB2['Test_AUROC'], 'Test_ECE': resB2['Test_ECE'], 'Test_Brier': resB2['Test_Brier'],
        'n_features_used': kB2, 'notes': ''
    })

    # Route D
    resD, nD = route_D_stats_uncert(mu_tr, ytr, mu_val, yval, mu_te, yte, logvar_tr, logvar_val, logvar_te)
    records.append({
        'seed': seed, 'method': 'D_stats_uncert', 'chosen_k': None, 'chosen_C': None, 'params': json.dumps({'with_logvar': logvar_tr is not None}),
        'Test_AUPRC': resD['Test_AUPRC'], 'Test_AUROC': resD['Test_AUROC'], 'Test_ECE': resD['Test_ECE'], 'Test_Brier': resD['Test_Brier'],
        'n_features_used': nD, 'notes': ''
    })

    # Route A
    resA, nA = route_A_attention(mu_tr, ytr, mu_val, yval, mu_te, yte, epochs=args.epochs, device=device)
    records.append({
        'seed': seed, 'method': 'A_attention', 'chosen_k': None, 'chosen_C': None, 'params': json.dumps({'epochs': args.epochs}),
        'Test_AUPRC': resA['Test_AUPRC'], 'Test_AUROC': resA['Test_AUROC'], 'Test_ECE': resA['Test_ECE'], 'Test_Brier': resA['Test_Brier'],
        'n_features_used': nA, 'notes': ''
    })

    # Route F
    resF, nF = route_F_small_mlp(mu_tr, ytr, mu_val, yval, mu_te, yte, epochs=args.epochs, device=device)
    records.append({
        'seed': seed, 'method': 'F_small_mlp', 'chosen_k': None, 'chosen_C': None, 'params': json.dumps({'epochs': args.epochs}),
        'Test_AUPRC': resF['Test_AUPRC'], 'Test_AUROC': resF['Test_AUROC'], 'Test_ECE': resF['Test_ECE'], 'Test_Brier': resF['Test_Brier'],
        'n_features_used': nF, 'notes': ''
    })

    # Route E (ICP on best validation method among A,B1,B2,C,D)
    # Build VAL predictions for each and select best by Val_AUPRC
    # For simplicity, reuse already trained objects by quickly refitting and getting probs on VAL only
    # C
    Xtr_C, Xval_C, Xte_C = mean_pool(mu_tr), mean_pool(mu_val), mean_pool(mu_te)
    pipe_C = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')),
    ])
    pipe_C.fit(Xtr_C, ytr)
    val_C = pipe_C.predict_proba(Xval_C)[:, 1]
    te_C = pipe_C.predict_proba(Xte_C)[:, 1]
    cand = [('C', float(average_precision_score(yval, val_C)), val_C, te_C)]
    # B1
    au = au_scores(mu_tr)
    best_val_b1 = -1.0
    best_valpred_b1 = None
    best_te_b1 = None
    for k in args.k_list:
        idx = np.argsort(-au)[:k]
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')),
        ])
        pipe.fit(Xtr_C[:, idx], ytr)
        vp = pipe.predict_proba(Xval_C[:, idx])[:, 1]
        sc = float(average_precision_score(yval, vp))
        if sc > best_val_b1:
            best_val_b1 = sc
            best_valpred_b1 = vp
            best_te_b1 = pipe.predict_proba(Xte_C[:, idx])[:, 1]
    cand.append(('B1', best_val_b1, best_valpred_b1, best_te_b1))
    # B2
    best_val_b2 = -1.0
    best_valpred_b2 = None
    best_te_b2 = None
    for Cgrid in args.l1_C:
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', LogisticRegression(penalty='l1', C=float(Cgrid), solver='liblinear', max_iter=2000, class_weight='balanced')),
        ])
        pipe.fit(Xtr_C, ytr)
        vp = pipe.predict_proba(Xval_C)[:, 1]
        sc = float(average_precision_score(yval, vp))
        if sc > best_val_b2:
            best_val_b2 = sc
            best_valpred_b2 = vp
            best_te_b2 = pipe.predict_proba(Xte_C)[:, 1]
    cand.append(('B2', best_val_b2, best_valpred_b2, best_te_b2))
    # D
    if logvar_tr is not None and logvar_val is not None and logvar_te is not None:
        v_tr = var_from_logvar(logvar_tr)
        v_val = var_from_logvar(logvar_val)
        v_te = var_from_logvar(logvar_te)
        Xtr_D = np.concatenate([stats_pool(mu_tr), stats_pool(v_tr)], axis=1)
        Xval_D = np.concatenate([stats_pool(mu_val), stats_pool(v_val)], axis=1)
        Xte_D = np.concatenate([stats_pool(mu_te), stats_pool(v_te)], axis=1)
    else:
        Xtr_D = stats_pool(mu_tr)
        Xval_D = stats_pool(mu_val)
        Xte_D = stats_pool(mu_te)
    pipe_D = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')),
    ])
    pipe_D.fit(Xtr_D, ytr)
    val_D = pipe_D.predict_proba(Xval_D)[:, 1]
    te_D = pipe_D.predict_proba(Xte_D)[:, 1]
    cand.append(('D', float(average_precision_score(yval, val_D)), val_D, te_D))
    # A
    with torch.no_grad():
        Ddim = mu_tr.shape[-1]
        netA = AttentionPoolNet(Ddim, 64).to(device)
        # quick train few epochs for validation selection (reuse provided epochs)
        netA, infoA = train_torch_model(netA, (mu_tr, ytr), (mu_val, yval), epochs=args.epochs, device=device, is_attention=True)
        val_A_probs = torch.sigmoid(netA(torch.tensor(mu_val, dtype=torch.float32, device=device))).detach().cpu().numpy()
        te_A_probs = torch.sigmoid(netA(torch.tensor(mu_te, dtype=torch.float32, device=device))).detach().cpu().numpy()
        cand.append(('A', float(average_precision_score(yval, val_A_probs)), val_A_probs, te_A_probs))
    # F
    with torch.no_grad():
        Ddim = mu_tr.shape[-1]
        netF = SmallMLP(Ddim).to(device)
        XtrF = mean_pool(mu_tr)
        XvalF = mean_pool(mu_val)
        XteF = mean_pool(mu_te)
        netF, infoF = train_torch_model(netF, (XtrF, ytr), (XvalF, yval), epochs=args.epochs, device=device, is_attention=False)
        val_F_probs = torch.sigmoid(netF(torch.tensor(XvalF, dtype=torch.float32, device=device))).detach().cpu().numpy()
        te_F_probs = torch.sigmoid(netF(torch.tensor(XteF, dtype=torch.float32, device=device))).detach().cpu().numpy()
        cand.append(('F', float(average_precision_score(yval, val_F_probs)), val_F_probs, te_F_probs))

    best_method, best_val_score, best_val_probs, best_te_probs = max(cand, key=lambda x: x[1])
    df_icp = icp_eval(best_method, best_val_probs, yval, best_te_probs, yte)
    for _, row in df_icp.iterrows():
        records.append({
            'seed': seed, 'method': row['method'], 'chosen_k': None, 'chosen_C': None, 'params': json.dumps({'coverage': row['coverage']}),
            'Test_AUPRC': row['test_auprc'], 'Test_AUROC': None, 'Test_ECE': None, 'Test_Brier': None,
            'n_features_used': None, 'notes': ''
        })

    return records


def summarize_and_decide(df_all: pd.DataFrame):
    # Compute means by method
    summary = df_all.groupby('method')['Test_AUPRC'].mean().to_dict()
    A0 = summary.get('C_baseline', float('nan'))
    best_B = max(summary.get('B1_au_topk', float('-inf')), summary.get('B2_l1_sparse', float('-inf')))
    A_mean = summary.get('A_attention', float('-inf'))
    D_mean = summary.get('D_stats_uncert', float('-inf'))
    F_mean = summary.get('F_small_mlp', float('-inf'))
    # ICP at 0.7
    icp_rows = df_all[df_all['method'].str.startswith('ICP_')]
    icp_07 = icp_rows[icp_rows['params'].apply(lambda s: json.loads(s).get('coverage') == 0.7)]['Test_AUPRC'].mean() if not icp_rows.empty else float('-inf')

    beneficial = {'A': False, 'B': False, 'D': False, 'E': False, 'F': False}

    # 2) B beneficial
    if ((best_B >= (A0 - 0.005) and any(df_all[df_all['method']=='B1_au_topk']['chosen_k'] <= 64)) or (best_B >= (A0 + 0.01))):
        beneficial['B'] = True

    # 3) A beneficial
    if A_mean >= max(A0, best_B) + 0.01:
        beneficial['A'] = True

    # 4) D beneficial
    if D_mean >= max(A0, best_B, A_mean) + 0.005:
        beneficial['D'] = True

    # 5) E beneficial
    if icp_07 >= (A0 - 0.005):
        beneficial['E'] = True

    # 6) F beneficial
    if (F_mean >= max(A0, best_B) + 0.01) and (not beneficial['A']):
        beneficial['F'] = True

    # Print compact summary
    print("=== Mean Test AUPRC by method ===")
    for m, v in sorted(summary.items()):
        print(f"{m}: {v:.4f}")

    # Final route recommendation
    if beneficial['A']:
        suffix = (' + D' if beneficial['D'] else '') + (' + E' if beneficial['E'] else '')
        print(f"ROUTE: A (attention pooling){suffix}")
    elif beneficial['B']:
        b1 = summary.get('B1_au_topk', float('-inf'))
        b2 = summary.get('B2_l1_sparse', float('-inf'))
        which = 'B2' if b2 >= b1 else 'B1'
        suffix = (' + D' if beneficial['D'] else '') + (' + E' if beneficial['E'] else '')
        print(f"ROUTE: B (dim-selection, prefer {which}){suffix}")
    elif beneficial['F']:
        suffix = (' + D' if beneficial['D'] else '') + (' + E' if beneficial['E'] else '')
        print(f"ROUTE: F (small MLP head){suffix}")
    else:
        print("ROUTE: C (baseline). Consider D/E only if marked beneficial.")


def main():
    parser = argparse.ArgumentParser(description="Quick A–F route tests")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='*', default=[0,1,2])
    parser.add_argument('--k_list', type=int, nargs='*', default=[16,32,64])
    parser.add_argument('--l1_C', type=float, nargs='*', default=[0.02,0.05,0.1,0.2,0.5])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--out_csv', type=str, default='routes_results.csv')
    args = parser.parse_args()

    all_records: List[Dict] = []
    for seed in args.seeds:
        recs = run_for_seed(args, seed)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    if os.path.isfile(args.out_csv):
        df_old = pd.read_csv(args.out_csv)
        df = pd.concat([df_old, df], axis=0, ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved results to {args.out_csv}")

    summarize_and_decide(df)


if __name__ == '__main__':
    main()

