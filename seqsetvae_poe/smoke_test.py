import os
import sys
import math
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

# Make runnable both as package (python -m seqsetvae_poe.smoke_test)
# and as a standalone script (python seqsetvae_poe/smoke_test.py)
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
try:
    from .model import PoESeqSetVAEPretrain  # type: ignore
    from .dataset import DataModule  # type: ignore
except Exception:
    from model import PoESeqSetVAEPretrain  # type: ignore
    from dataset import DataModule  # type: ignore


def build_synthetic_batch(batch_size: int = 2, max_sets: int = 8, tokens_per_set: int = 16, embed_dim: int = 768):
    """
    Create a synthetic batch compatible with DataModule/_collate_lvcf outputs.
    Shapes:
      - var: [B, N, D]
      - val: [B, N, 1]
      - minute: [B, N, 1]
      - set_id: [B, N, 1]
      - age: [B, N, 1]
      - carry_mask: [B, N, 1]
      - padding_mask: [B, N]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = batch_size
    # total tokens N per patient
    S = max_sets
    n_per_set = tokens_per_set
    N = S * n_per_set

    var = torch.randn(B, N, embed_dim, device=device)
    # values: mostly small; some zeros for masked
    val = (torch.randn(B, N, 1, device=device) * 0.5).clamp_(-3, 3)
    # time: minutes increasing; each set one minute apart
    base = torch.arange(S, device=device).float().view(1, S, 1)
    minute = base.repeat(B, 1, 1).repeat_interleave(n_per_set, dim=1)
    minute = minute + torch.zeros(B, N, 1, device=device)
    # set ids
    set_id = torch.arange(S, device=device).view(1, S, 1).repeat(B, 1, 1).repeat_interleave(n_per_set, dim=1)
    # age (time since last observation for carry-forwarded tokens)
    age = torch.zeros(B, N, 1, device=device)
    # carry mask: within each set, randomly mark ~40% as carry
    carry_mask = (torch.rand(B, N, 1, device=device) < 0.4).float()
    # padding mask: no padding in synthetic batch
    padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    return {
        "var": var,
        "val": val,
        "minute": minute,
        "set_id": set_id,
        "age": age,
        "carry_mask": carry_mask,
        "padding_mask": padding_mask,
    }


class NoPoESeqSetVAE(PoESeqSetVAEPretrain):
    """
    Variant that disables PoE by returning q_x as posterior. Used for Baseline/A/B/AB.
    """

    def _poe(self, mu_qx, logvar_qx, mu_p, logvar_p):  # type: ignore[override]
        return mu_qx, logvar_qx


class ConcatInjectSeqSetVAE(PoESeqSetVAEPretrain):
    """
    Variant that combines q_x and p via simple concatenation + linear map.
    Serves as a naive alternative to PoE for ablation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._merge = nn.Linear(self.hparams.latent_dim * 2, self.hparams.latent_dim * 2)

    def _poe(self, mu_qx, logvar_qx, mu_p, logvar_p):  # type: ignore[override]
        cat = torch.cat([mu_qx, mu_p], dim=-1)
        merged = self._merge(cat)
        mu_post, logvar_post = merged.chunk(2, dim=-1)
        return mu_post, logvar_post


def _compress_like_raw(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    A: 反扩容(exp->raw). Drop carry-forward tokens per set; drop empty sets.
    """
    device = batch["var"].device
    B, N, D = batch["var"].shape
    set_id = batch["set_id"].long().squeeze(-1)
    carry = (batch.get("carry_mask") or torch.zeros(B, N, 1, device=device)).squeeze(-1) > 0.5
    minute = batch["minute"].squeeze(-1)
    var = batch["var"]
    val = batch["val"].squeeze(-1)
    padding = batch.get("padding_mask", torch.zeros(B, N, dtype=torch.bool, device=device))

    new_vars = []
    new_vals = []
    new_minutes = []
    new_setids = []
    for b in range(B):
        mask_valid = ~padding[b]
        sid = set_id[b, mask_valid]
        car = carry[b, mask_valid]
        t = minute[b, mask_valid]
        v = var[b, mask_valid]
        x = val[b, mask_valid]
        # split by consecutive set ids
        uniq, counts = torch.unique_consecutive(sid, return_counts=True)
        idx_splits = torch.split(torch.arange(len(sid), device=device), [int(c) for c in counts])
        kept_tokens = []
        kept_minutes = []
        kept_setids = []
        running_sid = 0
        for idx in idx_splits:
            # keep only non-carry tokens
            keep = ~car[idx]
            if keep.sum() == 0:
                continue
            kept_tokens.append(v[idx][keep])
            kept_minutes.append(t[idx][keep])
            kept_setids.append(torch.full((int(keep.sum()),), running_sid, dtype=torch.long, device=device))
            running_sid += 1
        if len(kept_tokens) == 0:
            # if everything is dropped, keep a single zero token to avoid empty patient
            kept_tokens = [torch.zeros(1, D, device=device)]
            kept_minutes = [torch.zeros(1, device=device)]
            kept_setids = [torch.zeros(1, dtype=torch.long, device=device)]
            x = torch.zeros(1, device=device)
        v_new = torch.cat(kept_tokens, dim=0)
        t_new = torch.cat(kept_minutes, dim=0)
        s_new = torch.cat(kept_setids, dim=0)
        # values for selected tokens
        # regenerate x per kept positions only
        x_new = torch.ones(len(v_new), device=device)
        new_vars.append(v_new)
        new_vals.append(x_new)
        new_minutes.append(t_new)
        new_setids.append(s_new)

    # pad to max length
    max_len = max(v.shape[0] for v in new_vars)
    out_var = torch.zeros(B, max_len, D, device=device)
    out_val = torch.zeros(B, max_len, 1, device=device)
    out_minute = torch.zeros(B, max_len, 1, device=device)
    out_setid = torch.zeros(B, max_len, 1, dtype=torch.long, device=device)
    out_age = torch.zeros(B, max_len, 1, device=device)
    out_carry = torch.zeros(B, max_len, 1, device=device)
    out_pad = torch.ones(B, max_len, dtype=torch.bool, device=device)
    for b in range(B):
        n = new_vars[b].shape[0]
        out_var[b, :n] = new_vars[b]
        out_val[b, :n, 0] = new_vals[b]
        out_minute[b, :n, 0] = new_minutes[b]
        out_setid[b, :n, 0] = new_setids[b]
        out_pad[b, :n] = False
    return {
        "var": out_var,
        "val": out_val,
        "minute": out_minute,
        "set_id": out_setid,
        "age": out_age,
        "carry_mask": out_carry,
        "padding_mask": out_pad,
    }


def _calc_batch_stats(batch: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
    pad = batch.get("padding_mask")
    if pad is None:
        pad = torch.zeros(batch["var"].shape[:2], dtype=torch.bool, device=batch["var"].device)
    total = (~pad).sum().item()
    carry = (batch.get("carry_mask") is not None) and (batch["carry_mask"] > 0.5)
    carry_num = int(carry.sum().item()) if isinstance(carry, torch.Tensor) else 0
    eff = (batch["val"].abs() > 1e-8).sum().item()
    return (
        carry_num / max(1, total),  # staleness ratio
        eff / max(1, total),        # effective mask ratio (non-zero values)
        total / max(1, batch["set_id"].max().item() + 1),  # avg tokens per set
    )


def _build_model(variant: str, device: torch.device, set_mae_ratio: float, enable_c: bool):
    # Small model for speed
    input_dim = 768
    reduced_dim = 256
    latent_dim = 128
    levels = 2
    heads = 4
    m = 16
    ff_dim = 512
    transformer_heads = 4
    transformer_layers = 2

    common_kwargs = dict(
        input_dim=input_dim,
        reduced_dim=reduced_dim,
        latent_dim=latent_dim,
        levels=levels,
        heads=heads,
        m=m,
        beta=1.0,
        lr=3e-4,
        ff_dim=ff_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        transformer_dropout=0.1,
        warmup_beta=True,
        max_beta=0.05,
        beta_warmup_steps=1000,
        free_bits=0.03,
        stale_dropout_p=0.2,
        set_mae_ratio=set_mae_ratio,
        enable_next_change=enable_c,
        next_change_weight=0.3,
    )

    if variant in {"baseline", "A", "B", "AB"}:
        model = NoPoESeqSetVAE(**common_kwargs)
    elif variant in {"AB_concat"}:
        model = ConcatInjectSeqSetVAE(**common_kwargs)
    else:
        model = PoESeqSetVAEPretrain(**common_kwargs)
    return model.to(device)


def run_variant(variant: str, use_real: bool, data_dir: str, steps: int, mask_ratio: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose set_mae_ratio
    set_mae_ratio = mask_ratio if variant in {"B", "AB", "AB_poe", "AB_concat", "ABC"} else 0.0
    enable_c = variant == "ABC"

    model = _build_model(variant, device, set_mae_ratio=set_mae_ratio, enable_c=enable_c)
    optimizer = AdamW(model.parameters(), lr=3e-4)

    # batch source
    if use_real:
        dm = DataModule(saved_dir=data_dir, batch_size=8, num_workers=0, smoke=True, smoke_batch_size=8)
        dm.setup()
        batch = dm.build_smoke_batch()
        # move to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    else:
        batch = build_synthetic_batch()

    # Variant A: reverse expansion
    if variant in {"A", "AB", "AB_poe", "AB_concat", "ABC"}:
        batch = _compress_like_raw(batch)

    stale_ratio, eff_ratio, tokens_per_set = _calc_batch_stats(batch)

    hist = {"recon": [], "kl": [], "next_change": [], "beta": [], "total": []}
    model.train()
    for _ in range(max(1, steps)):
        recon, kl, next_c = model.forward(batch)
        beta = model._beta()
        total = recon + beta * kl + (0.3 * next_c if enable_c else 0.0)
        total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        hist["recon"].append(float(recon.detach().cpu()))
        hist["kl"].append(float(kl.detach().cpu()))
        hist["next_change"].append(float(next_c.detach().cpu()))
        hist["beta"].append(float(beta))
        hist["total"].append(float(total.detach().cpu()))

    out = {
        "variant": variant,
        "steps": steps,
        "recon_mean": sum(hist["recon"]) / len(hist["recon"]),
        "kl_mean": sum(hist["kl"]) / len(hist["kl"]),
        "next_change_mean": sum(hist["next_change"]) / len(hist["next_change"]),
        "beta_last": hist["beta"][-1],
        "total_last": hist["total"][-1],
        "stale_ratio": stale_ratio,
        "effective_value_ratio": eff_ratio,
        "tokens_per_set": tokens_per_set,
        "device": str(device),
        "set_mae_ratio": set_mae_ratio,
        "enable_poe": variant in {"AB_poe", "ABC"},
        "enable_next_change": enable_c,
    }
    print(out)


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAE-PoE smoke ablation runner")
    parser.add_argument("--variant", type=str, default="AB_poe", choices=[
        "baseline", "A", "B", "AB", "AB_poe", "AB_concat", "ABC",
    ])
    parser.add_argument("--use_real_batch", action="store_true")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SEQSET_PTN_DATA", ""))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Set-MAE mask ratio for B variants (0.2-0.4 suggested)")
    args = parser.parse_args()

    if args.use_real_batch and not args.data_dir:
        raise SystemExit("--data_dir must be provided when --use_real_batch is set")

    run_variant(args.variant, args.use_real_batch, args.data_dir, args.steps, args.mask_ratio)


if __name__ == "__main__":
    main()

