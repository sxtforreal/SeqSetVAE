#!/usr/bin/env python3
"""
Unified training entrypoint with --branch {setvae, generative, discriminative}.

- setvae: SetVAE-only pretraining (formerly Stage A)
- generative: PoE dynamics + GRU backbone (formerly Stage B → GRU under generative)
- discriminative: downstream classifiers (transformer baseline over SetVAE, or PoE-based classifier; formerly Stage C)

Backwards compatibility: optional --stage {A,B,C} is accepted and mapped to the new --branch.
"""
from __future__ import annotations
import argparse
import os
import sys
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

import config as cfg  # type: ignore
from dataset import DataModule, MortalityDataModule  # type: ignore
from model import (
    SetVAEOnlyPretrain,
    PoESeqSetVAEPretrain,
    MortalityClassifier,
    _load_state_dict,
    _build_poe_from_state,
    TransformerSetVAEClassifier,
    _build_setvae_from_state,
)  # type: ignore


def _load_state_dict_flex(path: str) -> dict:
    """Load a checkpoint file and return a plain state_dict.

    - Unwraps under 'state_dict' if present
    - Strips leading 'model.' prefix if present
    """
    try:
        state = pl.utilities.cloud_io.load(path, map_location="cpu")  # type: ignore
    except Exception:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    return state


def _detect_setvae_prefix(keys: list[str]) -> str:
    """Detect the most likely SetVAE prefix in a state_dict's keys."""
    candidates = [
        "set_encoder.",
        "setvae.setvae.",
        "setvae.",
    ]
    counts = {p: 0 for p in candidates}
    for k in keys:
        for p in candidates:
            if k.startswith(p):
                counts[p] += 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else ""


def _remap_setvae_to_target(
    setvae_state: dict,
    target_model_state: dict,
    drop_flows: bool = True,
) -> dict:
    """Map Stage A SetVAE weights to target model's SetVAE prefix and filter by matching shape.

    Returns a dict containing only remapped SetVAE weights compatible with the target model.
    """
    src_prefix = _detect_setvae_prefix(list(setvae_state.keys()))
    # Determine destination prefix from model state
    dst_prefix = "set_encoder."
    if not any(k.startswith(dst_prefix) for k in target_model_state.keys()):
        # Rare older layout fallback
        if any(k.startswith("setvae.setvae.") for k in target_model_state.keys()):
            dst_prefix = "setvae.setvae."
        elif any(k.startswith("setvae.") for k in target_model_state.keys()):
            dst_prefix = "setvae."

    if not src_prefix:
        return {}

    remapped = {}
    for k, v in setvae_state.items():
        if not k.startswith(src_prefix):
            continue
        if drop_flows and ".flows." in k:
            continue
        target_key = dst_prefix + k[len(src_prefix):]
        if target_key in target_model_state and target_model_state[target_key].shape == v.shape:
            remapped[target_key] = v
    return remapped


def _strip_flags_from_argv(flag_names: list[str]):
    """Remove flags (and their values when provided as separate tokens) from sys.argv."""
    argv = sys.argv
    out = [argv[0]]
    skip = 0
    for i, tok in enumerate(argv[1:], start=1):
        if skip:
            skip -= 1
            continue
        matched = False
        for name in flag_names:
            if tok == name:
                skip = 1  # skip its value
                matched = True
                break
            if tok.startswith(name + "="):
                matched = True
                break
        if matched:
            continue
        out.append(tok)
    sys.argv = out


def main():
    ap = argparse.ArgumentParser(description="Train with --branch {setvae,generative,discriminative}", add_help=False)
    ap.add_argument("--branch", type=str, choices=["setvae", "generative", "discriminative"], required=True)
    args, _ = ap.parse_known_args()
    branch = args.branch
    # Drop --branch itself before deeper parsing
    _strip_flags_from_argv(["--branch"])
    if branch == "setvae":
        return _run_setvae()
    if branch == "generative":
        return _run_generative()
    return _run_discriminative()


def _run_setvae():
    parser = argparse.ArgumentParser(description="setvae - SetVAE pretraining")
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    parser.add_argument("--params_map_path", type=str, default=getattr(cfg, "params_map_path", ""))
    parser.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 4))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 4))
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run_name", type=str, default="SetVAE-Only-PT")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=getattr(cfg, "log_every_n_steps", 50))
    parser.add_argument("--limit_val_batches", type=float, default=getattr(cfg, "limit_val_batches", 1.0))
    # Validate once mid-epoch by default (0.5) for GRU pretraining
    parser.add_argument("--val_check_interval", type=float, default=0.5)
    # InfoVAE / Flows
    parser.add_argument("--use_flows", action="store_true", default=False)
    parser.add_argument("--num_flows", type=int, default=0)
    parser.add_argument("--mmd_weight", type=float, default=0.0)
    parser.add_argument("--mmd_scales", type=float, nargs='*', default=[1.0, 2.0, 4.0, 8.0])
    # KL fairness & stability controls
    parser.add_argument("--kl_fairness_weight", type=float, default=0.1)
    parser.add_argument("--kl_spread_tol", type=float, default=1.0)
    parser.add_argument("--kl_over_weight", type=float, default=1.0)
    parser.add_argument("--var_stability_weight", type=float, default=0.01)
    parser.add_argument("--per_dim_free_bits", type=float, default=0.002)
    # Optim & schedules
    parser.add_argument("--lr", type=float, default=getattr(cfg, "lr", 3e-4))
    parser.add_argument("--gradient_clip_val", type=float, default=getattr(cfg, "gradient_clip_val", 0.2))
    parser.add_argument("--warmup_beta", action="store_true", default=getattr(cfg, "warmup_beta", True))
    parser.add_argument("--max_beta", type=float, default=0.2)
    parser.add_argument("--beta_warmup_steps", type=int, default=getattr(cfg, "beta_warmup_steps", 8000))
    parser.add_argument("--free_bits", type=float, default=0.05)
    parser.add_argument("--use_kl_capacity", action="store_true", default=getattr(cfg, "use_kl_capacity", True))
    parser.add_argument("--capacity_per_dim_end", type=float, default=getattr(cfg, "capacity_per_dim_end", 0.03))
    parser.add_argument("--capacity_warmup_steps", type=int, default=getattr(cfg, "capacity_warmup_steps", 20000))
    # Reconstruction weighting
    parser.add_argument("--recon_alpha", type=float, default=1.0)
    parser.add_argument("--recon_beta", type=float, default=4.0)
    parser.add_argument("--recon_gamma", type=float, default=3.0)
    parser.add_argument("--recon_scale_calib", type=float, default=1.0)
    parser.add_argument("--recon_beta_var", type=float, default=0.0)
    # Sinkhorn (方案A)
    parser.add_argument("--use_sinkhorn", action="store_true", default=True)
    parser.add_argument("--sinkhorn_eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iters", type=int, default=100)
    # Dims
    parser.add_argument("--input_dim", type=int, default=getattr(cfg, "input_dim", 768))
    # Default to no learnable reduction by setting reduced_dim == input_dim
    parser.add_argument("--reduced_dim", type=int, default=getattr(cfg, "input_dim", 768))
    parser.add_argument("--latent_dim", type=int, default=getattr(cfg, "latent_dim", 256))
    parser.add_argument("--levels", type=int, default=getattr(cfg, "levels", 2))
    parser.add_argument("--heads", type=int, default=getattr(cfg, "heads", 2))
    parser.add_argument("--m", type=int, default=getattr(cfg, "m", 16))
    # Probabilistic head & schema (optional; defaults treat all features as continuous and disable head)
    parser.add_argument("--enable_prob_head", action="store_true", default=False)
    parser.add_argument("--num_features", type=int, default=0)
    parser.add_argument("--feature_types", type=int, nargs='*', default=None, help="Per-feature type codes: 0=cont,1=bin,2=cat")
    parser.add_argument("--categorical_feat_ids", type=int, nargs='*', default=None)
    parser.add_argument("--categorical_cardinalities", type=int, nargs='*', default=None)
    args = parser.parse_args()

    dm = DataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=args.smoke,
        smoke_batch_size=10,
        apply_A=False,
    )

    model = SetVAEOnlyPretrain(
        input_dim=args.input_dim,
        reduced_dim=args.reduced_dim,
        latent_dim=args.latent_dim,
        levels=args.levels,
        heads=args.heads,
        m=args.m,
        beta=getattr(cfg, "beta", 0.1),
        lr=args.lr,
        warmup_beta=bool(args.warmup_beta),
        max_beta=args.max_beta,
        beta_warmup_steps=args.beta_warmup_steps,
        free_bits=args.free_bits,
        use_kl_capacity=args.use_kl_capacity,
        capacity_per_dim_end=args.capacity_per_dim_end,
        capacity_warmup_steps=args.capacity_warmup_steps,
        use_flows=bool(args.use_flows),
        num_flows=int(args.num_flows),
        mmd_weight=float(args.mmd_weight),
        mmd_scales=tuple(args.mmd_scales) if isinstance(args.mmd_scales, list) else (args.mmd_scales,),
        recon_alpha=args.recon_alpha,
        recon_beta=args.recon_beta,
        recon_gamma=args.recon_gamma,
        recon_scale_calib=args.recon_scale_calib,
        recon_beta_var=args.recon_beta_var,
        use_sinkhorn=bool(args.use_sinkhorn),
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        kl_fairness_weight=float(args.kl_fairness_weight),
        kl_spread_tol=float(args.kl_spread_tol),
        kl_over_weight=float(args.kl_over_weight),
        var_stability_weight=float(args.var_stability_weight),
        per_dim_free_bits=float(args.per_dim_free_bits),
        enable_prob_head=bool(args.enable_prob_head),
        num_features=(int(args.num_features) if args.num_features is not None else 0),
        feature_types=(list(args.feature_types) if args.feature_types is not None else None),
        categorical_feat_ids=(list(args.categorical_feat_ids) if args.categorical_feat_ids is not None else None),
        categorical_cardinalities=(list(args.categorical_cardinalities) if args.categorical_cardinalities is not None else None),
    )

    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        try:
            state = pl.utilities.cloud_io.load(args.init_ckpt, map_location="cpu")  # type: ignore
        except Exception:
            import torch
            state = torch.load(args.init_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Initialized weights: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"Failed to initialize from init_ckpt: {e}")

    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "setvae")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    version_dir = os.path.dirname(log_dir) if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs") else log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="setvae_PT", dirpath=ckpt_dir)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[ckpt, early, lrmon],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
    )
    ckpt_path = args.resume_ckpt if args.resume_ckpt else None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


def _run_generative():
    parser = argparse.ArgumentParser(description="generative - PoE+GRU pretraining (merged B1/B2)")
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    parser.add_argument("--params_map_path", type=str, default=getattr(cfg, "params_map_path", ""))
    parser.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 4))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--freeze_epochs", type=int, default=1, help="#epochs to keep SetVAE encoder frozen before unfreezing")
    parser.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 4))
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run_name", type=str, default="PoE-GRU-PT")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--setvae_ckpt", type=str, default=None, help="Path to SetVAE checkpoint for initializing encoder (optional)")
    parser.add_argument("--log_every_n_steps", type=int, default=getattr(cfg, "log_every_n_steps", 50))
    parser.add_argument("--limit_val_batches", type=float, default=getattr(cfg, "limit_val_batches", 1.0))
    parser.add_argument("--val_check_interval", type=float, default=getattr(cfg, "val_check_interval", 0.1))
    # Optim & regularization
    parser.add_argument("--lr", type=float, default=getattr(cfg, "lr", 2e-4))
    parser.add_argument("--gradient_clip_val", type=float, default=getattr(cfg, "gradient_clip_val", 0.2))
    # KL & schedules
    parser.add_argument("--warmup_beta", action="store_true", default=getattr(cfg, "warmup_beta", True))
    parser.add_argument("--max_beta", type=float, default=getattr(cfg, "max_beta", 0.05))
    parser.add_argument("--beta_warmup_steps", type=int, default=getattr(cfg, "beta_warmup_steps", 8000))
    parser.add_argument("--free_bits", type=float, default=getattr(cfg, "free_bits", 0.03))
    parser.add_argument("--use_kl_capacity", action="store_true", default=getattr(cfg, "use_kl_capacity", True))
    parser.add_argument("--capacity_per_dim_end", type=float, default=getattr(cfg, "capacity_per_dim_end", 0.03))
    parser.add_argument("--capacity_warmup_steps", type=int, default=getattr(cfg, "capacity_warmup_steps", 20000))
    # Model dims
    parser.add_argument("--input_dim", type=int, default=getattr(cfg, "input_dim", 768))
    parser.add_argument("--reduced_dim", type=int, default=getattr(cfg, "reduced_dim", 256))
    parser.add_argument("--latent_dim", type=int, default=getattr(cfg, "latent_dim", 128))
    parser.add_argument("--levels", type=int, default=getattr(cfg, "levels", 2))
    parser.add_argument("--heads", type=int, default=getattr(cfg, "heads", 2))
    parser.add_argument("--m", type=int, default=getattr(cfg, "m", 16))
    # PoE extras
    parser.add_argument("--stale_dropout_p", type=float, default=getattr(cfg, "stale_dropout_p", 0.2))
    parser.add_argument("--set_mae_ratio", type=float, default=getattr(cfg, "set_mae_ratio", 0.0))
    parser.add_argument("--enable_next_change", action="store_true", default=True)
    parser.add_argument("--next_change_weight", type=float, default=0.3)
    parser.add_argument("--use_adaptive_poe", action="store_true", default=True)
    parser.add_argument("--poe_beta_min", type=float, default=0.1)
    parser.add_argument("--poe_beta_max", type=float, default=3.0)
    parser.add_argument("--freeze_set_encoder", action="store_true", default=True)
    parser.add_argument("--poe_mode", type=str, choices=["conditional", "naive"], default="conditional")
    # Stage A-style regularizers to mitigate dimension concentration
    parser.add_argument("--kl_fairness_weight", type=float, default=0.1)
    parser.add_argument("--kl_spread_tol", type=float, default=1.0)
    parser.add_argument("--kl_over_weight", type=float, default=1.0)
    parser.add_argument("--var_stability_weight", type=float, default=0.01)
    parser.add_argument("--per_dim_free_bits", type=float, default=0.002)
    # Recon weighting & partial unfreeze
    parser.add_argument("--recon_alpha", type=float, default=1.0)
    parser.add_argument("--recon_beta", type=float, default=6.0)
    parser.add_argument("--recon_gamma", type=float, default=2.5)
    parser.add_argument("--recon_scale_calib", type=float, default=1.0)
    parser.add_argument("--recon_beta_var", type=float, default=0.0)
    # Sinkhorn (方案A)
    parser.add_argument("--use_sinkhorn", action="store_true", default=True)
    parser.add_argument("--sinkhorn_eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iters", type=int, default=100)
    parser.add_argument("--partial_unfreeze", action="store_true", default=False)
    parser.add_argument("--unfreeze_dim_reducer", action="store_true", default=False)
    # Joint classification (minimal integration of Stage C)
    parser.add_argument("--enable_cls", action="store_true", default=False, help="Enable joint sequence classification head in Stage B")
    parser.add_argument("--cls_loss_weight", type=float, default=0.6, help="Weight for classification loss in joint objective")
    parser.add_argument("--cls_dropout", type=float, default=0.2)
    parser.add_argument("--label_csv", type=str, default=None, help="Label CSV (required when --enable_cls)")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable class-balanced sampling when --enable_cls")
    parser.add_argument("--freeze_encoder", action="store_true", default=True, help="Freeze SetVAE encoder during training (recommended baseline)")
    parser.add_argument("--cls_lr", type=float, default=2e-4, help="Learning rate for classifier head and transformer")
    parser.add_argument("--enc_lr", type=float, default=1e-5, help="Learning rate for (optionally) unfrozen encoder")

    args = parser.parse_args()

    if args.enable_cls:
        if not args.label_csv:
            raise ValueError("--label_csv is required when --enable_cls")
        dm = MortalityDataModule(
            saved_dir=args.data_dir,
            label_csv=args.label_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            smoke=args.smoke,
            smoke_batch_size=max(2, args.batch_size),
            use_weighted_sampler=(not args.no_weighted_sampler),
        )
        # Setup early to obtain pos_weight for model init
        try:
            dm.setup()
            pos_weight = getattr(dm, "pos_weight", None)
            # Avoid double-balancing: if using WeightedRandomSampler, drop BCE pos_weight
            if getattr(dm, "use_weighted_sampler", False):
                pos_weight = None
        except Exception:
            pos_weight = None
    else:
        dm = DataModule(
            saved_dir=args.data_dir,
            params_map_path=args.params_map_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            smoke=args.smoke,
            smoke_batch_size=10,
            apply_A=False,
        )
        pos_weight = None

    model = PoESeqSetVAEPretrain(
        input_dim=args.input_dim,
        reduced_dim=args.reduced_dim,
        latent_dim=args.latent_dim,
        levels=args.levels,
        heads=args.heads,
        m=args.m,
        beta=getattr(cfg, "beta", 0.1),
        lr=args.lr,
        ff_dim=getattr(cfg, "ff_dim", 512),
        transformer_heads=getattr(cfg, "transformer_heads", 8),
        transformer_layers=getattr(cfg, "transformer_layers", 4),
        transformer_dropout=getattr(cfg, "transformer_dropout", 0.15),
        warmup_beta=bool(args.warmup_beta),
        max_beta=args.max_beta,
        beta_warmup_steps=args.beta_warmup_steps,
        free_bits=args.free_bits,
        use_kl_capacity=args.use_kl_capacity,
        capacity_per_dim_end=args.capacity_per_dim_end,
        capacity_warmup_steps=args.capacity_warmup_steps,
        stale_dropout_p=args.stale_dropout_p,
        set_mae_ratio=args.set_mae_ratio,
        enable_next_change=args.enable_next_change,
        next_change_weight=args.next_change_weight,
        use_adaptive_poe=args.use_adaptive_poe,
        poe_beta_min=args.poe_beta_min,
        poe_beta_max=args.poe_beta_max,
        freeze_set_encoder=args.freeze_set_encoder,
        poe_mode=args.poe_mode,
        recon_alpha=args.recon_alpha,
        recon_beta=args.recon_beta,
        recon_gamma=args.recon_gamma,
        recon_scale_calib=args.recon_scale_calib,
        recon_beta_var=args.recon_beta_var,
        use_sinkhorn=bool(args.use_sinkhorn),
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        partial_unfreeze=False,
        unfreeze_dim_reducer=False,
        kl_fairness_weight=float(args.kl_fairness_weight),
        kl_spread_tol=float(args.kl_spread_tol),
        kl_over_weight=float(args.kl_over_weight),
        var_stability_weight=float(args.var_stability_weight),
        per_dim_free_bits=float(args.per_dim_free_bits),
        enable_cls=bool(args.enable_cls),
        cls_loss_weight=float(args.cls_loss_weight),
        cls_dropout=float(args.cls_dropout),
        pos_weight=pos_weight,
    )

    # If --setvae_ckpt provided, merge其 SetVAE 权重到初始化（或模型状态）
    fused_loaded = False
    base_state: dict | None = None
    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        base_state = _load_state_dict_flex(args.init_ckpt)
    elif args.init_ckpt:
        print(f"[WARN] init_ckpt not found: {args.init_ckpt}")

    if args.setvae_ckpt is not None and os.path.isfile(args.setvae_ckpt):
        try:
            state_A = _load_state_dict_flex(args.setvae_ckpt)
            target_model_state = model.state_dict()
            setvae_from_A = _remap_setvae_to_target(state_A, target_model_state, drop_flows=True)

            # --- Compatibility advisory: detect SetVAE reduced_dim vs current generative config ---
            try:
                # Infer source SetVAE prefix in Stage A state
                src_prefix = _detect_setvae_prefix(list(state_A.keys()))
                # Heuristic: if Stage A used a dim_reducer, the weight exists under '<prefix>dim_reducer.weight'
                rd_key = f"{src_prefix}dim_reducer.weight" if src_prefix else ""
                has_rd_A = bool(rd_key and rd_key in state_A)
                rd_A = int(state_A[rd_key].shape[0]) if has_rd_A else None
                # Effective B reduced_dim: treat equal to input_dim (or non-positive) as no reducer
                rd_B = None
                try:
                    if args.reduced_dim is not None:
                        rd_val = int(args.reduced_dim)
                        if rd_val > 0 and rd_val != int(args.input_dim):
                            rd_B = rd_val
                except Exception:
                    rd_B = None
                # Warn if mismatch; this impacts which SetVAE layers can be loaded by shape
                if (has_rd_A and (rd_B is None or rd_A != rd_B)) or ((not has_rd_A) and (rd_B is not None)):
                    msg = (
                        f"[WARN] SetVAE dim_reducer mismatch: "
                        f"A {'has' if has_rd_A else 'no'} reducer{f' ({rd_A})' if rd_A is not None else ''}, "
                        f"B configured reduced_dim={'none' if rd_B is None else rd_B}.\n"
                        f"       Some SetVAE layers (e.g., dim_reducer/embed/out) may not load due to shape differences.\n"
                        f"       For maximum reuse, consider running with --reduced_dim="
                        f"{int(args.input_dim) if not has_rd_A else rd_A}."
                    )
                    print(msg)
            except Exception:
                # Best-effort advisory only; ignore any detection errors
                pass

            # Choose base to fuse into: provided init_ckpt if available, else current model state
            if base_state is None:
                base_state = target_model_state

            # Remove existing target setvae keys, then overlay from A
            dst_prefix = "set_encoder."
            if not any(k.startswith(dst_prefix) for k in target_model_state.keys()):
                if any(k.startswith("setvae.setvae.") for k in target_model_state.keys()):
                    dst_prefix = "setvae.setvae."
                elif any(k.startswith("setvae.") for k in target_model_state.keys()):
                    dst_prefix = "setvae."

            fused = {k: v for k, v in base_state.items() if not k.startswith(dst_prefix)}
            fused.update(setvae_from_A)

            incompatible = model.load_state_dict(fused, strict=False)
            print(
                f"Initialized with fused SetVAE->generative: "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}, "
                f"A_setvae_applied={len(setvae_from_A)}"
            )
            fused_loaded = True
        except Exception as e:
            print(f"[WARN] Failed to merge SetVAE into generative init: {e}")

    if not fused_loaded:
        # Fallback to original behavior: load init_ckpt directly if present
        if base_state is not None:
            try:
                incompatible = model.load_state_dict(base_state, strict=False)
                print(
                    f"Initialized from init_ckpt: missing={len(incompatible.missing_keys)}, "
                    f"unexpected={len(incompatible.unexpected_keys)}"
                )
            except Exception as e:
                print(f"Failed to initialize from init_ckpt: {e}")

    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "generative", "gru")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    version_dir = os.path.dirname(log_dir) if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs") else log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Always keep a checkpoint for lowest validation loss
    val_loss_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="gru_PT",
    )
    # When joint classification is enabled, also track classification metrics robust to imbalance
    callbacks = [LearningRateMonitor(logging_interval="step"), val_loss_ckpt]
    if bool(args.enable_cls):
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    save_top_k=1,
                    monitor="val_auprc",
                    mode="max",
                    filename="poe_cls_auprc",
                ),
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    save_top_k=1,
                    monitor="val_auroc",
                    mode="max",
                    filename="poe_cls_auroc",
                ),
                # Removed best-accuracy-based checkpoint; rely on AUPRC/AUROC only
            ]
        )
        # Early stop on validation classification loss which is always logged
        # (AUPRC may not be available during early validation phases)
        early = EarlyStopping(monitor="val_cls", mode="min", patience=8)
    else:
        early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    callbacks.append(early)
    # Enable manual unfreeze after --freeze_epochs by setting an initial freeze and relying on model's on_train_epoch_end
    if args.freeze_set_encoder:
        for p in model.set_encoder.parameters():
            p.requires_grad = False
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
    )
    ckpt_path = args.resume_ckpt if args.resume_ckpt else None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


def _run_discriminative():
    parser = argparse.ArgumentParser(description="discriminative - Mortality classifier (transformer over SetVAE or PoE-based)")
    parser.add_argument("--mode", type=str, choices=["poe", "transformer"], default="poe")
    parser.add_argument("--checkpoint", required=False, type=str, help="PoE checkpoint (.ckpt) when --mode=poe")
    parser.add_argument("--setvae_ckpt", required=False, type=str, help="SetVAE checkpoint (.ckpt) required when --mode=transformer")
    parser.add_argument("--label_csv", required=True, type=str)
    parser.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 2))
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable class-balanced sampler (default enabled)")
    # Validate once mid-epoch for transformer/classifier training as well
    parser.add_argument("--val_check_interval", type=float, default=0.5)
    # Optional partial unfreeze of pretrained encoders
    parser.add_argument("--unfreeze_after_epochs", type=int, default=1, help="Epoch index (1-based) after which to unfreeze pretrained encoder; 0 to keep frozen")
    parser.add_argument(
        "--unfreeze_scope",
        type=str,
        choices=["none", "light", "full"],
        default="light",
        help="What to unfreeze: none=keep frozen; light=dim_reducer+embed; full=entire set encoder",
    )
    # Transformer hyperparams (baseline)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "poe":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode=poe")
        # Load PoE backbone (frozen)
        state = _load_state_dict(args.checkpoint)
        poe = _build_poe_from_state(state)
    else:
        if not args.setvae_ckpt:
            raise ValueError("--setvae_ckpt is required when --mode=transformer")
        setvae_state = _load_state_dict(args.setvae_ckpt)
        setvae = _build_setvae_from_state(setvae_state)

    # Data
    dm = MortalityDataModule(
        saved_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=args.smoke,
        smoke_batch_size=max(2, args.batch_size),
        use_weighted_sampler=(not args.no_weighted_sampler),
    )
    dm.setup()

    if args.mode == "poe":
        model = MortalityClassifier(
            poe_model=poe,
            latent_dim=getattr(cfg, "latent_dim", 128),
            mu_proj_dim=64,
            scalar_proj_dim=16,
            gru_hidden=128,
            gru_layers=2,
            dropout=args.dropout,
            lr=args.cls_lr,
            # Avoid double-balancing: if using WeightedRandomSampler, drop BCE pos_weight
            pos_weight=(None if getattr(dm, "use_weighted_sampler", False) else getattr(dm, "pos_weight", None)),
            unfreeze_after_epochs=(0 if args.freeze_encoder else max(0, int(args.unfreeze_after_epochs))),
            unfreeze_scope=("none" if args.freeze_encoder else str(args.unfreeze_scope)),
        )
    else:
        model = TransformerSetVAEClassifier(
            setvae_model=setvae,
            latent_dim=int(getattr(setvae, "latent_dim", getattr(cfg, "latent_dim", 128))),
            mu_proj_dim=64,
            prec_proj_dim=32,
            scalar_proj_dim=16,
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            num_layers=int(args.num_layers),
            dropout=args.dropout,
            lr=args.cls_lr,
            # Avoid double-balancing: if using WeightedRandomSampler, drop BCE pos_weight
            pos_weight=(None if getattr(dm, "use_weighted_sampler", False) else getattr(dm, "pos_weight", None)),
            unfreeze_after_epochs=(0 if args.freeze_encoder else max(0, int(args.unfreeze_after_epochs))),
            unfreeze_scope=("none" if args.freeze_encoder else str(args.unfreeze_scope)),
        )

    out_root = args.output_dir if args.output_dir else "./output"
    # Discriminative branch outputs
    if args.mode == "poe":
        project_dir = os.path.join(out_root, "discriminative", "poe")
    else:
        project_dir = os.path.join(out_root, "discriminative", "transformer")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    version_dir = os.path.dirname(log_dir) if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs") else log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Track multiple metrics and save best by AUPRC; also keep AUROC-best
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="val_auprc", mode="max", filename="mortality_cls"),
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="val_auroc", mode="max", filename="mortality_cls_auroc"),
        # Removed best-accuracy-based checkpoint; rely on AUPRC/AUROC only
        EarlyStopping(monitor="val_auprc", mode="max", patience=8),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    print("\nEvaluating best checkpoint on test split...")
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
