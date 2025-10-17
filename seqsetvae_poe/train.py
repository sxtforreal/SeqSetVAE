#!/usr/bin/env python3
"""
Unified train entrypoint with --stage {A,B,C}.

- Stage A: SetVAE-only pretraining (internals from _setvae_PT)
- Stage B: Dynamics + conditional PoE pretraining (internals from _poe_PT)
- Stage C: Downstream classifier (internals from classifier)

All other arguments are forwarded to the underlying stage logic.
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
from model import SetVAEOnlyPretrain, PoESeqSetVAEPretrain, MortalityClassifier, _load_state_dict, _build_poe_from_state  # type: ignore


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


def _remap_setvae_from_A_to_target(
    state_stageA: dict,
    target_model_state: dict,
    drop_flows: bool = True,
) -> dict:
    """Map Stage A SetVAE weights to target model's SetVAE prefix and filter by matching shape.

    Returns a dict containing only remapped SetVAE weights compatible with the target model.
    """
    src_prefix = _detect_setvae_prefix(list(state_stageA.keys()))
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
    for k, v in state_stageA.items():
        if not k.startswith(src_prefix):
            continue
        if drop_flows and ".flows." in k:
            continue
        target_key = dst_prefix + k[len(src_prefix):]
        if target_key in target_model_state and target_model_state[target_key].shape == v.shape:
            remapped[target_key] = v
    return remapped


def _strip_stage_from_argv():
    # remove --stage and its value from sys.argv to avoid unknown-arg errors downstream
    argv = sys.argv
    out = [argv[0]]
    skip = 0
    for i, tok in enumerate(argv[1:], start=1):
        if skip:
            skip -= 1
            continue
        if tok == "--stage":
            skip = 1  # skip its value
            continue
        if tok.startswith("--stage="):
            continue
        out.append(tok)
    sys.argv = out


def main():
    ap = argparse.ArgumentParser(description="Unified train with --stage {A,B,C}", add_help=False)
    ap.add_argument("--stage", type=str, choices=["A", "B", "C"], required=True)
    # parse only known stage and then strip
    args, _ = ap.parse_known_args()
    _strip_stage_from_argv()
    if args.stage == "A":
        return _run_stage_a()
    if args.stage == "B":
        return _run_stage_b()
    return _run_stage_c()


def _run_stage_a():
    parser = argparse.ArgumentParser(description="Stage A - SetVAE pretraining")
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
    parser.add_argument("--val_check_interval", type=float, default=getattr(cfg, "val_check_interval", 0.1))
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
    project_dir = os.path.join(out_root, "Stage_A")
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


def _run_stage_b():
    parser = argparse.ArgumentParser(description="Stage B - PoE+GRU pretraining")
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    parser.add_argument("--params_map_path", type=str, default=getattr(cfg, "params_map_path", ""))
    parser.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 4))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 4))
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run_name", type=str, default="PoE-GRU-PT")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--stageA_ckpt", type=str, default=None, help="Path to Stage A SetVAE checkpoint for merging into B1 init")
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
        partial_unfreeze=bool(args.partial_unfreeze),
        unfreeze_dim_reducer=bool(args.unfreeze_dim_reducer),
    )

    # Default behavior: if --stageA_ckpt provided, merge its SetVAE weights into initializer (or model state)
    fused_loaded = False
    base_state: dict | None = None
    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        base_state = _load_state_dict_flex(args.init_ckpt)
    elif args.init_ckpt:
        print(f"[WARN] init_ckpt not found: {args.init_ckpt}")

    if args.stageA_ckpt is not None and os.path.isfile(args.stageA_ckpt):
        try:
            state_A = _load_state_dict_flex(args.stageA_ckpt)
            target_model_state = model.state_dict()
            setvae_from_A = _remap_setvae_from_A_to_target(state_A, target_model_state, drop_flows=True)

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
                f"Initialized with fused StageA->B1: "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}, "
                f"A_setvae_applied={len(setvae_from_A)}"
            )
            fused_loaded = True
        except Exception as e:
            print(f"[WARN] Failed to merge Stage A SetVAE into B1 init: {e}")

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
    stage_name = "Stage_B2" if bool(args.partial_unfreeze) else "Stage_B1"
    project_dir = os.path.join(out_root, stage_name)
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    version_dir = os.path.dirname(log_dir) if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs") else log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="poe_GRU_PT", dirpath=ckpt_dir)
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


def _run_stage_c():
    parser = argparse.ArgumentParser(description="Stage C - Mortality classifier on PoE features")
    parser.add_argument("--checkpoint", required=True, type=str)
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
    args = parser.parse_args()

    # Load PoE backbone (frozen)
    state = _load_state_dict(args.checkpoint)
    poe = _build_poe_from_state(state)

    # Data
    dm = MortalityDataModule(
        saved_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=args.smoke,
        smoke_batch_size=max(2, args.batch_size),
    )
    dm.setup()

    model = MortalityClassifier(
        poe_model=poe,
        latent_dim=getattr(cfg, "latent_dim", 128),
        mu_proj_dim=64,
        logvar_proj_dim=32,
        scalar_proj_dim=16,
        gru_hidden=128,
        gru_layers=2,
        dropout=args.dropout,
        lr=args.lr,
        pos_weight=getattr(dm, "pos_weight", None),
    )

    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "Stage_C")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    version_dir = os.path.dirname(log_dir) if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs") else log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="val_auprc", mode="max", filename="mortality_cls"),
        EarlyStopping(monitor="val_auprc", mode="max", patience=5),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    print("\nEvaluating best checkpoint on test split...")
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
