""" 
python -u /home/sunx/data/aiiih/projects/sunx/projects/SSV/main/_setvae_PT.py \
  --data_dir /home/sunx/data/aiiih/data/mimic/processed/SeqSetVAE \
  --params_map_path /home/sunx/data/aiiih/data/mimic/processed/stats.csv \
  --batch_size 10 --max_epochs 50 --num_workers 1 \
  --precision 16-mixed --lr 3e-4 \
  --warmup_beta --max_beta 0.05 --beta_warmup_steps 8000 --free_bits 0.03 \
  --use_kl_capacity --capacity_per_dim_end 0.03 --capacity_warmup_steps 20000 \
  --limit_val_batches 0.25 --val_check_interval 0.2 \
  --init_ckpt /home/sunx/data/aiiih/projects/sunx/projects/SSV/output/setvae-PT/version_0/checkpoints/setvae_PT.ckpt \
  --run_name SetVAE-PT-capacity-init
"""

import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger


import config
from dataset import DataModule
from model import SetVAEOnlyPretrain


def main():
    parser = argparse.ArgumentParser(
        description="SetVAE-only pretraining over LVCF-expanded sets"
    )
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=getattr(config, "data_dir", ""))
    parser.add_argument(
        "--params_map_path", type=str, default=getattr(config, "params_map_path", "")
    )
    parser.add_argument(
        "--batch_size", type=int, default=getattr(config, "batch_size", 4)
    )
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument(
        "--num_workers", type=int, default=getattr(config, "num_workers", 4)
    )
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument(
        "--smoke", action="store_true", help="Build one train batch for smoke test"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="SetVAE-Only-PT",
        help="TensorBoard run name subfolder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root out_dir for experiments (saves to out_dir/setvae-PT/version_X/{logs,checkpoints,eval})",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume training from a Lightning checkpoint (.ckpt). Restores optimizer/scheduler state",
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Initialize weights from a checkpoint (weights only). Does NOT restore optimizer/scheduler",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=getattr(config, "log_every_n_steps", 50),
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=getattr(config, "limit_val_batches", 1.0),
        help="Fraction/number of validation batches to run each check (Lightning's limit_val_batches)",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=getattr(config, "val_check_interval", 0.1),
        help="How often to run validation within an epoch (fraction of train epoch)",
    )

    # InfoVAE / Flows
    parser.add_argument("--use_flows", action="store_true", default=False, help="Enable planar flows on last-layer latent")
    parser.add_argument("--num_flows", type=int, default=0, help="Number of planar flow layers")
    parser.add_argument("--mmd_weight", type=float, default=0.0, help="Weight for InfoVAE MMD regularizer")
    parser.add_argument("--mmd_scales", type=float, nargs='*', default=[1.0, 2.0, 4.0, 8.0], help="RBF MMD kernel scales")

    # KL fairness & stability controls
    parser.add_argument("--kl_fairness_weight", type=float, default=0.1, help="Weight for KL fairness penalty (promote uniform per-dim KL)")
    parser.add_argument("--kl_spread_tol", type=float, default=1.0, help="Tolerance for over-allocation relative to uniform share")
    parser.add_argument("--kl_over_weight", type=float, default=1.0, help="Weight for over-allocation L2 term")
    parser.add_argument("--var_stability_weight", type=float, default=0.01, help="Weight for posterior variance deviation penalty")
    parser.add_argument("--per_dim_free_bits", type=float, default=0.002, help="Per-dimension free bits (nats)")
    parser.add_argument("--posterior_logvar_min", type=float, default=-2.5, help="Clamp min for posterior log-variance")
    parser.add_argument("--posterior_logvar_max", type=float, default=2.5, help="Clamp max for posterior log-variance")
    parser.add_argument("--enable_posterior_std_augmentation", action="store_true", default=False)
    parser.add_argument("--posterior_std_aug_sigma", type=float, default=0.0)

    # Auto-tune & monitoring
    parser.add_argument("--auto_tune_kl", action="store_true", default=True, help="Auto-increase capacity/beta if KL_last below target")
    parser.add_argument("--kl_target_nats", type=float, default=10.0)
    parser.add_argument("--kl_patience_epochs", type=int, default=2)
    parser.add_argument("--beta_step", type=float, default=0.2)
    parser.add_argument("--capacity_per_dim_step", type=float, default=0.02)
    parser.add_argument("--max_beta_ceiling", type=float, default=2.0)
    parser.add_argument("--capacity_per_dim_max", type=float, default=0.20)
    parser.add_argument("--active_ratio_warn_threshold", type=float, default=0.5)

    # Optim & regularization
    parser.add_argument("--lr", type=float, default=getattr(config, "lr", 3e-4))
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=getattr(config, "gradient_clip_val", 0.2),
    )

    # Anti posterior-collapse
    parser.add_argument(
        "--warmup_beta",
        action="store_true",
        default=getattr(config, "warmup_beta", True),
    )
    parser.add_argument("--max_beta", type=float, default=0.2)
    parser.add_argument(
        "--beta_warmup_steps",
        type=int,
        default=getattr(config, "beta_warmup_steps", 8000),
    )
    parser.add_argument(
        "--free_bits", type=float, default=0.05, help="Per-dim free bits (nats) for KL"
    )
    # KL capacity schedule
    parser.add_argument(
        "--use_kl_capacity",
        action="store_true",
        default=getattr(config, "use_kl_capacity", True),
    )
    parser.add_argument(
        "--capacity_per_dim_end",
        type=float,
        default=getattr(config, "capacity_per_dim_end", 0.03),
    )
    parser.add_argument(
        "--capacity_warmup_steps",
        type=int,
        default=getattr(config, "capacity_warmup_steps", 20000),
    )

    # Perturbations
    parser.add_argument(
        "--p_stale", type=float, default=0.5, help="Dropout prob for carried values"
    )
    parser.add_argument(
        "--p_live", type=float, default=0.05, help="Dropout prob for non-carried values"
    )
    parser.add_argument(
        "--set_mae_ratio",
        type=float,
        default=0.15,
        help="Mask ratio for larger sets (> small_set_threshold)",
    )
    parser.add_argument(
        "--small_set_mask_prob",
        type=float,
        default=0.4,
        help="Prob to mask exactly 1 token if N<=threshold",
    )
    parser.add_argument("--small_set_threshold", type=int, default=5)
    parser.add_argument("--max_masks_per_set", type=int, default=2)
    parser.add_argument("--val_noise_std", type=float, default=0.07)
    parser.add_argument("--dir_noise_std", type=float, default=0.01)
    parser.add_argument("--train_decoder_noise_std", type=float, default=0.3)
    parser.add_argument("--eval_decoder_noise_std", type=float, default=0.05)

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
        input_dim=getattr(config, "input_dim", 768),
        reduced_dim=getattr(config, "reduced_dim", 256),
        latent_dim=getattr(config, "latent_dim", 256),
        levels=getattr(config, "levels", 2),
        heads=getattr(config, "heads", 2),
        m=getattr(config, "m", 16),
        beta=getattr(config, "beta", 0.1),
        lr=args.lr,
        warmup_beta=bool(args.warmup_beta),
        max_beta=args.max_beta,
        beta_warmup_steps=args.beta_warmup_steps,
        free_bits=args.free_bits,
        use_kl_capacity=args.use_kl_capacity,
        capacity_per_dim_end=args.capacity_per_dim_end,
        capacity_warmup_steps=args.capacity_warmup_steps,
        p_stale=args.p_stale,
        p_live=args.p_live,
        set_mae_ratio=args.set_mae_ratio,
        small_set_mask_prob=args.small_set_mask_prob,
        small_set_threshold=args.small_set_threshold,
        max_masks_per_set=args.max_masks_per_set,
        val_noise_std=args.val_noise_std,
        dir_noise_std=args.dir_noise_std,
        train_decoder_noise_std=args.train_decoder_noise_std,
        eval_decoder_noise_std=args.eval_decoder_noise_std,
        use_flows=bool(args.use_flows),
        num_flows=int(args.num_flows),
        mmd_weight=float(args.mmd_weight),
        mmd_scales=tuple(args.mmd_scales) if isinstance(args.mmd_scales, list) else (args.mmd_scales,),
        # Fairness & stability
        kl_fairness_weight=float(args.kl_fairness_weight),
        kl_spread_tol=float(args.kl_spread_tol),
        kl_over_weight=float(args.kl_over_weight),
        var_stability_weight=float(args.var_stability_weight),
        per_dim_free_bits=float(args.per_dim_free_bits),
        posterior_logvar_min=float(args.posterior_logvar_min),
        posterior_logvar_max=float(args.posterior_logvar_max),
        enable_posterior_std_augmentation=bool(args.enable_posterior_std_augmentation),
        posterior_std_aug_sigma=float(args.posterior_std_aug_sigma),
        auto_tune_kl=bool(args.auto_tune_kl),
        kl_target_nats=float(args.kl_target_nats),
        kl_patience_epochs=int(args.kl_patience_epochs),
        beta_step=float(args.beta_step),
        capacity_per_dim_step=float(args.capacity_per_dim_step),
        max_beta_ceiling=float(args.max_beta_ceiling),
        capacity_per_dim_max=float(args.capacity_per_dim_max),
        active_ratio_warn_threshold=float(args.active_ratio_warn_threshold),
    )

    # Optional: initialize weights from checkpoint (weights only)
    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        try:
            state = torch.load(args.init_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"🔁 Initialized model from weights: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as e:
            print(f"⚠️  Failed to initialize from init_ckpt: {e}")

    # Loggers & callbacks with standardized directory layout:
    # out_dir/setvae-PT/version_X/{logs,checkpoints,eval}
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")

    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "setvae-PT")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs"):
        version_dir = os.path.dirname(log_dir)
    else:
        version_dir = log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    eval_dir = os.path.join(version_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="setvae_PT", dirpath=ckpt_dir)

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
    # Resume if provided; if both resume and init are passed, resume takes precedence
    ckpt_path = args.resume_ckpt if args.resume_ckpt else None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
