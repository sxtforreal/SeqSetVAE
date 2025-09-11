"""
python -u seqsetvae_poe/_poe_PT.py \
  --data_dir /path/to/SeqSetVAE \
  --params_map_path /path/to/stats.csv \
  --batch_size 8 --max_epochs 50 --num_workers 1 \
  --precision 16-mixed --lr 2e-4 \
  --warmup_beta --max_beta 0.05 --beta_warmup_steps 8000 --free_bits 0.03 \
  --use_kl_capacity --capacity_per_dim_end 0.03 --capacity_warmup_steps 20000 \
  --limit_val_batches 0.25 --val_check_interval 0.2 \
  --run_name PoE-GRU-PT
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
from model import PoESeqSetVAEPretrain


def main():
    parser = argparse.ArgumentParser(
        description="PoE pretraining with GRU dynamics over LVCF-expanded sets"
    )
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=getattr(config, "data_dir", ""))
    parser.add_argument(
        "--params_map_path", type=str, default=getattr(config, "params_map_path", "")
    )
    parser.add_argument("--batch_size", type=int, default=getattr(config, "batch_size", 4))
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=getattr(config, "num_workers", 4))
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--smoke", action="store_true", help="Build one train batch for smoke test")
    parser.add_argument(
        "--run_name", type=str, default="PoE-GRU-PT", help="TensorBoard run name subfolder"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Root out_dir for experiments (saves to out_dir/setvae-PT/version_X/{logs,checkpoints,eval})"
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
        "--log_every_n_steps", type=int, default=getattr(config, "log_every_n_steps", 50)
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

    # Optim & regularization
    parser.add_argument("--lr", type=float, default=getattr(config, "lr", 2e-4))
    parser.add_argument(
        "--gradient_clip_val", type=float, default=getattr(config, "gradient_clip_val", 0.2)
    )

    # Anti posterior-collapse / KL capacity
    parser.add_argument("--warmup_beta", action="store_true", default=getattr(config, "warmup_beta", True))
    parser.add_argument("--max_beta", type=float, default=getattr(config, "max_beta", 0.05))
    parser.add_argument("--beta_warmup_steps", type=int, default=getattr(config, "beta_warmup_steps", 8000))
    parser.add_argument("--free_bits", type=float, default=getattr(config, "free_bits", 0.03))
    parser.add_argument("--use_kl_capacity", action="store_true", default=getattr(config, "use_kl_capacity", True))
    parser.add_argument(
        "--capacity_per_dim_end", type=float, default=getattr(config, "capacity_per_dim_end", 0.03)
    )
    parser.add_argument(
        "--capacity_warmup_steps", type=int, default=getattr(config, "capacity_warmup_steps", 20000)
    )

    # Model dims
    parser.add_argument("--input_dim", type=int, default=getattr(config, "input_dim", 768))
    parser.add_argument("--reduced_dim", type=int, default=getattr(config, "reduced_dim", 256))
    parser.add_argument("--latent_dim", type=int, default=getattr(config, "latent_dim", 128))
    parser.add_argument("--levels", type=int, default=getattr(config, "levels", 2))
    parser.add_argument("--heads", type=int, default=getattr(config, "heads", 2))
    parser.add_argument("--m", type=int, default=getattr(config, "m", 16))

    # Legacy transformer args (kept for constructor compatibility; unused in GRU version)
    parser.add_argument("--ff_dim", type=int, default=getattr(config, "ff_dim", 512))
    parser.add_argument("--transformer_heads", type=int, default=getattr(config, "transformer_heads", 8))
    parser.add_argument("--transformer_layers", type=int, default=getattr(config, "transformer_layers", 4))
    parser.add_argument(
        "--transformer_dropout", type=float, default=getattr(config, "transformer_dropout", 0.15)
    )

    # PoE extras
    parser.add_argument("--stale_dropout_p", type=float, default=getattr(config, "stale_dropout_p", 0.2))
    parser.add_argument("--set_mae_ratio", type=float, default=getattr(config, "set_mae_ratio", 0.0))
    parser.add_argument("--enable_next_change", action="store_true", default=True)
    parser.add_argument("--next_change_weight", type=float, default=0.3)

    # Adaptive PoE gate
    parser.add_argument("--use_adaptive_poe", action="store_true", default=True)
    parser.add_argument("--poe_beta_min", type=float, default=0.1)
    parser.add_argument("--poe_beta_max", type=float, default=3.0)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--freeze_set_encoder", dest="freeze_set_encoder", action="store_true")
    grp.add_argument("--no_freeze_set_encoder", dest="freeze_set_encoder", action="store_false")
    parser.set_defaults(freeze_set_encoder=True)

    # PoE mode
    parser.add_argument("--poe_mode", type=str, choices=["conditional", "naive"], default="conditional")

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
        beta=getattr(config, "beta", 0.1),
        lr=args.lr,
        ff_dim=args.ff_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_dropout=args.transformer_dropout,
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
    )

    # Optional: initialize weights from checkpoint (weights only)
    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        try:
            state = torch.load(args.init_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"🔁 Initialized model: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"⚠️  Failed to initialize from init_ckpt: {e}")

    # Loggers & callbacks with standardized directory layout:
    # out_dir/setvae-PT/version_X/{logs,checkpoints,eval}
    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "setvae-PT")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        # Fallback if sub_dir is unsupported
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    # Determine the version directory regardless of whether sub_dir is supported
    log_dir = getattr(logger, "log_dir", project_dir)
    if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs"):
        version_dir = os.path.dirname(log_dir)
    else:
        version_dir = log_dir
    # Ensure required subfolders exist
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    eval_dir = os.path.join(version_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")
    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="poe_GRU_PT", dirpath=ckpt_dir)

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


if __name__ == "__main__":
    main()

