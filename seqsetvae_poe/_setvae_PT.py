""" 
python /home/sunx/data/aiiih/projects/sunx/projects/SSV/main/_setvae_PT.py \
  --data_dir /home/sunx/data/aiiih/data/mimic/processed/SeqSetVAE \
  --output_dir /home/sunx/data/aiiih/projects/sunx/projects/SSV/output/setvae-PT \
  --batch_size 10 \
  --max_epochs 50 \
  --num_workers 1 \
  --precision 16-mixed \
  --lr 3e-4 \
  --warmup_beta --max_beta 0.2 --beta_warmup_steps 8000 --free_bits 0.05 \
  --p_stale 0.5 --p_live 0.05 \
  --set_mae_ratio 0.15 --small_set_mask_prob 0.4 --small_set_threshold 5 --max_masks_per_set 2 \
  --val_noise_std 0.07 --dir_noise_std 0.01 \
  --train_decoder_noise_std 0.3 --eval_decoder_noise_std 0.05 \
  --gradient_clip_val 0.2 \
  --run_name SetVAE-Only-PT
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
        default=None,
        help="Experiment output dir (default ./outputs/<run_name>)",
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
    parser.add_argument("--use_kl_capacity", action="store_true", default=getattr(config, "use_kl_capacity", True))
    parser.add_argument("--capacity_per_dim_end", type=float, default=getattr(config, "capacity_per_dim_end", 0.03))
    parser.add_argument("--capacity_warmup_steps", type=int, default=getattr(config, "capacity_warmup_steps", 20000))

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
    )

    # Optional: initialize weights from checkpoint (weights only)
    if args.init_ckpt is not None and os.path.isfile(args.init_ckpt):
        try:
            state = torch.load(args.init_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"🔁 Initialized model from weights: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"⚠️  Failed to initialize from init_ckpt: {e}")

    ckpt = ModelCheckpoint(
        save_top_k=1, monitor="val_loss", mode="min", filename="setvae_PT"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")

    # Prepare experiment output directory
    experiment_output_dir = (
        args.output_dir if args.output_dir else os.path.join("./outputs", args.run_name)
    )
    os.makedirs(experiment_output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=experiment_output_dir, name="")

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
