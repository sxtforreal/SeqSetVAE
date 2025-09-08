#!/usr/bin/env python3
import os
import sys
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Make script runnable both as package and as a standalone file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from . import config  # type: ignore
    from .dataset import DataModule  # type: ignore
    from .model import PoESeqSetVAEPretrain  # type: ignore
except Exception:
    import config
    from dataset import DataModule
    from model import PoESeqSetVAEPretrain

# Optional posterior collapse monitor (lives at project root)
try:
    from posterior_collapse_detector import PosteriorMetricsMonitor  # type: ignore
except Exception:
    PosteriorMetricsMonitor = None  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAE-PoE Pretraining (A+B+PoE+C ready)")
    # Data & loader
    parser.add_argument("--data_dir", type=str, default=config.data_dir)
    parser.add_argument("--params_map_path", type=str, default=config.params_map_path)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--smoke", action="store_true", help="Build one train batch of 10 random samples for smoke test")
    parser.add_argument("--run_name", type=str, default=config.name, help="TensorBoard run name subfolder")
    parser.add_argument("--output_dir", type=str, default=None, help="Experiment output directory. If not set, defaults to ./outputs/<run_name>")
    parser.add_argument("--log_every_n_steps", type=int, default=getattr(config, "log_every_n_steps", 50), help="Frequency (in steps) for logging")

    # Task toggles
    parser.add_argument("--enable_A", action="store_true", help="Apply A: reverse expansion compression (drop carry tokens)")
    parser.add_argument("--enable_B", action="store_true", help="Apply B: Set-MAE masking inside each set")
    parser.add_argument("--enable_C", action="store_true", help="Apply C: next-change prediction head")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Set-MAE mask ratio when --enable_B")
    parser.add_argument("--next_change_weight", type=float, default=0.3, help="Loss weight for task C when enabled")

    # Optim & regularization
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--gradient_clip_val", type=float, default=config.gradient_clip_val)
    parser.add_argument("--stale_dropout_p", type=float, default=config.stale_dropout_p, help="Dropout prob only on carried values")

    # Anti posterior-collapse
    parser.add_argument("--warmup_beta", action="store_true", default=config.warmup_beta)
    parser.add_argument("--max_beta", type=float, default=config.max_beta)
    parser.add_argument("--beta_warmup_steps", type=int, default=config.beta_warmup_steps)
    parser.add_argument("--free_bits", type=float, default=config.free_bits, help="Per-dim free bits (nats) for KL")
    parser.add_argument("--monitor_posterior", action="store_true", help="Enable PosteriorMetricsMonitor callback")
    args = parser.parse_args()

    dm = DataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=args.smoke,
        smoke_batch_size=10,
        apply_A=bool(args.enable_A),
    )

    model = PoESeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=args.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_dropout=config.transformer_dropout,
        warmup_beta=bool(args.warmup_beta),
        max_beta=args.max_beta,
        beta_warmup_steps=args.beta_warmup_steps,
        free_bits=args.free_bits,
        stale_dropout_p=args.stale_dropout_p,
        set_mae_ratio=(args.mask_ratio if args.enable_B else 0.0),
        enable_next_change=bool(args.enable_C),
        next_change_weight=args.next_change_weight,
    )

    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="poe_pretrain")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")

    # Prepare experiment output directory
    experiment_output_dir = args.output_dir if args.output_dir else os.path.join("./outputs", args.run_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=experiment_output_dir, name="")

    callbacks = [ckpt, early, lrmon]
    if args.monitor_posterior and PosteriorMetricsMonitor is not None:
        callbacks.append(PosteriorMetricsMonitor(update_frequency=50, plot_frequency=500, log_dir=os.path.join(experiment_output_dir, "posterior")))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=0.1,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

