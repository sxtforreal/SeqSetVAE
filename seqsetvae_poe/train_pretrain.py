#!/usr/bin/env python3
import os
import sys
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Make script runnable both as package and as a standalone file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

try:
    from . import config  # type: ignore
    from .dataset import DataModule  # type: ignore
    from .model import PoESeqSetVAEPretrain  # type: ignore
except Exception:
    import config
    from dataset import DataModule
    from model import PoESeqSetVAEPretrain


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAE-PoE Pretraining")
    parser.add_argument("--data_dir", type=str, default=config.data_dir)
    parser.add_argument("--params_map_path", type=str, default=config.params_map_path)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    args = parser.parse_args()

    dm = DataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PoESeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_dropout=config.transformer_dropout,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        free_bits=config.free_bits,
        stale_dropout_p=config.stale_dropout_p,
        set_mae_ratio=config.set_mae_ratio,
    )

    ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="poe_pretrain")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    lrmon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=os.path.join("./outputs", config.name), name="")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[ckpt, early, lrmon],
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        val_check_interval=0.1,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

