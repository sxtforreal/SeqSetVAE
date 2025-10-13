#!/usr/bin/env python3
"""
Unified eval entrypoint with --stage {A,B,C}.
 - A,B: delegate to _eval_PT (unsupervised diagnostics)
 - C: evaluate frozen-backbone classifier on test split
"""
from __future__ import annotations
import argparse
import sys
from _eval_PT import main as eval_ab_main  # type: ignore
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar
import config as cfg  # type: ignore
from dataset import MortalityDataModule  # type: ignore
from classifier import MortalityClassifier, _load_state_dict, _build_poe_from_state  # type: ignore


def _inject_ckpt_type_from_stage():
    # If user provided --stage, map to ckpt_type and inject into argv if not explicitly set
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=str, choices=["A", "B", "C"], required=False)
    args, _ = parser.parse_known_args()
    if not args.stage:
        return
    # Stage C handled separately below. For A/B only: inject ckpt_type
    for tok in sys.argv:
        if tok.startswith("--ckpt_type"):
            return
    if args.stage in {"A", "B"}:
        map_type = {"A": "setvae", "B": "poe"}[args.stage]
        sys.argv.append("--ckpt_type")
        sys.argv.append(map_type)
    # Remove --stage to avoid unknown-arg in _eval_PT
    new_argv = [sys.argv[0]]
    skip = 0
    for i, tok in enumerate(sys.argv[1:], start=1):
        if skip:
            skip -= 1
            continue
        if tok == "--stage":
            skip = 1
            continue
        if tok.startswith("--stage="):
            continue
        new_argv.append(tok)
    sys.argv = new_argv


def _run_stage_c():
    import argparse as _ap
    ap = _ap.ArgumentParser(description="Stage C evaluation (classifier on PoE features)")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to PoE checkpoint (.ckpt)")
    ap.add_argument("--label_csv", required=True, type=str)
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 1))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--output_dir", type=str, default="./output")
    args, _ = ap.parse_known_args()

    state = _load_state_dict(args.checkpoint)
    poe = _build_poe_from_state(state)
    dm = MortalityDataModule(
        saved_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=False,
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
    project_dir = os.path.join(out_root, "classifier_eval")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    trainer = pl.Trainer(logger=logger, callbacks=[TQDMProgressBar()], log_every_n_steps=10)
    # Run only test (no fit)
    trainer.test(model, dataloaders=dm.test_dataloader())


def main():
    # Peek stage
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stage", type=str, choices=["A", "B", "C"], required=False)
    s_args, _ = p.parse_known_args()
    if s_args.stage == "C":
        return _run_stage_c()
    _inject_ckpt_type_from_stage()
    return eval_ab_main()


if __name__ == "__main__":
    main()
