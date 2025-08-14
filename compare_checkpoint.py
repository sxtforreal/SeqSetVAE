#!/usr/bin/env python3
import argparse
from typing import Dict, List

import torch

import config
from model import SeqSetVAE, load_checkpoint_weights


def summarize_prefix_matches(model_state: Dict[str, torch.Tensor], ckpt_state: Dict[str, torch.Tensor], prefixes: List[str]):
    """Return per-prefix stats: model_keys, ckpt_keys, matched_shape_keys."""
    stats = {}
    for prefix in prefixes:
        model_keys = [k for k in model_state.keys() if k.startswith(prefix)]
        ckpt_keys = [k for k in ckpt_state.keys() if k.startswith(prefix)]
        matched = [
            k for k in model_keys
            if k in ckpt_state and isinstance(ckpt_state[k], torch.Tensor) and ckpt_state[k].shape == model_state[k].shape
        ]
        stats[prefix] = {
            "model": len(model_keys),
            "ckpt": len(ckpt_keys),
            "matched": len(matched),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoint coverage against SeqSetVAE model")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to pretrained checkpoint (Lightning ckpt or state_dict)")
    args = parser.parse_args()

    # Build a fresh model (random init)
    model = SeqSetVAE(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        num_classes=config.num_classes,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        pretrained_ckpt=None,
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
        use_focal_loss=getattr(config, 'use_focal_loss', True),
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
    model_state = model.state_dict()

    # Load checkpoint weights only
    ckpt_state = load_checkpoint_weights(args.checkpoint, device='cpu')

    prefixes = [
        "setvae",
        "transformer",
        "post_transformer_norm",
        "decoder",
        "feature_fusion",
        "feature_projection",
        "time_encoder",
        "pos_embedding",
        "cls_head",
    ]

    stats = summarize_prefix_matches(model_state, ckpt_state, prefixes)

    print("==== Checkpoint vs Model Coverage (by module prefix) ====")
    for p in prefixes:
        st = stats[p]
        print(f"{p:>24s}: model={st['model']:4d} | ckpt={st['ckpt']:4d} | matched_shape={st['matched']:4d}")

    # Heuristic conclusions
    issues = []
    if stats["transformer"]["ckpt"] == 0:
        issues.append("No transformer weights in checkpoint -> transformer would remain random if you freeze it.")
    elif stats["transformer"]["matched"] == 0:
        issues.append("Transformer keys exist but shapes don't match -> not loaded (random if frozen).")

    if stats["setvae"]["ckpt"] == 0:
        issues.append("No setvae weights found in checkpoint.")

    if issues:
        print("\nPotential issues:")
        for s in issues:
            print(f" - {s}")
    else:
        print("\nAll major modules appear in the checkpoint with matching shapes.")

    # List a few example keys per module for manual inspection
    print("\nExamples (first 3 keys per present module in checkpoint):")
    for p in prefixes:
        keys = [k for k in ckpt_state.keys() if k.startswith(p)]
        if not keys:
            continue
        print(f"[{p}] -> {len(keys)} keys")
        for k in keys[:3]:
            v = ckpt_state[k]
            shape = tuple(v.shape) if isinstance(v, torch.Tensor) else None
            print(f"  - {k}: {shape}")


if __name__ == "__main__":
    main()