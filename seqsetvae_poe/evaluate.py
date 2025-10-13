#!/usr/bin/env python3
"""
Unified eval entrypoint with --stage {A,B}.
Delegates to _eval_PT.py (which detects ckpt type automatically), while allowing a --stage shortcut.
"""
from __future__ import annotations
import argparse
import sys
from _eval_PT import main as eval_main  # type: ignore


def _inject_ckpt_type_from_stage():
    # If user provided --stage, map to ckpt_type and inject into argv if not explicitly set
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=str, choices=["A", "B"], required=False)
    args, _ = parser.parse_known_args()
    if not args.stage:
        return
    # If user already provided --ckpt_type, do nothing
    for tok in sys.argv:
        if tok.startswith("--ckpt_type"):
            return
    # Map A->setvae, B->poe
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


def main():
    _inject_ckpt_type_from_stage()
    return eval_main()


if __name__ == "__main__":
    main()
