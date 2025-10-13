#!/usr/bin/env python3
"""
Unified train entrypoint with --stage {A,B,CA,B,C}.

- Stage A: SetVAE-only pretraining (_setvae_PT)
- Stage B: Dynamics + conditional PoE pretraining (_poe_PT)
- Stage C: Downstream classifier (classifier)

All other arguments are forwarded to the underlying stage script.
"""
from __future__ import annotations
import argparse
import sys

# Import stage mains (they each parse sys.argv themselves)
from _setvae_PT import main as stage_a_main  # type: ignore
from _poe_PT import main as stage_b_main  # type: ignore
from classifier import main as stage_c_main  # type: ignore


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
        return stage_a_main()
    if args.stage == "B":
        return stage_b_main()
    return stage_c_main()


if __name__ == "__main__":
    main()
