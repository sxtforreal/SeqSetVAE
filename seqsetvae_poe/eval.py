#!/usr/bin/env python3
"""
Evaluation utilities and entrypoint.
Re-exports from _eval_PT to provide a stable eval.py module.
"""
from _eval_PT import (
    compute_kl_dataset,
    compute_posterior_moments_dataset,
    collect_mu_var_heatmaps,
    collect_recon_events,
)

# Keep a stable CLI entrypoint
from _eval_PT import main


if __name__ == "__main__":
    main()
