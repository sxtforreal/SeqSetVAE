#!/usr/bin/env python3
"""
Deprecated: This script has been renamed to LVCF.py.

Please use `LVCF.py` which additionally supports event embedding mapping and value
normalization from offline CSVs, plus a smoke test mode. This wrapper forwards
all CLI args to the new entrypoint.
"""

try:
    # When running as module (python -m seqsetvae_poe.preprocess_expand)
    from .LVCF import main  # type: ignore
except Exception:  # noqa: E722
    # When running as script from the same directory
    from LVCF import main  # type: ignore


if __name__ == "__main__":
    main()

