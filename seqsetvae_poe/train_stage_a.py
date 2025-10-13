#!/usr/bin/env python3
"""
Stage A training wrapper (SetVAE-only pretraining).
Delegates to _setvae_PT.main without changing core logic.
"""
from _setvae_PT import main


if __name__ == "__main__":
    main()
