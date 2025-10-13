#!/usr/bin/env python3
"""
Stage B training wrapper (Dynamics + conditional PoE pretraining).
Delegates to _poe_PT.main without changing core logic.
"""
from _poe_PT import main


if __name__ == "__main__":
    main()
