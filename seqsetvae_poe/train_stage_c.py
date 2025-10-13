#!/usr/bin/env python3
"""
Stage C training wrapper (Downstream classifier on frozen PoE backbone).
Delegates to classifier.main without changing core logic.
"""
from classifier import main


if __name__ == "__main__":
    main()
