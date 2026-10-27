#!/usr/bin/env python3
"""Compatibility entry point for running directly from a source checkout."""

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from logic4life_sleep.cli import main

    raise SystemExit(main())
