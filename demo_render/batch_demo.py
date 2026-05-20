#!/usr/bin/env python3
"""Thin shim — delegates to apps/batch/main.py for backward compatibility."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    batch_main = Path(__file__).resolve().parent.parent / "apps" / "batch" / "main.py"
    sys.argv[0] = str(batch_main)
    runpy.run_path(str(batch_main), run_name="__main__")
