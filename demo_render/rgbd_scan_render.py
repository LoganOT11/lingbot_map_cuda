#!/usr/bin/env python3
"""Thin shim — delegates to apps/batch/rgbd_scan_render.py."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "apps" / "batch" / "rgbd_scan_render.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
