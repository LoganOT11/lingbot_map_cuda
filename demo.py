#!/usr/bin/env python3
"""Thin shim — delegates to apps/cli/demo.py for backward compatibility."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    cli_demo = Path(__file__).resolve().parent / "apps" / "cli" / "demo.py"
    sys.argv[0] = str(cli_demo)
    runpy.run_path(str(cli_demo), run_name="__main__")
