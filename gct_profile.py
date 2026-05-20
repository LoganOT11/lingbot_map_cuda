#!/usr/bin/env python3
"""Thin shim — delegates to apps/cli/gct_profile.py for backward compatibility."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    cli_profile = Path(__file__).resolve().parent / "apps" / "cli" / "gct_profile.py"
    sys.argv[0] = str(cli_profile)
    runpy.run_path(str(cli_profile), run_name="__main__")
