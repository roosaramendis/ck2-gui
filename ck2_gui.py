#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["PySide6>=6.7"]
# ///
"""CorridorKey 2.0 GUI: installer + keyer front-end (PySide6).

Run with:  uv run --script ck2_gui.py   (or the ./ck2-gui launcher)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ck2.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
