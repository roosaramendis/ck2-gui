#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["PySide6>=6.7", "certifi"]
# ///
"""CorridorKey 2.0 GUI: installer + keyer front-end (PySide6).

Run with:  uv run --script ck2_gui.py   (or ./ck2-gui, ck2-gui.bat, or the packaged app)
`--task <name> ...` runs one background setup step (used by the app itself).
"""
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def entry() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--task":
        from ck2.tools import run_task
        return run_task(sys.argv[2:])
    from ck2.app import main
    return main()


if __name__ == "__main__":
    sys.exit(entry())
