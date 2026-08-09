#!/usr/bin/env python3
"""Convenience entry point for smoothing rosbag2 storage timestamps."""

from pathlib import Path
import runpy
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rl_sar"
    / "scripts"
    / "offline"
    / "smooth_bag_record_time.py"
)

if not SCRIPT.is_file():
    print(f"[smooth-bag] target script not found: {SCRIPT}", file=sys.stderr)
    raise SystemExit(2)

sys.argv[0] = str(SCRIPT)
if len(sys.argv) > 1 and "-h" not in sys.argv[1:] and "--help" not in sys.argv[1:]:
    if "--force" not in sys.argv:
        sys.argv.insert(1, "--force")
runpy.run_path(str(SCRIPT), run_name="__main__")
