#!/usr/bin/env python3
"""Run the argmap validator over this directory: `python check.py`.

The validator itself is generic and lives in `../argmap/argmap.py`; this
file only points it at these notes so the command works from here. Any options
given here are passed straight through, so `python check.py --json` and
`python check.py --strict` work too. Settings for this project are in
`argmap.json`. Stdlib only, no dependencies.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATOR = HERE.parent / "argmap" / "argmap.py"

if not VALIDATOR.exists():
    sys.exit(f"validator not found at {VALIDATOR}")

sys.argv = [str(VALIDATOR), str(HERE)] + [a for a in sys.argv[1:]
                                          if a.startswith("-")]
runpy.run_path(str(VALIDATOR), run_name="__main__")
