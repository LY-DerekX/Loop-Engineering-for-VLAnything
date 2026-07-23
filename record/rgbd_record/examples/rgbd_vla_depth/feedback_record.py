"""Compatibility wrapper for human-guided RGB-D feedback collection."""

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
TOOLKIT_ROOT = PROJECT_ROOT.parents[2]
for path in (SRC_ROOT, TOOLKIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "feedback_record.py"), run_name="__main__")
