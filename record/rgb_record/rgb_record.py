#!/usr/bin/env python

"""RGB-only LeRobot recording wrapper.

Run this file from an official LeRobot checkout after copying the
`rgb_record/` folder (plus `camera_profiles.yaml` and `camera_profiles.py`)
into the LeRobot repository root.

Optional shared camera selection:

    python ./rgb_record/rgb_record.py \
      --config_path=./rgb_record/configs/head_wrist_rgb_record.yaml \
      --camera_profile=rgb_head_wrist

Profiles live in `<lerobot_root>/camera_profiles.yaml`. Profiles that enable
depth are rejected here; use the rgbd_record path for depth capture.
"""

import sys
from pathlib import Path

LEROBOT_ROOT = Path(__file__).resolve().parents[1]


def _apply_camera_profile(argv: list[str]) -> list[str]:
    if not any(arg.startswith("--camera_profile") for arg in argv[1:]):
        return argv
    if str(LEROBOT_ROOT) not in sys.path:
        sys.path.insert(0, str(LEROBOT_ROOT))
    from camera_profiles import rewrite_argv_with_camera_profile

    return rewrite_argv_with_camera_profile(argv, allow_depth=False, search_root=LEROBOT_ROOT)


from lerobot.scripts.lerobot_record import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = _apply_camera_profile(sys.argv)
    main()
