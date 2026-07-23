"""Compatibility wrapper for the project-local RGB-D recording entrypoint.

Optional shared camera selection (profiles in `<lerobot_root>/camera_profiles.yaml`):

    python ./examples/rgbd_vla_depth/record.py \
      --config_path=./examples/rgbd_vla_depth/configs/record_rgbd.yaml \
      --camera_profile=realsense_d435_front_d405_wrist
"""

import sys
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
LEROBOT_ROOT = PROJECT_ROOT.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _apply_camera_profile(argv: list[str]) -> list[str]:
    if not any(arg.startswith("--camera_profile") for arg in argv[1:]):
        return argv
    if str(LEROBOT_ROOT) not in sys.path:
        sys.path.insert(0, str(LEROBOT_ROOT))
    from camera_profiles import rewrite_argv_with_camera_profile

    return rewrite_argv_with_camera_profile(argv, allow_depth=True, search_root=LEROBOT_ROOT)


if __name__ == "__main__":
    sys.argv = _apply_camera_profile(sys.argv)
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "record.py"), run_name="__main__")
