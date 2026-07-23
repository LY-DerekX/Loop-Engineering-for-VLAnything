# LeRobot RGB-D Capture (Orbbec / Intel RealSense)

This folder contains the LeRobot-side code needed to collect RGB-D data with an Orbbec Femto Bolt or an Intel RealSense D405 / D435 / D435i camera.

It extends the data pipeline upstream of merge, audit, clean, and upload by adding a real robot data collection path.

## Supported RGB-D Cameras

| Camera | Config `type` | Depth resolution | Notes |
| --- | --- | --- | --- |
| Orbbec Femto Bolt | `orbbec` | independent `depth_width/height/fps`, optional align-to-color | overlay backend (`src/lerobot/cameras/orbbec/`) |
| Intel RealSense D435 / D435i | `intelrealsense` | same `width/height/fps` as the color stream | uses LeRobot's built-in RealSense backend; 848x480@30 is the depth sweet spot |
| Intel RealSense D405 | `intelrealsense` | same `width/height/fps` as the color stream | near-range depth (~7cm-50cm), ideal wrist camera |
| Plain UVC webcam | `opencv` | no depth | RGB only |

Any camera with `use_depth: true` produces an `<name>_depth` observation that is
stored as a lossless uint16 PNG sidecar; the SO follower normalizes both the
Orbbec `(H, W)` and RealSense `(H, W, 1)` depth buffers to the same dataset
shape.

Two ways to switch cameras:

1. **Shared camera inventory (recommended)**: edit hardware facts once in
   `<lerobot_root>/camera_profiles.yaml` (copied from `record/camera_profiles.yaml`)
   and select at record time with `--camera_profile=<name>`. The same inventory
   serves the RGB, RGB-D, and future visuo-tactile recording paths.
2. **Inline YAML blocks**: keep cameras inside the record config — see the
   commented variants in `examples/rgbd_vla_depth/configs/record_rgbd.yaml` and
   the ready-to-run all-RealSense config
   `examples/rgbd_vla_depth/configs/record_rgbd_realsense.yaml`.

## Scope

This folder is only for data collection. It adds RGB-D capture support (Orbbec backend plus RealSense-compatible wiring), recording scripts, and lossless depth sidecar storage helpers.

It does not contain this project's depth model training code, depth model inference code, evaluation config, or pretrained model weights. The recorder follows LeRobot's normal recording interfaces, but the included project-local Python package only provides RGB-D data helpers.

## Where To Put It In LeRobot

Copy the contents of this folder into the root of a `huggingface/lerobot` checkout. The target is the directory that contains LeRobot's own `pyproject.toml` and `src/lerobot/`.

Do not copy this folder as:

```text
<lerobot_root>/rgbd_record/
```

Merge these contents into the LeRobot repo root instead:

```powershell
$LEROBOT_ROOT="<path-to-lerobot>"
Copy-Item -Recurse -Force .\record\rgbd_record\examples $LEROBOT_ROOT\
Copy-Item -Recurse -Force .\record\rgbd_record\src $LEROBOT_ROOT\
Copy-Item -Force .\record\rgbd_record\pyproject.toml $LEROBOT_ROOT\
Copy-Item -Force .\record\camera_profiles.yaml $LEROBOT_ROOT\
Copy-Item -Force .\record\camera_profiles.py $LEROBOT_ROOT\
Copy-Item -Recurse -Force .\policy_improvement $LEROBOT_ROOT\
```

After copying, the expected layout is:

```text
<lerobot_root>/examples/rgbd_vla_depth/
<lerobot_root>/policy_improvement/
<lerobot_root>/src/lerobot/cameras/orbbec/
<lerobot_root>/src/lerobot/scripts/
<lerobot_root>/src/lerobot/robots/so_follower/
<lerobot_root>/pyproject.toml
```

Install from the LeRobot repo root:

```powershell
cd $LEROBOT_ROOT
python -m pip install -e ".[feetech,orbbec,intelrealsense,rgbd_vla]" -e .\examples\rgbd_vla_depth
```

Drop `orbbec` or `intelrealsense` from the extras list if that camera family is
not used:

```powershell
python -m pip install -e ".[feetech,intelrealsense,rgbd_vla]" -e .\examples\rgbd_vla_depth
```

## What It Provides

- `src/lerobot/cameras/orbbec/`: Orbbec RGB-D camera backend.
- `src/lerobot/scripts/`: patched LeRobot commands for finding cameras, recording, calibration, and teleoperation.
- `src/lerobot/robots/so_follower/`: SO follower support for RGB-D observations.
- `examples/rgbd_vla_depth/record.py`: RGB-D recording entrypoint.
- `examples/rgbd_vla_depth/feedback_record.py`: human-guided policy feedback entrypoint.
- `examples/rgbd_vla_depth/configs/record_rgbd.yaml`: camera-switchable recording config (Orbbec active, RealSense D435/D405 variants included as commented blocks).
- `examples/rgbd_vla_depth/configs/record_rgbd_realsense.yaml`: ready-to-run all-RealSense config (D435 front + D405 wrist).
- `examples/rgbd_vla_depth/configs/record_rgbd_feedback.yaml`: recovery/correction, event-buffer, and continuous capture config.
- `examples/rgbd_vla_depth/src/lerobot_policy_rgbd_vla_depth/data/`: helpers for depth feature naming and uint16 PNG sidecar storage.

## Dataset Output

The recorder keeps RGB streams in standard LeRobot video storage and saves depth frames as lossless uint16 PNG files:

```text
data/
videos/
depth_sidecar/observation.images.front_depth/episode-000000/frame-000000.png
meta/rgbd_vla_depth_recording.json
```

## Typical Entry Points

Find the cameras:

```powershell
lerobot-find-cameras orbbec
lerobot-find-cameras realsense
```

Record RGB-D data (edit the camera blocks in the YAML to switch cameras):

```powershell
python .\examples\rgbd_vla_depth\record.py --config_path=.\examples\rgbd_vla_depth\configs\record_rgbd.yaml
```

Record with the all-RealSense rig (D435 front + D405 wrist):

```powershell
python .\examples\rgbd_vla_depth\record.py --config_path=.\examples\rgbd_vla_depth\configs\record_rgbd_realsense.yaml
```

Or select any camera set from the shared inventory without editing the record
config (works for `feedback_record.py` too):

```powershell
python .\examples\rgbd_vla_depth\record.py `
  --config_path=.\examples\rgbd_vla_depth\configs\record_rgbd.yaml `
  --camera_profile=realsense_d435_front_d405_wrist
```

Collect policy feedback with explicit Recovery/Correction labels:

```powershell
python .\examples\rgbd_vla_depth\feedback_record.py `
  --config_path=.\examples\rgbd_vla_depth\configs\record_rgbd_feedback.yaml `
  --policy.path=.\outputs\train\run\checkpoints\last\pretrained_model
```

The safe default is `feedback.mode: corrections_only`. Press `i` to align the
actuated teleoperator and begin Recovery, `c` to mark Correction, and `i` again
to save the intervention and hand control back to the policy. The associated
feedback sidecar stays outside the recorded dataset.

Visualize recorded depth after collection from this repository root:

```powershell
python .\tools\depth_visualize.py <dataset_root>\depth_sidecar --stats
```

The main config to edit before recording is:

```text
examples/rgbd_vla_depth/configs/record_rgbd.yaml
```

Important fields in `record_rgbd.yaml`:

```yaml
robot:
  cameras:
    front:
      type: orbbec
      serial_number_or_name: "YOUR_FEMTO_BOLT_SERIAL"
      use_depth: true
      align_depth_to_color: true
dataset:
  repo_id: YOUR_HF_USERNAME/so100_rgbd_dataset
  fps: 30
  num_episodes: 10
  episode_time_s: 60
```
