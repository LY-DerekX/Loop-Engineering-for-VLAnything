# LeRobot RGB Capture

This folder is a copy-in RGB-only recording overlay for the official Hugging Face LeRobot repository.

Copy this whole folder into a LeRobot checkout, then run the local wrappers in this folder. The wrappers call LeRobot's own `lerobot.scripts.lerobot_record` and `lerobot.scripts.lerobot_teleoperate`; they do not fork or patch LeRobot source code.

The default configs record two RGB cameras:

- `head`: fixed head/global camera.
- `wrist`: wrist-mounted camera.

Official LeRobot project:

```text
https://github.com/huggingface/lerobot
```

Official recording and camera documentation:

```text
https://huggingface.co/docs/lerobot/il_robots
https://huggingface.co/docs/lerobot/cameras
```

## Scope

Use this path when the dataset contains:

- RGB camera videos through stock LeRobot camera backends such as OpenCV;
- robot state and action trajectories;
- language task metadata;
- no depth stream and no `depth_sidecar/` directory.

## Copy Into LeRobot

From this toolkit repository:

```powershell
$LEROBOT_ROOT="E:\path\to\lerobot"
Copy-Item -Recurse -Force .\record\rgb_record $LEROBOT_ROOT\
Copy-Item -Force .\record\camera_profiles.yaml $LEROBOT_ROOT\
Copy-Item -Force .\record\camera_profiles.py $LEROBOT_ROOT\
```

`camera_profiles.yaml` is the shared camera inventory used by every recording
path (RGB, RGB-D, visuo-tactile). It is optional here: without
`--camera_profile`, the inline `cameras:` block in the record config is used
unchanged.

Then work from the LeRobot root:

```powershell
cd $LEROBOT_ROOT
python -m pip install --upgrade pip
python -m pip install -e ".[feetech,core_scripts]"
```

Choose the LeRobot extras that match your hardware. For SO-100 and SO-101, `feetech` is normally required. On older LeRobot versions that do not define `core_scripts`, use `python -m pip install -e ".[feetech]"`.

## Find Cameras

List OpenCV cameras:

```powershell
lerobot-find-cameras opencv
```

Record the camera indices, resolution, and FPS that you want to use. On Windows, `index_or_path: 0` is usually the first USB camera, but indices can change after unplugging or rebooting.

Update these fields in:

```text
rgb_record/configs/head_wrist_rgb_record.yaml
rgb_record/configs/head_wrist_rgb_teleoperate.yaml
```

## Teleoperate With Head And Wrist Preview

```powershell
python .\rgb_record\rgb_teleoperate.py `
  --config_path=.\rgb_record\configs\head_wrist_rgb_teleoperate.yaml
```

Use the same robot and teleoperator IDs for teleoperation, recording, and replay so LeRobot can reuse the correct calibration files.

## Record Head And Wrist RGB Dataset

```powershell
python .\rgb_record\rgb_record.py `
  --config_path=.\rgb_record\configs\head_wrist_rgb_record.yaml
```

The same config can still be overridden from the command line:

```powershell
python .\rgb_record\rgb_record.py `
  --config_path=.\rgb_record\configs\head_wrist_rgb_record.yaml `
  --dataset.num_episodes=5 `
  --dataset.single_task="Pick up the cup"
```

Select the cameras from the shared inventory instead of the inline block:

```powershell
python .\rgb_record\rgb_record.py `
  --config_path=.\rgb_record\configs\head_wrist_rgb_record.yaml `
  --camera_profile=rgb_head_wrist
```

The selected profile replaces `robot.cameras`; every other record field stays
as configured. Profiles that enable `use_depth` are rejected on this RGB-only
path with a pointer to `rgbd_record/`.

Replace `COM5`, `COM6`, camera indices, resolution, FPS, repository ID, episode count, and task text with the values for the real setup.

## Dataset Output

The RGB-only dataset is a standard LeRobot dataset:

```text
data/
meta/
videos/
```

It should not contain:

```text
depth_sidecar/
meta/rgbd_vla_depth_recording.json
```

After recording, run this toolkit's merge, audit, clean, and upload steps on the dataset root.

## Operational Notes

- Keep camera names stable across datasets. The default names are `head` and `wrist`.
- Keep FPS, resolution, action space, and robot IDs stable when collecting data that will be merged.
- Use `--resume=true` only when continuing the same interrupted recording session.
- Set `--dataset.push_to_hub=False` when you want to audit and clean locally before uploading.
