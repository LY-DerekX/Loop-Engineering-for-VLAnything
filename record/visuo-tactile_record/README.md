# LeRobot Visuo-Tactile Capture

This folder is the visuo-tactile (vision + touch) data collection overlay for the official Hugging Face LeRobot repository.

The visuo-tactile pipeline is in progress. The recording scripts, tactile sensor backend, and dataset helpers will be added here and released open source when ready.

Official LeRobot project:

```text
https://github.com/huggingface/lerobot
```

## Scope

Use this path when the dataset contains:

- RGB camera videos;
- tactile sensor streams;
- robot state and action trajectories;
- language task metadata.

## Status

In progress. This folder currently only documents the planned layout; no recording code is included yet.

Planned contents:

- tactile sensor camera/serial backend for LeRobot;
- visuo-tactile recording entrypoint and example configs;
- tactile sidecar storage helpers aligned with the existing `depth_sidecar/` convention.

After recording, the dataset will go through the same toolkit steps as the RGB and RGB-D paths: merge, audit, clean, and upload.
