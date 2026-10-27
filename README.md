# Cloak

[![Project Page](https://img.shields.io/badge/Project_Page-blue)](https://tml.stanford.edu/cloak/)

Cloak fine-tunes π0.5 (built on openpi) on DROID with the gripper masked out of the wrist camera, so a single
checkpoint trained only on Franka + Robotiq data can control other end effectors and arms (Sharpa hand, UMI gripper,
YAM arm). We also release per-episode DROID wrist-camera extrinsics recovered with our Silhouette Calibration method.

## Installation

```bash
uv sync
```

This installs the base (policy-serving) dependencies. The robot client and
training loop pull in their own extras automatically via `uv run --group
client ...` / `uv run --group train ...` (used by `deploy_policy.sh` /
`train.sh`) — no separate install step needed.


## Training

Please see `README_train.md` for training details.


## Deployment

Follow the [Cloak setup guide](https://tml.stanford.edu/cloak/setup/) for hardware assembly, workstation and NUC setup, silhouette calibration, and policy deployment:

- [Robotiq](https://tml.stanford.edu/cloak/setup/robotiq.html)
- [UMI](https://tml.stanford.edu/cloak/setup/umi.html)
- [Sharpa](https://tml.stanford.edu/cloak/setup/sharpa.html)

The guide is maintained under `docs/setup/` and published with the project website. YAM instructions are forthcoming.

## Calibrated DROID wrist extrinsics

We release per-episode wrist-camera extrinsics for [DROID](https://droid-dataset.github.io/) episodes, the camera's
pose relative to the end-effector. Each is recovered by our **Silhouette Calibration** algorithm, which produces more
accurate alignment to sim than the extrinsics shipped with DROID.

- `assets/droid_wrist_extrinsics.json` — JSON containing the 6-DoF camera pose relative to the Franka attachment_site.
- `examples/render_extrinsics.py` — Minimal example of using the extrinsics to render the wrist view.

To regenerate them via Silhouette Calibration, see [Data preprocessing](README_train.md) (optional — we ship the precomputed extrinsics).

## Tests

```bash
uv run --group dev pytest
```
