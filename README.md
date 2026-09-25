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

Deployment has two halves connected over a websocket: a **policy server** (GPU workstation) and a **robot client**.

The robot client requires the deployment hardware:
- **Franka arm + ZED camera** — the standard DROID robot platform.
- **Sharpa hand** (for Sharpa configs) — obtain the `SharpaWaveSDK_4.3.4/` SDK and place it at the repo root; it is loaded at runtime. Requires Python 3.10–3.12.

Set `CLOAK_NUC_IP` to your Franka NUC's IP (e.g. `export CLOAK_NUC_IP=<nuc-ip>` in `~/.bashrc`).

1. Download trained checkpoints from [here](https://drive.google.com/drive/folders/1DWXAGzNjsc-iyNWnbEMEgyhw4JkP-a7f?usp=drive_link). The checkpoint name corresponds to the train config in `src/openpi/training/config.py`.
2. Before running, upload the end-effector config for your gripper to Franka Desk (Settings → End-Effector) so the
controller uses the correct mass and inertia. Configs live in `deployment/endeffector_configs/` (`robotiq_2f85.json`,
`sharpa_angled.json`, `umi_roll135.json`).

3. Calibrate the wrist camera with Silhouette Calibration. This produces a camera extrinsics vector saved to `deployment/calibration/calibration_info_<embodiment>.json`

```bash
# Make sure to set `WRIST_CAMERA_ID` and `EMBODIMENT` variables at the top of the script.
bash scripts/calibrate_wrist_camera.sh
```

4. Serve the policy.

Set `CONFIG`, `CHECKPOINT_DIR`, and `EMBODIMENT` at the top of `scripts/serve_policy.sh`.

```bash
bash scripts/serve_policy.sh
```

5. Run the client.

Set `EMBODIMENT` (matching the server's) and your ZED serials `EXTERNAL_CAMERA_ID` / `WRIST_CAMERA_ID` (`SN<serial>`
files under `/usr/local/zed/settings/`) at the top of `deploy_policy.sh`.

```bash
bash scripts/deploy_policy.sh
```

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
