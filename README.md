# Cloak

[![Project Page](https://img.shields.io/badge/Project_Page-blue)](https://tml.stanford.edu/cloak/)

A framework for robot policy calibration, training, and deployment.

## Installation

```bash
uv sync
```

This installs the base (policy-serving) dependencies. The robot client and
training loop pull in their own extras automatically via `uv run --group
client ...` / `uv run --group train ...` (used by `deploy_policy.sh` /
`train.sh`) — no separate install step needed.

For deployment, you also need the hardware dependencies:
- **Franka arm + ZED camera** — the standard DROID robot platform. See the
  [DROID setup docs](https://droid-dataset.github.io/droid).
- **Sharpa hand** (for Sharpa configs) — obtain the `SharpaWaveSDK_4.3.4/` SDK and
  place it at the repo root; it is loaded at runtime. Requires Python 3.10–3.12.


(Optional) Download trained checkpoints from [here](https://drive.google.com/drive/folders/1DWXAGzNjsc-iyNWnbEMEgyhw4JkP-a7f?usp=drive_link). The checkpoint name corresponds to the train config in `src/openpi/training/config.py`.



## Training

Please see `README_train.md` for training details.


## Deployment

Deployment has two halves: a **policy server** (GPU workstation) and a **robot client**, connected over a websocket.

To serve the policy, run:

```bash
bash scripts/serve_policy.sh
```

Set `CONFIG` and `CHECKPOINT_DIR` at the top of the script (available robots are
listed there).

Also run the client, which accepts actions from the policy server and runs the robot environment:

```bash
bash scripts/deploy_policy.sh
```

Before running, set the variables at the top of `deploy_policy.sh`:
- `EMBODIMENT` — `sharpa` or `robotiq`, matching the served config.
- `EXTERNAL_CAMERA_ID` / `WRIST_CAMERA_ID` — your ZED serial numbers (the
  `SN<serial>` files under `/usr/local/zed/settings/`).

The robot client reads the NUC's network address from an environment variable so
no site-specific address is committed to the repo:
- `CLOAK_NUC_IP` — IP of the NUC running the Franka controller. If unset, the
  Franka controller is launched locally instead of connecting over the network.

Keep this out of version control — for example, put your value in an untracked
script under `scratch/` and `source` it before deploying.

## Calibrated DROID wrist extrinsics

We release per-episode wrist-camera extrinsics for [DROID](https://droid-dataset.github.io/) episodes — the camera's pose relative to the end-effector ("cam_to_gripper"). Each is recovered by our **Silhouette Calibration** algorithm, which produces more accurate alignment to sim than the extrinsics shipped with DROID.

- `assets/droid_wrist_extrinsics.json` — JSON containing the 6-DoF camera pose relative to the Franka attachment_site.
- `examples/render_extrinsics.py` — Minimal example of using the extrinsics to render the wrist view.

### Regenerating the extrinsics

If you would like to rerun the optimization via Silhouette Calibration., use `scripts/preprocess_wrist_extrinsics.py`.

```
uv run python scripts/preprocess_wrist_extrinsics.py --data-dir /path/to/DROID --output out.json
```
