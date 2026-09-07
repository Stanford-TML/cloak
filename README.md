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

(Optional) Download trained checkpoints from [here](https://drive.google.com/drive/folders/1DWXAGzNjsc-iyNWnbEMEgyhw4JkP-a7f?usp=drive_link). The checkpoint name corresponds to the train config in `src/openpi/training/config.py`.



## Training

Please see `README_train.md` for training details.


## Deployment

Deployment has two halves connected over a websocket: a **policy server** (GPU workstation) and a **robot client**.

The robot client requires the deployment hardware:
- **Franka arm + ZED camera** — the standard DROID robot platform. See the [DROID setup docs](https://droid-dataset.github.io/droid).
- **Sharpa hand** (for Sharpa configs) — obtain the `SharpaWaveSDK_4.3.4/` SDK and place it at the repo root; it is loaded at runtime. Requires Python 3.10–3.12.

Serve the policy:

```bash
bash scripts/serve_policy.sh
```

Set `CONFIG`, `CHECKPOINT_DIR`, and `EMBODIMENT` at the top of the script. One config
(`pi05_full_droid_finetune_v3-1_ik`) serves every robot; `EMBODIMENT` (`robotiq` | `sharpa` | `umi` | `yam`) selects
the wrist mask renderer and cross-embodiment IK at serve time.

Run the client:

```bash
bash scripts/deploy_policy.sh
```

Set `EMBODIMENT` (matching the server's) and your ZED serials `EXTERNAL_CAMERA_ID` / `WRIST_CAMERA_ID` (`SN<serial>`
files under `/usr/local/zed/settings/`) at the top of `deploy_policy.sh`. Set the `CLOAK_NUC_IP` env var to the Franka
NUC's IP; if unset, the controller runs locally.

## Calibrated DROID wrist extrinsics

We release per-episode wrist-camera extrinsics for [DROID](https://droid-dataset.github.io/) episodes, the camera's
pose relative to the end-effector. Each is recovered by our **Silhouette Calibration** algorithm, which produces more
accurate alignment to sim than the extrinsics shipped with DROID.

- `assets/droid_wrist_extrinsics.json` — JSON containing the 6-DoF camera pose relative to the Franka attachment_site.
- `examples/render_extrinsics.py` — Minimal example of using the extrinsics to render the wrist view.

To regenerate them via Silhouette Calibration, see [Data preprocessing](README_train.md) (optional — we ship the precomputed extrinsics).
