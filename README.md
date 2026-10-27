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

### Run inference

Complete the embodiment's setup guide first, including the NUC bridge and wrist-camera calibration. The commands below assume the robot is configured and its matching calibration file is available.

Download **Cloak-VLA** from the [checkpoint folder](https://drive.google.com/drive/folders/1DWXAGzNjsc-iyNWnbEMEgyhw4JkP-a7f). Reuse an existing download. `CHECKPOINT_DIR` must point to the checkpoint directory containing `params/` and `assets/`.

From the repository root, install the client dependencies and create a settings file if one does not already exist:

```bash
uv sync --group client
[ -f .env ] || cp .env.example .env
```

Edit `.env` with your rig settings. Choose `robotiq`, `umi`, or `sharpa` for `EMBODIMENT`, and replace the placeholders:

```dotenv
EMBODIMENT=robotiq
CLOAK_NUC_IP=<your-nuc-ip>
WRIST_CAMERA_ID=<wrist-zed-serial>
EXTERNAL_CAMERA_ID=<external-zed-serial>
WRIST_CAM_LENS=left
POLICY_HOST=127.0.0.1
POLICY_PORT=8000
CONFIG=pi05_full_droid_finetune_v3-1_ik
CHECKPOINT_DIR=/absolute/path/to/cloak_VLA/100000
DEBUG_MASKS=true
```

Use `127.0.0.1` when the policy server and robot client run on the same workstation. Otherwise, set `POLICY_HOST` to the policy server's address.

Start the policy server in one terminal:

```bash
bash scripts/serve_policy.sh
```

Once it is listening, start the robot client in another terminal from the same repository:

```bash
bash scripts/deploy_policy.sh
```

Both scripts read `EMBODIMENT` from `.env`. To override it, pass the same `--embodiment robotiq`, `--embodiment umi`, or `--embodiment sharpa` argument to both scripts. The Sharpa launcher selects coordinated finger control automatically.

Client startup and reset move the arm home. Keep that path clear and supervise the robot. At the prompt, enter a task instruction such as `put the red marker in the bowl`. Press `Ctrl+C` to end the rollout, then follow the prompts to save it. Enter `q` at the instruction prompt to quit.

Videos are saved under `test_videos/`. With `DEBUG_MASKS=true` before starting the server, saving an episode also writes `<episode>_mask.mp4` with the inference mask overlay and `<episode>_masks.npz` with paired masks and wrist images. See the setup guide for calibration, gripper checks, and troubleshooting.

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
