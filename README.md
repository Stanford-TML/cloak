# Cloak

[![Project Page](https://img.shields.io/badge/Project_Page-blue)](https://tml.stanford.edu/cloak/)


## Calibrated DROID wrist extrinsics

We release per-episode wrist-camera extrinsics for [DROID](https://droid-dataset.github.io/) episodes — the camera's pose relative to the end-effector ("cam_to_gripper"). Each is recovered by our **Silhouette Calibration** algorithm, which produces more accurate alignment to sim than the extrinsics shipped with DROID.

- `assets/droid_wrist_extrinsics.json` — JSON containing the 6-DoF camera pose relative to the Franka attachment_site.
- `examples/render_extrinsics.py` — Minimal example of using the extrinsics to render the wrist view.

### Regenerating the extrinsics

If you would like to rerun the optimization via Silhouette Calibration., use `scripts/preprocess_wrist_extrinsics.py`.

```
uv run python scripts/preprocess_wrist_extrinsics.py --data-dir /path/to/DROID --output out.json
```
