# Training

JAX training for the DROID policy configs. Serving/deployment lives in the main
[README](README.md).


## Data preprocessing

```bash
# Download the DROID RLDS dataset (episodes hosted on GCS).
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 /path/to/DROID
```

### Metadata assets (optional; precomputed copies ship in `assets/`)

This step computes metadata needed for later steps: language annotations, camera intrinsics, and zed serial numbers.
It processes them into a simpler key format used by our repo. This step is optional; we provide pre-computed metadata
in `assets/`.

```bash
# First download assets from HuggingFace.
uv run --group preprocess hf download KarlP/droid \
    droid_language_annotations.json intrinsics.json --local-dir scratch

# Run the metadata processing script
uv run --group preprocess python preprocessing/preprocess_metadata.py \
    --annotations-path scratch/droid_language_annotations.json \
    --intrinsics-path scratch/intrinsics.json
```

### Wrist camera extrinsics (optional; precomputed copies ship in `assets/`)

This step recomputes the per-episode wrist `cam_to_gripper` extrinsics via Silhouette Calibration. This step is
optional; we provide precomputed extrinsics in `assets/droid_wrist_extrinsics.json`.

```bash
uv run --group preprocess python preprocessing/preprocess_wrist_extrinsics.py \
    --data-dir /path/to/DROID --output assets/droid_wrist_extrinsics.json
```

### Process DROID data

Our training code relies on additional features such as precomputed gripper masks and flags to exlcude certain episodes.
Run the preprocessing script to process your DROID dataset (takes several hours).

```bash
uv run --group preprocess python preprocessing/preprocess_data.py \
    --data-dir /path/to/DROID --n-workers 24
```


## Normalization stats

Precomputed norm stats are checked in under
`assets/norm_stats/<config>/droid/norm_stats.json` and loaded automatically at train time —
you don't need to compute anything for the existing configs.

### Recomputing (only if the data or config changes)

```bash
uv run --group train python preprocessing/compute_norm_stats.py \
    --config-name <config> --max-frames 1000000 --rlds-data-dir /path/to/droid_rlds
```

This writes `norm_stats.json` back to `assets/norm_stats/<config>/droid/`.


## Train

Set `CONFIG`, `EXP_NAME`, and `RLDS_DATA_DIR` at the top of `scripts/train.sh`. For the available training configs,
see `src/openpi/training/config.py`. In short:
- `pi05_full_droid_finetune_v0`: Corresponds to Pi05-droid in the paper
- `pi05_full_droid_finetune_v3-1`: Corresponds to Cloak-VLA in the paper
- `pi05_full_droid_finetune_v3-1_ik`: Inference-time only. Adds Tip-pose retargeting for running on other embodiments.

Then run the train script:

```bash
bash scripts/train.sh
```
