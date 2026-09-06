# Training

JAX training for the DROID policy configs. Serving/deployment lives in the main
[README](README.md).


## Normalization stats

Precomputed norm stats are checked in under
`assets/norm_stats/<config>/droid/norm_stats.json` and loaded automatically at train time —
you don't need to compute anything for the existing configs.

### Recomputing (only if the data or config changes)

```bash
uv run --group train python scripts/compute_norm_stats.py \
    --config-name <config> --max-frames 1000000 --rlds-data-dir /path/to/droid_rlds
```

This writes `norm_stats.json` back to `assets/norm_stats/<config>/droid/`. Commit the updated
file so training and serving stay in sync.



## Train

Set `CONFIG`, `EXP_NAME`, and `RLDS_DATA_DIR` at the top of `scripts/train.sh`, then:

```bash
bash scripts/train.sh
```

Trainable base configs (`v0` = baseline, no mask; `v3-1` = the full cloak method:
majority patch-masking + training-time blob mask augmentation). The deployment
config `..._v3-1_ik` is an inference-time wrapper over the `v3-1` checkpoint, so
you only ever *train* the bases:

- `pi05_full_droid_finetune_v0`
- `pi05_full_droid_finetune_v3-1`
- `pi05_full_droid_finetune_{v0,v3-1}_debug` — tiny `dummy` model + `droid_100`, no
  pretrained weights, 100 steps, wandb off. A fast end-to-end smoke test (actions
  are meaningless). `train.sh` defaults to `v0_debug`.

Deployment (all robots share one config; pick the robot at serve time with
`--embodiment {robotiq,sharpa,umi,yam}` — see the main [README](README.md)):

- `pi05_full_droid_finetune_v3-1_ik`

For one-off overrides (batch size, steps, …) call the script directly:

```bash
uv run --group train python scripts/train.py <config> \
    --exp-name <name> --data.rlds-data-dir /path/to/droid_rlds --batch-size 64
```
