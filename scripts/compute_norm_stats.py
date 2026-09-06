"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int,
) -> tuple[_data_loader.Dataset, int]:
    # Shrink the shuffle buffer relative to the training default (250k). At ~30 KB per
    # pre-decode sample (encoded JPEGs dominate), 250k overflows the 64G slurm allocation;
    # 50k still mixes well across episodes and stays comfortably under the limit.
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size,
        shuffle=True, decode_images=False, seed=0, shuffle_buffer_size=50_000,
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    num_batches = max_frames // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int, rlds_data_dir: str | None = None):
    config = _config.get_config(config_name)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, rlds_data_dir=rlds_data_dir))
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is None:
        raise ValueError(
            "Only RLDS (DROID) datasets are supported; pass --rlds-data-dir or set it in the config."
        )
    data_loader, num_batches = create_rlds_dataloader(
        data_config, config.model.action_horizon, config.batch_size, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
