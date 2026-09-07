from collections.abc import Iterator, Sequence
import logging
from typing import Protocol, SupportsIndex, TypeVar

import jax

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def _apply_transforms(self, source):
        for sample in source:
            if self._is_batched:
                # Batched path: every transform in the chain is batch-aware, so we
                # feed the batched dict straight through.
                yield self._transform(sample)
            else:
                yield self._transform(sample)

    def __iter__(self):
        return self._apply_transforms(self._dataset)

    def __len__(self) -> int:
        return len(self._dataset)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    decode_images: bool = True,
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
    seed: int | None = None,
    skip_batches: int = 0,
    shuffle_buffer_size: int = 250_000,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
        decode_images=decode_images,
        split=data_config.split,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
        seed=seed,
        skip_batches=skip_batches,
        shuffle_buffer_size=shuffle_buffer_size,
        load_embodiment_masks=data_config.load_embodiment_masks,
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `preprocessing/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.train_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm, mask=data_config.norm_mask),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    skip_batches: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        skip_batches: Number of batches to skip at the start (RLDS only, for resume).
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is None:
        raise ValueError(
            "Only RLDS (DROID) datasets are supported here; set data.rlds_data_dir. "
            "The LeRobot/torch loader was dropped in the cloak migration."
        )
    return create_rlds_data_loader(
        data_config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        skip_norm_stats=skip_norm_stats,
        seed=config.seed,
        skip_batches=skip_batches,
    )


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    seed: int | None = None,
    skip_batches: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see README_train.md
    (the `train` uv dependency group).

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        seed: Random seed for deterministic shuffle ordering.
        skip_batches: Number of batches to skip at the start of the pipeline (for resume).
    """
    dataset = create_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        shuffle=shuffle,
        seed=seed,
        skip_batches=skip_batches,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
    )
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def _to_sharded(self, source, num_batches=None):
        def _shard_one(x):
            # Pass non-numeric leaves through unchanged. step_id / file_path
            # passthroughs arrive here as object arrays from h5py, but
            # IterableTransformedDataset's per-sample-then-stack path
            # collapses lists of `bytes` into fixed-length |SN arrays — so
            # we gate on dtype.kind, not the literal `object` dtype.
            if hasattr(x, "dtype") and x.dtype.kind not in "biufc":
                return x
            return jax.make_array_from_process_local_data(self._sharding, x)

        for i, batch in enumerate(source):
            if num_batches is not None and i >= num_batches:
                return
            yield jax.tree.map(_shard_one, batch)

    def __iter__(self):
        return self._to_sharded(self._dataset, num_batches=self._num_batches)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    # Keys plucked out of the post-transform batch and exposed as the third
    # element of every yield, alongside (Observation, actions). These are
    # data-pipeline metadata that don't belong inside `Observation` (which is
    # a JAX-traced pytree of numeric arrays). file_path is used by the train
    # loop to track seen episodes across resumes; step_id/prompt are kept
    # for per-example tagging.
    _EXTRA_KEYS = ("step_id", "prompt", "file_path")

    def _to_model_input(self, source):
        for batch in source:
            extra = {k: batch[k] for k in self._EXTRA_KEYS if k in batch}
            yield _model.Observation.from_dict(batch), batch["actions"], extra

    def __iter__(self):
        return self._to_model_input(self._data_loader)
