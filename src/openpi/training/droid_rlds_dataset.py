"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum, auto
import functools
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

import openpi.shared.download as download


def _to_numpy(value):
    """Recursively convert a nested dict of TF arrays into numpy arrays.

    `np.asarray` does not copy: the arrays alias TF memory and are read-only.
    Only leaves a downstream transform mutates in place need a writable copy;
    in the training input chain that is solely `actions` (DeltaActions does
    `actions[..., :dims] -= ...`). The image/mask tensors — by far the largest
    leaves — are never written in place, so copying them here is pure overhead.
    `__iter__` makes the single needed `actions` copy.
    """
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    return np.asarray(value)


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


def _decode_images_batch(batch: dict, load_embodiment_masks: bool = False) -> dict:
    """Decode encoded image bytes in a batched sample.

    Images are stored as encoded JPEG/PNG bytes in RLDS and decoded lazily here
    (after any skip) so that fast-forwarding the data pipeline on resume does not
    pay the cost of image decoding for discarded batches.
    """
    obs = batch["observation"]

    def decode_rgb(encoded):
        return tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8, channels=3)

    def decode_mask(encoded):
        return tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8, channels=1)

    obs["image"] = tf.map_fn(
        decode_rgb, obs["image"], fn_output_signature=tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8)
    )
    obs["wrist_image"] = tf.map_fn(
        decode_rgb, obs["wrist_image"], fn_output_signature=tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8)
    )
    mask_keys = ["wrist_image_left_gripper_mask"]
    if load_embodiment_masks:
        mask_keys += [
            "wrist_image_left_gripper_mask_umi",
            "wrist_image_left_gripper_mask_sharpa",
        ]
    for mask_key in mask_keys:
        obs[mask_key] = tf.map_fn(
            decode_mask,
            obs[mask_key],
            fn_output_signature=tf.TensorSpec(shape=[None, None, 1], dtype=tf.uint8),
        )
    return batch


VAL_RATIO = 0.05
NUM_BUCKETS = 1000


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # Hack to not import tf. uses tf.data.AUTOTUNE
        num_parallel_calls: int = -1,
        decode_images: bool = True,
        split: str = "train",
        seed: int | None = None,
        skip_batches: int = 0,
        load_embodiment_masks: bool = False,
    ):
        import dlimp as dl
        import tensorflow as tf

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Set global TF seed so that all random ops in the pipeline (shuffle buffer,
        # any tf.random.* calls in traj_map, etc.) are fully deterministic.
        # Without this, tf.data.Dataset.shuffle with only an op-level seed still
        # depends on the global random state at construction time, causing different
        # runs to produce different sequences even with the same op seed.
        if seed is not None:
            tf.random.set_seed(seed)

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        # Store params needed by _build_filter_table and _prepare_single_dataset.
        self._data_dir = data_dir
        self._action_chunk_size = action_chunk_size
        self._action_space = action_space
        self._num_parallel_reads = num_parallel_reads
        self._num_parallel_calls = num_parallel_calls
        self._shuffle_buffer_size = shuffle_buffer_size
        self._load_embodiment_masks = load_embodiment_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.decode_images = decode_images
        self._seed = seed

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        weights = [dataset.weight for dataset in datasets]

        # Build filter tables once per dataset config (shared between train and val pipelines).
        filter_tables = [self._build_filter_table(d) for d in datasets]

        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.split = split
        _val_threshold = int(round(VAL_RATIO * NUM_BUCKETS))

        # file_path alone is unique per episode (verified on droid_100: 100 episodes,
        # 100 distinct file_path values). Using it directly keeps hash keys readable.
        def train_hash_filter(traj):
            episode_key = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.to_hash_bucket_fast(episode_key, NUM_BUCKETS) >= _val_threshold

        def val_hash_filter(traj):
            episode_key = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.to_hash_bucket_fast(episode_key, NUM_BUCKETS) < _val_threshold

        hash_filter = train_hash_filter if split == "train" else val_hash_filter

        ds = dl.DLataset.sample_from_datasets(
            [self._prepare_single_dataset(d, filter_tables[i], hash_filter) for i, d in enumerate(datasets)],
            weights=weights,
        )

        if shuffle:
            ds = ds.shuffle(self._shuffle_buffer_size, seed=seed)

        # Build pipeline: batch → skip (cheap, images still encoded) → decode
        # images → prefetch.
        #
        # with_ram_budget(N) sets a soft cap of N GB on the RAM tf.data's
        # autotuner may allocate for its internal buffers — chiefly the
        # `.map` parallelism and the `.prefetch` queue. It does NOT bound the
        # shuffle buffer (that 250k-traj buffer of encoded bytes is a separate,
        # much larger allocation). A decoded batch is ~180 MB
        # (256 * 2 * 180*320*3 for the two RGB views, plus masks), so the old
        # 1 GB budget let the autotuner hold only ~5 decoded batches *total*
        # across map+prefetch — too tight to actually prefetch ahead. 8 GB is
        # ~40 decoded batches of headroom: enough for a deep prefetch, still
        # negligible against the cluster nodes' RAM and the shuffle buffer.
        pipeline = ds.batch(batch_size).with_ram_budget(8)
        if split == "train" and skip_batches > 0:
            logging.info(f"Skipping {skip_batches} batches to approximate resume position...")
            pipeline = pipeline.skip(skip_batches)

        if decode_images:
            pipeline = pipeline.map(
                functools.partial(_decode_images_batch, load_embodiment_masks=load_embodiment_masks),
                num_parallel_calls=num_parallel_calls,
            )

        # Decode several batches ahead on tf.data's C++ threads so the Python
        # consumer (transform chain + host->device) isn't waiting on decode.
        pipeline = pipeline.prefetch(tf.data.AUTOTUNE)

        self.dataset = pipeline

    def _build_filter_table(self, dataset_cfg: RLDSDataset):
        """Build the StaticHashTable for frame-level filtering. Called once per dataset config."""
        filter_dict_path = dataset_cfg.filter_dict_path
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []
            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True)
        return table

    def _prepare_single_dataset(self, dataset_cfg: RLDSDataset, filter_table, hash_filter_fn=None):
        """Build a frame-level TF dataset for one RLDSDataset config."""
        import dlimp.dataset as _dlimp_dataset
        import tensorflow_datasets as tfds

        # Unpack instance params into locals for use in TF closures below.
        data_dir = self._data_dir
        shuffle = self.shuffle
        seed = self._seed
        action_space = self._action_space
        action_chunk_size = self._action_chunk_size
        num_parallel_reads = self._num_parallel_reads
        num_parallel_calls = self._num_parallel_calls
        load_embodiment_masks = self._load_embodiment_masks

        ds_name, version = dataset_cfg.name, dataset_cfg.version
        builder = tfds.builder(ds_name, data_dir=data_dir, version=version)

        # Replicate dl.DLataset.from_rlds but pass shuffle_seed into ReadConfig so the
        # shard-level interleave order is deterministic. from_rlds doesn't expose this.
        dataset = _dlimp_dataset._wrap(builder.as_dataset, False)(
            # Always "train" because thats how the data is saved.
            # This is different from our self.split train / val splits.
            split="train",
            shuffle_files=shuffle,
            decoders={"steps": tfds.decode.SkipDecoding()},
            read_config=tfds.ReadConfig(
                skip_prefetch=True,
                num_parallel_calls_for_interleave_files=num_parallel_reads,
                interleave_cycle_length=num_parallel_reads,
                shuffle_seed=seed,
            ),
        )._apply_options()
        dataset = dataset.enumerate().traj_map(_dlimp_dataset._broadcast_metadata_rlds)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # Drop episodes without optimized cam_to_gripper extrinsics (outliers were filtered out upstream).
        dataset = dataset.filter(
            lambda traj: tf.equal(traj["traj_metadata"]["episode_metadata"]["extrinsics_found"][0], 1)
        )

        # Apply the train or val episode-level hash filter if provided.
        if hash_filter_fn is not None:
            dataset = dataset.filter(hash_filter_fn)

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            traj = tf.nest.map_structure(
                lambda x: tf.cast(x, tf.float32) if x.dtype == tf.float64 else x, traj
            )

            # Build the per-step action prefix; gripper(1) is appended for every space.
            if action_space == DroidActionSpace.JOINT_POSITION:
                # Joint position is preferred since it's easy to simulate.
                action_prefix = traj["action_dict"]["joint_position"]
            elif action_space == DroidActionSpace.JOINT_VELOCITY:
                action_prefix = traj["action_dict"]["joint_velocity"]
            else:
                raise ValueError(f"Unsupported action_space: {action_space}")

            actions = tf.concat((action_prefix, traj["action_dict"]["gripper_position"]), axis=-1)
            # NOTE: Michael - Exterior camera view pre-selected once per episode during preprocessing
            # (one of exterior_image_1_left / exterior_image_2_left, chosen randomly).
            exterior_img = traj["observation"]["exterior_image_left"]
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = filter_table.lookup(step_id)

            observation = {
                "image": exterior_img,
                "wrist_image": wrist_img,
                "joint_position": traj["observation"]["joint_position"],
                "gripper_position": traj["observation"]["gripper_position"],
                "wrist_image_left_gripper_mask": traj["observation"]["wrist_image_left_gripper_mask"],
            }
            if load_embodiment_masks:
                observation["wrist_image_left_gripper_mask_umi"] = traj["observation"][
                    "wrist_image_left_gripper_mask_umi"
                ]
                observation["wrist_image_left_gripper_mask_sharpa"] = traj["observation"][
                    "wrist_image_left_gripper_mask_sharpa"
                ]

            return {
                "actions": actions,
                "observation": observation,
                "prompt": instruction,
                "step_id": step_id,
                "file_path": traj["traj_metadata"]["episode_metadata"]["file_path"],
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        return dataset

    def __iter__(self):
        for batch in self.dataset.as_numpy_iterator():
            # _to_numpy aliases TF memory (read-only, zero-copy). Only `actions`
            # is mutated in place downstream (DeltaActions), so it alone gets a
            # writable copy here; the big image/mask leaves pass through uncopied.
            result = _to_numpy(batch)
            if "actions" in result:
                result["actions"] = np.array(result["actions"])
            if not self.decode_images:
                # Replace raw encoded image bytes with small dummy arrays so downstream
                # transforms don't fail.
                bs = self.batch_size
                dummy = np.zeros((bs, 16, 16, 3), dtype=np.uint8)
                dummy_mask = np.zeros((bs, 16, 16, 1), dtype=np.uint8)
                result["observation"]["image"] = dummy
                result["observation"]["wrist_image"] = dummy
                result["observation"]["wrist_image_left_gripper_mask"] = dummy_mask
                if self._load_embodiment_masks:
                    result["observation"]["wrist_image_left_gripper_mask_umi"] = dummy_mask
                    result["observation"]["wrist_image_left_gripper_mask_sharpa"] = dummy_mask
            yield result

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
