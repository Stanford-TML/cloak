from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


SEEN_EPISODES_FILENAME = "seen_episodes.json"


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    train_metadata: dict | None = None,
    seen_episodes: set[str] | None = None,
):
    # Snapshot the seen-episodes set now so the async save thread doesn't race
    # the train loop's continued mutations.
    seen_episodes_snapshot = sorted(seen_episodes) if seen_episodes is not None else None

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        # Save training metadata for resuming.
        with (directory / "train_metadata.json").open("w") as f:
            json.dump(train_metadata or {}, f)
        # Save the set of unique episodes seen so far at the step-dir top level
        # (sibling of `assets/`, `params/`, `train_state/`).
        if seen_episodes_snapshot is not None:
            with (directory.parent / SEEN_EPISODES_FILENAME).open("w") as f:
                json.dump(seen_episodes_snapshot, f)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(int(state.step), items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: training_utils.TrainState | None = None,
    step: int | None = None,
) -> training_utils.TrainState:
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        items = {
            "train_state": train_state,
            "params": {"params": params},
        }
        restore_kwargs = None
        if state_sharding is not None:
            # Providing target shardings lets orbax reshard to the current mesh,
            # which is required when the checkpoint was saved under a different topology.
            ts_sharding, p_sharding = _split_params(state_sharding)
            restore_kwargs = {
                "train_state": {
                    "restore_args": ocp.checkpoint_utils.construct_restore_args(train_state, ts_sharding),
                },
                "params": {
                    "restore_args": ocp.checkpoint_utils.construct_restore_args(
                        {"params": params}, {"params": p_sharding}
                    ),
                },
            }
        restored = checkpoint_manager.restore(step, items=items, restore_kwargs=restore_kwargs)
    return _merge_params(restored["train_state"], restored["params"])


def restore_seen_episodes(checkpoint_manager: ocp.CheckpointManager) -> set[str]:
    """Load the set of episode keys persisted alongside the latest checkpoint.

    Returns an empty set if no checkpoint exists yet or the file is missing
    (e.g., resuming from a checkpoint that predates this feature).
    """
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return set()
    path = checkpoint_manager.directory / str(latest_step) / SEEN_EPISODES_FILENAME
    if not path.exists():
        return set()
    with path.open() as f:
        return set(json.load(f))


def restore_metadata(checkpoint_manager: ocp.CheckpointManager) -> dict:
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return {}
    
    metadata_path = checkpoint_manager.directory / str(latest_step) / "assets" / "train_metadata.json"
    if not metadata_path.exists():
        return {}
    
    with metadata_path.open() as f:
        metadata = json.load(f)
        
    return metadata


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
