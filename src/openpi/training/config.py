"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.policies.gripper_mask_transform as gripper_mask_transform
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Training-only transforms (e.g. data augmentation). Spliced into the training data loader
    # chain between `repack_transforms.inputs` and `data_transforms.inputs`. Never applied at
    # inference time — the inverse of `InferenceOptions`, which is serve-only.
    train_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # Optional per-dim bool mask for `Normalize` / `Unnormalize`. Where False,
    # the dim passes through unchanged. Used for v4 to skip rot6d action / state
    # dims (UMI's identity-rot6d convention). See `transforms.Normalize.mask`.
    norm_mask: Sequence[bool] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    # Which split to load: "train" or "val".
    split: str = "train"
    # If True, the RLDS loader also reads/decodes the per-embodiment wrist
    # gripper masks (`wrist_image_left_gripper_mask_{umi,sharpa}`). Off by
    # default so configs that don't need them pay no decode cost (and so
    # datasets lacking these features still load). Set by RLDSDroidDataConfig
    # when mask_augmentation_type == "embodiment".
    load_embodiment_masks: bool = False
    # Frame-level shuffle-buffer size for the RLDS loader. The first batch isn't
    # emitted until this many frames are buffered, so debug/smoke-test configs set
    # it small (e.g. 10_000) for a fast first batch. Ignored when shuffle is off.
    shuffle_buffer_size: int = 250_000


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            debug_mode=model_config.debug_mode,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            debug_mode=model_config.debug_mode,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    split: str = "train"
    use_droid_100: bool = False
    # If true, qpos (joint_position + gripper_position) is included in state/action.
    use_qpos: bool = True

    # If True, apply training-only augmentation of the wrist-camera gripper mask.
    # See `gripper_mask_transform.MaskAugmentation` for behavior.
    mask_augmentation: bool = False
    # Which mask-augmentation family to use when `mask_augmentation` is True.
    # "blob" = the capsule-add / circle-remove augmentation (v3.1).
    # "embodiment" = embodiment-style mask augmentation (v3.2).
    mask_augmentation_type: Literal["blob", "embodiment"] = "blob"

    # Frame-level shuffle-buffer size (see DataConfig.shuffle_buffer_size). Small
    # values give a fast first batch for debug/smoke tests.
    shuffle_buffer_size: int = 250_000

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_droid_100:
            datasets = (droid_rlds_dataset.RLDSDataset(name="droid_100", version="1.0.0", weight=1.0),)
        else:
            datasets = self.datasets
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    },
                    # Gripper masks are optional — pass through if present.
                    # step_id passthrough for per-example tagging.
                    # file_path passthrough so the train loop can track unique episodes seen.
                    passthrough_keys=(
                        "observation/wrist_image_left_gripper_mask",
                        "observation/wrist_image_left_gripper_mask_sharpa",
                        "step_id",
                        "file_path",
                    ),
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                # Training: data arrives batched from RLDS → MaybeBatch is a no-op.
                # Inference: data arrives per-sample from the client → MaybeBatch adds the
                # leading batch dim so DroidInputs (batched-only) sees what it expects.
                _transforms.MaybeBatch(),
                droid_policy.DroidInputs(
                    model_type=model_config.model_type,
                    use_qpos=self.use_qpos,
                ),
            ],
            outputs=[
                droid_policy.DroidOutputs(
                    use_qpos=self.use_qpos,
                )
            ],
        )

        norm_mask: tuple[bool, ...] | None = None
        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            # AbsoluteActions (outputs) is NOT used during training — that only applies
            # data_transforms.inputs. It is only wired in at real-world inference time via
            # policy_config.create_trained_policy, which hands data_transforms.outputs to Policy; the
            # Policy then runs them on sampled model outputs to convert predictions back to absolute
            # joint positions before sending them to the robot.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt="do something")(model_config)

        train_transforms = _transforms.Group()
        if self.mask_augmentation:
            train_transforms = _transforms.Group(
                inputs=[gripper_mask_transform.MaskAugmentation(augmentation_type=self.mask_augmentation_type)]
            )

        # assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."  # GUY: I commented this out because irrelevant to deployment.

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            train_transforms=train_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            norm_mask=norm_mask,
            datasets=datasets,
            split=self.split,
            load_embodiment_masks=self.mask_augmentation and self.mask_augmentation_type == "embodiment",
            shuffle_buffer_size=self.shuffle_buffer_size,
        )


@dataclasses.dataclass(frozen=True)
class InferenceOptions:
    """Serve-only knobs read by `_build_inference_transforms` in policy_config.

    Lives on `TrainConfig` (not `DataConfigFactory`) because these flags pick
    which renderers/rewrites to splice into the inference chain — they have
    no effect at training time.
    """

    # If true, render + inject the deploy embodiment's wrist end-effector mask at
    # inference. Only meaningful for a checkpoint trained with patch masking
    # (v3-1). The embodiment (robotiq/sharpa/umi/yam) is chosen at serve time via
    # serve_policy.py's --embodiment, not here — so this flag is embodiment-generic.
    use_mask: bool = False
    # If true, splice the embodiment's <Emb>ToRobotiqRewrite into the input chain
    # (before DroidInputs) and its <Emb>IKTransform into the output chain (after
    # DroidOutputs/AbsoluteActions), retargeting the (T, 8) Robotiq action chunk to
    # the deploy robot via Robotiq FK + mink IK. No-op for embodiment=robotiq
    # (native — no retargeting).
    use_ik: bool = False

    # Inference-only safety-margin dilation applied to the rendered wrist-image
    # mask before it reaches the model. The mask combine itself stays
    # training-equivalent (matches preprocess_data.compute_mask); this knob
    # adds a single morphological dilation on top of OR(gripper, base) with a
    # square kernel of side ``gripper_dilation_kernel_size``. Default 16
    # matches the safety margin previously hardcoded in combine_masks. Set to
    # 0 to disable the inference-time dilation entirely.
    gripper_dilation_kernel_size: int = 16


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "hand_openpi_proj"
    # Wandb entity (team or user). If None, uses the default entity from wandb config.
    wandb_entity: str = "hand_openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Serve-only knobs. See `InferenceOptions`.
    inference: InferenceOptions = dataclasses.field(default_factory=InferenceOptions)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets/norm_stats"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints. <= 0 disables checkpoint saving
    # entirely (including the final step) — used by the debug smoke test.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


"""
Training / Robotiq Configs
"""

_PI05_FULL_DROID_FINETUNE_V0 = TrainConfig(
    # This config is for fine-tuning pi05 on the *full* DROID dataset.
    # We use RLDS data loading to make training on this large dataset tractable.
    # For fine-tuning on your own DROID dataset, see below.
    name="pi05_full_droid_finetune_v0",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
    ),
    data=RLDSDroidDataConfig(
        repo_id="droid",
        # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
        # Defaults to None; pass --data.rlds-data-dir at the CLI to override.
        action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        assets=AssetsConfig(
            assets_dir="assets/norm_stats/pi05_full_droid_finetune_v0",
            asset_id="droid",
        ),
        use_qpos=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=100_000,
        decay_lr=5e-5,
    ),
    num_train_steps=100_000,
    batch_size=256,
    log_interval=100,
    save_interval=5000,
    keep_period=20_000,
    num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
)


# v3-1: the full cloak method — majority patch masking PLUS training-time blob
# mask augmentation. Augmentation is train-only and hard-blocked at serve time
# (see serve_policy.py), so every v3-1 checkpoint is trained-with-aug and
# served-without. norm_stats match v0 (augmentation only alters the wrist mask
# image, never state/actions).
_PI05_FULL_DROID_FINETUNE_V3_1 = dataclasses.replace(
    _PI05_FULL_DROID_FINETUNE_V0,
    name="pi05_full_droid_finetune_v3-1",
    model=dataclasses.replace(_PI05_FULL_DROID_FINETUNE_V0.model, patch_masking_strategy="majority"),
    data=dataclasses.replace(
        _PI05_FULL_DROID_FINETUNE_V0.data,
        assets=AssetsConfig(
            assets_dir="assets/norm_stats/pi05_full_droid_finetune_v3-1",
            asset_id="droid",
        ),
        use_qpos=True,
        mask_augmentation=True,
        mask_augmentation_type="blob",
    ),
)


"""
Debug Configs — fast end-to-end training smoke test with no hardware/GPU demands:
a tiny "dummy" model (no PaliGemma/action-expert weights), no pretrained loader,
tiny batch/steps, wandb off, on the 100-episode droid_100 dataset. Actions are
meaningless — this only checks the train loop + data pipeline run and save.
"""

_PI05_FULL_DROID_FINETUNE_V0_DEBUG = dataclasses.replace(
    _PI05_FULL_DROID_FINETUNE_V0,
    name="pi05_full_droid_finetune_v0_debug",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=16,
    ),
    # Full preprocessed droid/1.0.1; filter_dict_path=None keeps all frames (no
    # gs:// fetch); small shuffle buffer for a fast first batch.
    data=dataclasses.replace(
        _PI05_FULL_DROID_FINETUNE_V0.data,
        datasets=(droid_rlds_dataset.RLDSDataset(name="droid", version="1.0.1", weight=1.0, filter_dict_path=None),),
        shuffle_buffer_size=10_000,
    ),
    weight_loader=weight_loaders.NoOpWeightLoader(),
    num_train_steps=100,
    log_interval=10,
    save_interval=10_000,
    batch_size=4,
    wandb_enabled=False,
)


# Masking-only (no mask_augmentation): the debug configs run on the small,
# *vanilla* droid_100 dataset, which has no precomputed gripper-mask streams.
# patch_masking_strategy gracefully no-ops when the mask is absent (DroidInputs
# only wires pixel_mask when present), but MaskAugmentation hard-requires the
# mask stream — so it's left off here. The full v3-1 (with blob augmentation)
# trains on the mask-preprocessed DROID dataset.
_PI05_FULL_DROID_FINETUNE_V3_1_DEBUG = dataclasses.replace(
    _PI05_FULL_DROID_FINETUNE_V0_DEBUG,
    name="pi05_full_droid_finetune_v3-1_debug",
    model=dataclasses.replace(_PI05_FULL_DROID_FINETUNE_V0_DEBUG.model, patch_masking_strategy="majority"),
    data=dataclasses.replace(
        _PI05_FULL_DROID_FINETUNE_V0_DEBUG.data,
        assets=AssetsConfig(assets_dir="assets/norm_stats/pi05_full_droid_finetune_v3-1", asset_id="droid"),
    ),
)


"""
Deployment (cross-embodiment) config
"""

# The v3-1 (full-cloak) checkpoint served to a cross-embodiment client. One
# config serves every robot: the embodiment (robotiq/sharpa/umi/yam) is chosen at
# serve time via serve_policy.py's --embodiment. use_mask renders that robot's
# wrist end-effector mask; use_ik FK-rewrites the incoming proprio into the
# Robotiq-equivalent state the policy was trained on and IK-retargets the (T, 8)
# Robotiq action chunk to the robot (no-op retargeting for native robotiq). Same
# checkpoint + norm stats as v3-1 — only the inference chain differs.
_PI05_FULL_DROID_FINETUNE_V3_1_IK = dataclasses.replace(
    _PI05_FULL_DROID_FINETUNE_V3_1,
    name="pi05_full_droid_finetune_v3-1_ik",
    inference=dataclasses.replace(
        _PI05_FULL_DROID_FINETUNE_V3_1.inference,
        use_mask=True,
        use_ik=True,
    ),
)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # DROID pi0.5 fine-tuning bases.
    _PI05_FULL_DROID_FINETUNE_V0,
    _PI05_FULL_DROID_FINETUNE_V3_1,
    # Debug (tiny model, droid_100) — training smoke test.
    _PI05_FULL_DROID_FINETUNE_V0_DEBUG,
    _PI05_FULL_DROID_FINETUNE_V3_1_DEBUG,
    # Cross-embodiment deployment config (robot chosen at serve time via --embodiment).
    _PI05_FULL_DROID_FINETUNE_V3_1_IK,
]
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
