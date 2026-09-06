import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies.gripper_mask_transform import get_mask_transform as get_gripper_mask_transform
from openpi.policies.gripper_mask_transform import get_sharpa_mask_transform
from openpi.policies.gripper_mask_transform import InjectGripperMask
import openpi.policies.policy as _policy
from openpi.policies.sharpa_ik_transform import get_ik_transform as get_sharpa_ik_transform
from openpi.policies.sharpa_ik_transform import get_input_rewrite_transform as get_sharpa_input_rewrite_transform
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as transforms


def _resolve_checkpoint_dir(checkpoint_dir: pathlib.Path | str) -> pathlib.Path:
    """Resolve a path to the directory that actually holds ``params/`` (+ ``assets/``).

    Accepts either that servable leaf directory directly, or a parent that
    contains it nested somewhere below — e.g. the raw orbax ``CheckpointManager``
    layout ``<exp>/<step>/params``. When several checkpoints are present, returns
    the one with the highest numeric step.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if (checkpoint_dir / "params").is_dir():
        return checkpoint_dir

    # `rglob` recursively finds every `params` entry below the given path; each
    # one's parent is a candidate servable dir.
    candidates = [p.parent for p in checkpoint_dir.rglob("params") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No servable checkpoint found under {checkpoint_dir}: expected a directory "
            "containing a 'params/' subdirectory, either at this path or nested within it."
        )
    # Prefer the highest numeric step dir (e.g. 100000); else lexicographic.
    return max(
        candidates,
        key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else -1, str(p)),
    )


def _repr_transform(t, max_len: int = 200) -> str:
    """Human-readable repr for a transform: uses ``repr(t)`` if the class
    overrides ``__repr__`` (true for frozen dataclasses); otherwise builds
    ``ClassName(public_attr=value, ...)`` from ``vars(t)``. Truncated so
    fields like ``Normalize.norm_stats`` don't dominate the output.
    """
    cls = type(t)
    if cls.__repr__ is not object.__repr__:
        s = repr(t)
    else:
        attrs = {k: v for k, v in vars(t).items() if not k.startswith("_")}
        body = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        s = f"{cls.__name__}({body})"
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def get_inference_mask_transform(train_config: _config.TrainConfig) -> InjectGripperMask | None:
    """Return the singleton mask transform ``_build_inference_transforms`` would
    splice into the input chain, or ``None`` if the model doesn't use patch
    masking. Idempotent — repeat calls return the same instance.
    """
    strategy = getattr(train_config.model, "patch_masking_strategy", None)
    if strategy is None:
        return None

    inf = train_config.inference
    if inf.use_sharpa_mask:
        getter = get_sharpa_mask_transform
    else:
        getter = get_gripper_mask_transform

    return getter(
        patch_masking_strategy=strategy,
        gripper_dilation_kernel_size=inf.gripper_dilation_kernel_size,
    )


def _build_inference_transforms(train_config: _config.TrainConfig) -> transforms.Group:
    """Assemble serve-only renderers/rewrites for this checkpoint.

    Input order: mask, input_rewrite — mask reads raw client proprio so it
    runs before the Sharpa->Robotiq rewrite.
    """
    assert isinstance(
        train_config.data, _config.RLDSDroidDataConfig
    ), "_build_inference_transforms only supports RLDSDroidDataConfig"
    inf = train_config.inference

    inputs: list = []

    # Mask renderer.
    mask_transform = get_inference_mask_transform(train_config)
    if mask_transform is not None:
        inputs.append(mask_transform)

    # Input rewrite (Sharpa -> Robotiq state).
    if inf.apply_sharpa_ik:
        inputs.append(get_sharpa_input_rewrite_transform())

    # Output IK chain.
    outputs: list = []
    if inf.apply_sharpa_ik:
        outputs.append(get_sharpa_ik_transform())

    return transforms.Group(inputs=inputs, outputs=outputs)


def create_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str | None = None,
    *,
    random_weights: bool = False,
    rng_seed: int = 0,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a serving policy for ``train_config``.

    The transform chain (input rewrite + mask renderer + norm + model transforms
    + output IK) is identical whether weights come from a checkpoint or are
    random. Only the weights and the norm stats differ, so both paths share this
    one function.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load params + norm stats from. Required
            unless ``random_weights`` is True (in which case it is ignored).
        random_weights: If True, initialize the model with random weights and use
            no norm stats — ``Normalize``/``Unnormalize`` become no-ops. For
            testing the full serve pipeline without a checkpoint; actions are
            meaningless.
        rng_seed: Seed for the random-weight initialization (only used when
            ``random_weights`` is True).
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not
            provided, the default kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the
            prompt into the input data if it doesn't already exist.
    """
    inference_transforms = _build_inference_transforms(train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if random_weights:
        model = train_config.model.create(jax.random.key(rng_seed))
        norm_stats = None
        logging.warning(
            "Serving RANDOM-WEIGHT policy for %r — actions are meaningless (testing only).", train_config.name
        )
    else:
        assert checkpoint_dir is not None
        checkpoint_dir = download.maybe_download(str(checkpoint_dir))
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
        logging.info("Serving checkpoint %s", checkpoint_dir)
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
        norm_stats_dir = checkpoint_dir / "assets" / data_config.asset_id
        norm_stats = _normalize.load(norm_stats_dir)
        logging.info("Loaded norm stats from %s", norm_stats_dir)

    input_transforms = [
        *inference_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm, mask=data_config.norm_mask),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm, mask=data_config.norm_mask),
        *data_config.data_transforms.outputs,
        *inference_transforms.outputs,
    ]

    print(f"Inference input transforms (in order) for {train_config.name!r}:")
    for i, t in enumerate(input_transforms):
        print(f"  [{i}] {_repr_transform(t)}")
    print(f"Inference output transforms (in order) for {train_config.name!r}:")
    for i, t in enumerate(output_transforms):
        print(f"  [{i}] {_repr_transform(t)}")

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
