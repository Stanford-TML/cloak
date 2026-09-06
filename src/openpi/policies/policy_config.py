import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies.gripper_mask_transform import get_mask_transform as get_robotiq_mask_transform
from openpi.policies.gripper_mask_transform import get_sharpa_mask_transform
from openpi.policies.gripper_mask_transform import get_umi_mask_transform
from openpi.policies.gripper_mask_transform import get_yam_mask_transform
from openpi.policies.gripper_mask_transform import InjectGripperMask
import openpi.policies.policy as _policy
from openpi.policies import sharpa_ik_transform as _sharpa_ik
from openpi.policies import umi_ik_transform as _umi_ik
from openpi.policies import yam_ik_transform as _yam_ik
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as transforms


# embodiment -> (mask_transform_getter, ik_getter, input_rewrite_getter). The IK
# and input-rewrite getters are None for robotiq (native — no cross-embodiment
# retargeting; a masked checkpoint still renders the robotiq wrist mask). This is
# the single dispatch point that turns the serve-time --embodiment choice into
# the right renderers/rewrites, so InferenceOptions stays embodiment-agnostic.
_EMBODIMENT_TRANSFORMS = {
    "robotiq": (get_robotiq_mask_transform, None, None),
    "sharpa": (get_sharpa_mask_transform, _sharpa_ik.get_ik_transform, _sharpa_ik.get_input_rewrite_transform),
    "umi": (get_umi_mask_transform, _umi_ik.get_ik_transform, _umi_ik.get_input_rewrite_transform),
    "yam": (get_yam_mask_transform, _yam_ik.get_ik_transform, _yam_ik.get_input_rewrite_transform),
}
EMBODIMENTS = tuple(_EMBODIMENT_TRANSFORMS)


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


def get_inference_mask_transform(
    train_config: _config.TrainConfig, embodiment: str = "robotiq"
) -> InjectGripperMask | None:
    """Return the singleton mask transform ``_build_inference_transforms`` would
    splice into the input chain, or ``None`` if masking is off for this serve
    (model not trained with patch masking, or ``inference.use_mask`` is False).
    The renderer is selected by ``embodiment``. Idempotent.
    """
    strategy = getattr(train_config.model, "patch_masking_strategy", None)
    inf = train_config.inference
    if strategy is None or not inf.use_mask:
        return None

    mask_getter = _EMBODIMENT_TRANSFORMS[embodiment][0]
    return mask_getter(
        patch_masking_strategy=strategy,
        gripper_dilation_kernel_size=inf.gripper_dilation_kernel_size,
    )


def _build_inference_transforms(
    train_config: _config.TrainConfig, embodiment: str = "robotiq"
) -> transforms.Group:
    """Assemble serve-only renderers/rewrites for this checkpoint + deploy embodiment.

    Input order: mask, input_rewrite — mask reads raw client proprio so it runs
    before the <Emb>->Robotiq rewrite. Robotiq has no IK (native), so only the
    mask (if any) is spliced.
    """
    assert isinstance(
        train_config.data, _config.RLDSDroidDataConfig
    ), "_build_inference_transforms only supports RLDSDroidDataConfig"
    if embodiment not in _EMBODIMENT_TRANSFORMS:
        raise ValueError(f"Unknown embodiment {embodiment!r}; expected one of {EMBODIMENTS}.")
    inf = train_config.inference
    _, ik_getter, input_rewrite_getter = _EMBODIMENT_TRANSFORMS[embodiment]

    inputs: list = []

    # Mask renderer (embodiment-specific).
    mask_transform = get_inference_mask_transform(train_config, embodiment)
    if mask_transform is not None:
        inputs.append(mask_transform)

    # Input rewrite (<Emb> -> Robotiq state) + output IK chain. None for robotiq.
    if inf.use_ik and input_rewrite_getter is not None:
        inputs.append(input_rewrite_getter())
    outputs: list = []
    if inf.use_ik and ik_getter is not None:
        outputs.append(ik_getter())

    return transforms.Group(inputs=inputs, outputs=outputs)


def create_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str | None = None,
    *,
    embodiment: str = "robotiq",
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
    inference_transforms = _build_inference_transforms(train_config, embodiment)
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
