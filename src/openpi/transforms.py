from collections.abc import Callable, Sequence
import dataclasses
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


def _to_str(x) -> str:
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }

    passthrough_keys: optional tuple of flattened keys to copy through from the
    input unchanged, if they exist. Missing keys are silently skipped.
    """

    structure: at.PyTree[str]
    passthrough_keys: tuple[str, ...] = ()

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        result = jax.tree.map(lambda k: flat_item[k], self.structure)
        # Copy through optional keys if present in the input.
        for key in self.passthrough_keys:
            if key in flat_item:
                result[key] = flat_item[key]
        return result


@dataclasses.dataclass(frozen=True)
class MaybeBatch(DataTransformFn):
    """Add a leading batch dim to every leaf if the input is per-sample; no-op when already batched.

    Lets downstream transforms (which now assume batched inputs, e.g. `DroidInputs`) work in
    both the training data pipeline (already batched from RLDS) and the inference path
    (single observation from a client). Detection probes `observation/joint_position` —
    `ndim < 2` is per-sample.
    """

    def __call__(self, data: DataDict) -> DataDict:
        probe = data.get("observation/joint_position")
        if probe is None or np.asarray(probe).ndim >= 2:
            return data
        return jax.tree.map(
            lambda x: np.array([x]) if isinstance(x, str | bytes) else np.asarray(x)[None, ...],
            data,
        )


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is None:
            return data
        # Inject the default where the prompt is missing or an empty byte string
        # b"" (unannotated DROID episodes arrive as b""). np.where is element-wise
        # so this also handles the batched training dict.
        prompt = np.asarray(data.get("prompt", b""))
        data["prompt"] = np.where(prompt == b"", self.prompt, prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False
    # Optional per-dim bool mask. Where True, the dim is normalized as usual;
    # where False, the dim passes through unchanged. Same length convention as
    # `make_bool_mask` (length can be smaller than the actual dim count, in
    # which case missing tail dims default to True). Applied to every leaf
    # whose stats are present, so all leaves must share the same layout.
    mask: Sequence[bool] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _apply_mask(self, x, normalized):
        if self.mask is None:
            return normalized
        # Honor the documented "missing tail dims default to True" semantics:
        # pad the mask up to ``x.shape[-1]`` with True so np.where broadcasts
        # cleanly against padded action vectors (e.g. v4 emits a length-10
        # ``norm_mask`` but the model output is padded to action_dim=32).
        mask = np.asarray(self.mask)
        if mask.size < x.shape[-1]:
            mask = np.concatenate(
                [mask, np.ones(x.shape[-1] - mask.size, dtype=mask.dtype)]
            )
        else:
            mask = mask[: x.shape[-1]]
        return np.where(mask, normalized, x)

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return self._apply_mask(x, (x - mean) / (std + 1e-6))

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return self._apply_mask(x, (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # See `Normalize.mask`. Inverse semantics: where True, un-normalize; where
    # False, pass through unchanged.
    mask: Sequence[bool] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _apply_mask(self, x, unnormalized):
        if self.mask is None:
            return unnormalized
        # Honor the documented "missing tail dims default to True" semantics:
        # pad the mask up to ``x.shape[-1]`` with True so np.where broadcasts
        # cleanly against padded action vectors (e.g. v4 emits a length-10
        # ``norm_mask`` but the model output is padded to action_dim=32).
        mask = np.asarray(self.mask)
        if mask.size < x.shape[-1]:
            mask = np.concatenate(
                [mask, np.ones(x.shape[-1] - mask.size, dtype=mask.dtype)]
            )
        else:
            mask = mask[: x.shape[-1]]
        return np.where(mask, unnormalized, x)

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return self._apply_mask(x, x * (std + 1e-6) + mean)

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            unnormed = np.concatenate(
                [(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1
            )
        else:
            unnormed = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return self._apply_mask(x, unnormed)


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    debug_mode: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        # Note: keep `prompt` in the dict (don't pop) so downstream extras
        # passthrough (DataLoaderImpl._EXTRA_KEYS) can surface it for tagging
        # / analysis. Observation.from_dict ignores extra keys.
        prompt = data.get("prompt")
        if prompt is None:
            raise ValueError("Prompt is required")

        # Rank-polymorphic: scalar/0-d prompt -> per-sample; 1-d -> batched.
        is_batched = isinstance(prompt, np.ndarray) and prompt.ndim >= 1
        if is_batched:
            prompts = [_to_str(p) for p in prompt]
        else:
            prompts = [_to_str(prompt)]

        if self.discrete_state_input:
            state = data.get("state")
            if state is None:
                raise ValueError("State is required.")
            state_arr = np.asarray(state)
            states = state_arr if is_batched else state_arr[None]
        else:
            states = None

        tokens, token_masks = self.tokenizer.tokenize_batch(prompts, states, debug=self.debug_mode)
        if not is_batched:
            tokens = tokens[0]
            token_masks = token_masks[0]
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
