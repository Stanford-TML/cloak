from collections.abc import Iterable
import functools

import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
@at.typecheck
def compute_valid_mask(orig_h: int, orig_w: int, target_h: int, target_w: int) -> at.Bool[at.Array, "{target_h} {target_w}"]:
    """Returns a bool mask (target_h, target_w) that is True where real image pixels appear
    after resize_with_pad(orig_h x orig_w -> target_h x target_w).

    Mirrors the geometry of resize_with_pad in image_tools.py.
    """
    ratio = max(orig_w / target_w, orig_h / target_h)
    resized_h = int(orig_h / ratio)
    resized_w = int(orig_w / ratio)
    pad_h0 = (target_h - resized_h) // 2
    pad_w0 = (target_w - resized_w) // 2
    valid = jnp.zeros((target_h, target_w), dtype=jnp.bool_)
    valid = valid.at[pad_h0:pad_h0 + resized_h, pad_w0:pad_w0 + resized_w].set(True)
    return valid


@functools.partial(jax.jit, static_argnums=(1,))
@at.typecheck
def patch_mask_any(
    pixel_mask: at.Bool[at.Array, "h w"],
    patch_size: int = 14,
) -> at.Bool[at.Array, "ph pw"]:
    """(c) Patch is masked if any pixel in the patch is masked."""
    h, w = pixel_mask.shape
    ps = patch_size
    ph, pw = h // ps, w // ps
    return pixel_mask.reshape(ph, ps, pw, ps).any(axis=(1, 3))


@functools.partial(jax.jit, static_argnums=(2,))
@at.typecheck
def patch_mask_majority(
    pixel_mask: at.Bool[at.Array, "h w"],
    valid_mask: at.Bool[at.Array, "h w"] | None,
    patch_size: int = 14,
) -> at.Bool[at.Array, "ph pw"]:
    """(b) Patch is masked if the majority (>50%) of its pixels are masked.

    If valid_mask is given, only valid (non-padded) pixels count toward the
    denominator. Fully-padded patches are never masked.
    """
    h, w = pixel_mask.shape
    ps = patch_size
    ph, pw = h // ps, w // ps
    pm = pixel_mask.reshape(ph, ps, pw, ps)
    if valid_mask is None:
        return pm.mean(axis=(1, 3)) > 0.5
    vm = valid_mask.reshape(ph, ps, pw, ps)
    n_valid = vm.sum(axis=(1, 3))
    n_masked = (pm & vm).sum(axis=(1, 3))
    return (n_masked / jnp.maximum(n_valid, 1) > 0.5) & (n_valid > 0)


@functools.partial(jax.jit, static_argnums=(2,))
@at.typecheck
def patch_mask_quarter(
    pixel_mask: at.Bool[at.Array, "h w"],
    valid_mask: at.Bool[at.Array, "h w"] | None,
    patch_size: int = 14,
) -> at.Bool[at.Array, "ph pw"]:
    """(d) Patch is masked if at least 25% of its pixels are masked.

    Same logic as `patch_mask_majority` with a more lenient threshold —
    patches with even modest mask coverage get dropped. Useful when you
    want to be conservative about which patches reach the kept set.
    """
    h, w = pixel_mask.shape
    ps = patch_size
    ph, pw = h // ps, w // ps
    pm = pixel_mask.reshape(ph, ps, pw, ps)
    if valid_mask is None:
        return pm.mean(axis=(1, 3)) > 0.25
    vm = valid_mask.reshape(ph, ps, pw, ps)
    n_valid = vm.sum(axis=(1, 3))
    n_masked = (pm & vm).sum(axis=(1, 3))
    return (n_masked / jnp.maximum(n_valid, 1) > 0.25) & (n_valid > 0)


@functools.partial(jax.jit, static_argnums=(2,))
@at.typecheck
def patch_mask_centroid(
    pixel_mask: at.Bool[at.Array, "h w"],
    valid_mask: at.Bool[at.Array, "h w"] | None,
    patch_size: int = 14,
) -> at.Bool[at.Array, "ph pw"]:
    """(a) Patch is masked if the pixel at its centroid is masked.

    If valid_mask is given, the centroid is placed at the mean position of the
    valid (non-padded) pixels in the patch rather than the geometric centre.
    Fully-padded patches are never masked.
    """
    h, w = pixel_mask.shape
    ps = patch_size
    ph, pw = h // ps, w // ps
    half = ps // 2
    if valid_mask is None:
        return pixel_mask[half::ps, half::ps]
    vm = valid_mask.reshape(ph, ps, pw, ps)  # (ph, ps, pw, ps)
    row_idx = jnp.arange(ps).reshape(1, ps, 1, 1)
    col_idx = jnp.arange(ps).reshape(1, 1, 1, ps)
    n_valid = vm.sum(axis=(1, 3))  # (ph, pw)
    centroid_r = jnp.round((vm * row_idx).sum(axis=(1, 3)) / jnp.maximum(n_valid, 1)).astype(jnp.int32)
    centroid_c = jnp.round((vm * col_idx).sum(axis=(1, 3)) / jnp.maximum(n_valid, 1)).astype(jnp.int32)
    global_r = jnp.arange(ph)[:, None] * ps + centroid_r  # (ph, pw)
    global_c = jnp.arange(pw)[None, :] * ps + centroid_c  # (ph, pw)
    return pixel_mask[global_r, global_c] & (n_valid > 0)


_STRATEGY_FNS = {
    "centroid": patch_mask_centroid,
    "majority": patch_mask_majority,
    "quarter": patch_mask_quarter,
    "any": patch_mask_any,
}

VALID_STRATEGIES = tuple(_STRATEGY_FNS.keys())


def get_patch_mask_fn(strategy: str):
    """Return the patch masking function for the given strategy name."""
    if strategy not in _STRATEGY_FNS:
        raise ValueError(f"Unknown patch masking strategy: {strategy!r}. Must be one of {VALID_STRATEGIES}")
    return _STRATEGY_FNS[strategy]


def compute_patch_masks(
    pixel_masks: jnp.ndarray,
    valid_mask: jnp.ndarray | None,
    strategy: str,
    patch_size: int = 14,
) -> jnp.ndarray:
    """Compute patch masks for a batch of pixel masks.

    Args:
        pixel_masks: [b, h, w] bool array. True = pixel is masked.
        valid_mask: [h, w] bool array from compute_valid_mask, or None.
        strategy: one of "centroid", "majority", "any".
        patch_size: ViT patch size.

    Returns:
        [b, ph*pw] bool array. True = patch is masked (should be dropped).
    """
    fn = get_patch_mask_fn(strategy)

    def _single(pm):
        if strategy == "any":
            return fn(pm, patch_size).reshape(-1)
        return fn(pm, valid_mask, patch_size).reshape(-1)

    return jax.vmap(_single)(pixel_masks)


def compute_observation_patch_masks(
    pixel_masks: dict[str, jnp.ndarray] | None,
    original_image_dims: dict[str, tuple[int, int]],
    image_names: Iterable[str],
    strategy: str | None,
    image_resolution: tuple[int, int],
    patch_size: int = 14,
) -> dict[str, jnp.ndarray]:
    """Single source of truth for the per-image patch-mask computation used by both
    `Pi0.embed_prefix` and the sanity-check overlays.

    Gating matches `embed_prefix`: a patch mask is produced for an image only if
    `strategy` is set, `pixel_masks` is provided, the image name contains "wrist",
    and the same name appears in `pixel_masks`. Other images are absent from the
    returned dict.

    Args:
        pixel_masks: {image_key: [b, h, w]} bool arrays, True = pixel masked. None
            means patch masking is disabled.
        original_image_dims: {image_key: (orig_h, orig_w)} pre-resize dims, used to
            compute the valid (non-padding) mask.
        image_names: iterable of image names to consider (typically `obs.images.keys()`).
        strategy: one of "centroid", "majority", "any", or None to disable.
        image_resolution: (target_h, target_w) after resize_with_pad.
        patch_size: ViT patch size in pixels.

    Returns:
        {image_key: [b, num_patches]} bool arrays, True = patch is masked.
    """
    if strategy is None or pixel_masks is None:
        return {}
    target_h, target_w = image_resolution
    result = {}
    for name in image_names:
        if "wrist" not in name or name not in pixel_masks:
            continue
        pixel_mask = pixel_masks[name]
        orig_h, orig_w = original_image_dims.get(name, (target_h, target_w))
        valid_mask = compute_valid_mask(orig_h, orig_w, target_h, target_w)
        result[name] = compute_patch_masks(pixel_mask, valid_mask, strategy, patch_size)
    return result
