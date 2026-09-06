"""Sanity-check visualizations for the first training batch.

build_sanity_check_images constructs a {wandb_key: [wandb.Image, ...]} dict; the
caller is responsible for logging it (so this module stays side-effect-free
apart from writing files to disk).
"""

from pathlib import Path

import jax
import numpy as np
import wandb

import openpi.models.model as _model
import openpi.shared.masking as _masking
import openpi.shared.visualization as _visualization

WRIST_KEY = "left_wrist_0_rgb"


def build_sanity_check_images(
    observation: _model.Observation,
    output_dir: Path,
    *,
    n_samples: int = 10,
    patch_masking_strategy: str | None = None,
    patch_size: int = 14,
) -> dict[str, list[wandb.Image]]:
    """Save camera views, pixel masks, and patch overlays; return wandb payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_data: dict[str, list[wandb.Image]] = {}

    image_resolution = _model.IMAGE_RESOLUTION
    patches_per_side = image_resolution[0] // patch_size

    batch_size = len(next(iter(observation.images.values())))
    n_samples = min(n_samples, batch_size)

    _save_camera_views(observation, output_dir, n_samples, wandb_data)

    # preprocess_observation resizes masks and records original_image_dims.
    if patch_masking_strategy is None:
        return wandb_data

    processed_obs = _model.preprocess_observation(None, observation, train=False)
    _save_patch_overlays(
        processed_obs,
        output_dir,
        n_samples,
        patch_masking_strategy,
        image_resolution,
        patch_size,
        patches_per_side,
        wandb_data,
    )
    return wandb_data


def _save_camera_views(
    observation: _model.Observation,
    output_dir: Path,
    n_samples: int,
    wandb_data: dict[str, list[wandb.Image]],
) -> None:
    cpu_images = jax.device_get(observation.images)

    _visualization.save_camera_views(cpu_images, output_dir, n_samples)
    images_to_log = [np.concatenate([img[i] for img in cpu_images.values()], axis=1) for i in range(n_samples)]
    wandb_data["camera_views"] = [wandb.Image(img) for img in images_to_log]


def _save_patch_overlays(
    processed_obs: _model.Observation,
    output_dir: Path,
    n_samples: int,
    patch_masking_strategy: str,
    image_resolution: tuple[int, int],
    patch_size: int,
    patches_per_side: int,
    wandb_data: dict[str, list[wandb.Image]],
) -> None:
    patch_masks = _masking.compute_observation_patch_masks(
        processed_obs.pixel_masks,
        processed_obs.original_image_dims,
        processed_obs.images.keys(),
        patch_masking_strategy,
        image_resolution,
        patch_size,
    )
    pmask_np = np.asarray(patch_masks[WRIST_KEY])  # [b, num_patches], True=masked
    keep_mask = ~pmask_np  # True=kept
    wrist_images = jax.device_get(processed_obs.images[WRIST_KEY])
    wrist_pixel_masks = jax.device_get(processed_obs.pixel_masks[WRIST_KEY])
    overlay_images = []
    for i in range(n_samples):
        overlay_path = output_dir / f"patch_overlay_{i}.png"
        _visualization.save_patch_overlay(
            image=np.asarray(wrist_images[i]),
            pixel_mask=np.asarray(wrist_pixel_masks[i]),
            patch_mask_1d=keep_mask[i],
            path=overlay_path,
            patch_size=patch_size,
            patches_per_side=patches_per_side,
        )
        overlay_images.append(wandb.Image(str(overlay_path)))

    wandb_data["patch_overlays"] = overlay_images
