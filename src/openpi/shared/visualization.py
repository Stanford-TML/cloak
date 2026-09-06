"""Visualization utilities for training sanity checks."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_camera_views(
    images: dict[str, np.ndarray],
    output_dir: Path,
    n_samples: int,
) -> None:
    """Save concatenated camera views to disk.

    Args:
        images: {camera_key: [b, h, w, 3]} float32 arrays.
        output_dir: directory to write images into (must exist).
        n_samples: number of batch samples to save.
    """
    concat_images = [
        np.concatenate([img[i] for img in images.values()], axis=1)
        for i in range(n_samples)
    ]
    for i, img in enumerate(concat_images):
        img_ = (img - img.min()) / (img.max() - img.min())
        img_uint8 = (img_ * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(output_dir / f"sample_{i}.png")


def save_patch_overlay(
    image: np.ndarray,
    pixel_mask: np.ndarray,
    patch_mask_1d: np.ndarray,
    path: Path,
    patch_size: int,
    patches_per_side: int,
) -> None:
    """Save a concatenated image: [pixel mask overlay | patch grid overlay].

    Args:
        image: (H, W, 3) float32 in [-1, 1].
        pixel_mask: (H, W) bool, True = pixel is masked.
        patch_mask_1d: (num_patches,) bool, True = token is KEPT.
        path: output file path.
        patch_size: pixel size of each ViT patch.
        patches_per_side: number of patches along each spatial dimension.
    """
    img_uint8 = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]

    # Left panel: pixel mask overlaid on the image in red.
    mask_panel = img_uint8.copy()
    mask_panel[pixel_mask] = (
        mask_panel[pixel_mask] * 0.5 + np.array([160, 0, 0], dtype=np.uint8) * 0.5
    ).astype(np.uint8)

    # Right panel: patch overlay with grid.
    patch_grid = patch_mask_1d.reshape(patches_per_side, patches_per_side)

    img = Image.fromarray(img_uint8).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for i in range(patches_per_side):
        for j in range(patches_per_side):
            if not patch_grid[i, j]:
                draw_ov.rectangle(
                    [j * patch_size, i * patch_size, (j + 1) * patch_size - 1, (i + 1) * patch_size - 1],
                    fill=(200, 0, 0, 150),
                )

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    for y in range(0, h + 1, patch_size):
        draw.line([(0, y), (w, y)], fill=(180, 180, 180, 80))
    for x in range(0, w + 1, patch_size):
        draw.line([(x, 0), (x, h)], fill=(180, 180, 180, 80))

    patch_panel = np.array(img.convert("RGB"))

    # Concatenate side by side.
    combined = np.concatenate([mask_panel, patch_panel], axis=1)
    Image.fromarray(combined).save(path)


def save_affordance_overlay(
    image: np.ndarray,
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    left_patch_idx: int,
    right_patch_idx: int,
    path: Path,
    patch_size: int,
    patches_per_side: int,
) -> None:
    """Save an overlay showing left/right affordance patches + fingertip pixels.

    - Left patch tinted green, right patch tinted blue. Collision (same patch) => yellow.
    - Fingertip pixel coords drawn as lime (left) / cyan (right) dots.

    Args:
        image: (H, W, 3) float32 in [-1, 1].
        left_xy, right_xy: (2,) float pixel coords in (x, y) order within the image.
        left_patch_idx, right_patch_idx: flat patch indices (row * patches_per_side + col).
        path: output file path.
        patch_size: pixel size of each ViT patch.
        patches_per_side: number of patches along each spatial dimension.
    """
    img_uint8 = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]

    img = Image.fromarray(img_uint8).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    def _patch_box(idx: int) -> tuple[int, int, int, int]:
        row = idx // patches_per_side
        col = idx % patches_per_side
        return (col * patch_size, row * patch_size, (col + 1) * patch_size - 1, (row + 1) * patch_size - 1)

    if left_patch_idx == right_patch_idx:
        draw_ov.rectangle(_patch_box(left_patch_idx), fill=(220, 200, 0, 170))  # yellow: collision
    else:
        draw_ov.rectangle(_patch_box(left_patch_idx), fill=(0, 200, 0, 150))    # green: left
        draw_ov.rectangle(_patch_box(right_patch_idx), fill=(0, 120, 220, 150))  # blue: right

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    for y in range(0, h + 1, patch_size):
        draw.line([(0, y), (w, y)], fill=(180, 180, 180, 80))
    for x in range(0, w + 1, patch_size):
        draw.line([(x, 0), (x, h)], fill=(180, 180, 180, 80))

    r = 3
    lx, ly = float(left_xy[0]), float(left_xy[1])
    rx, ry = float(right_xy[0]), float(right_xy[1])
    draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(50, 255, 50, 255), outline=(0, 0, 0, 255))
    draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=(50, 180, 255, 255), outline=(0, 0, 0, 255))

    Image.fromarray(np.array(img.convert("RGB"))).save(path)
