"""DataTransformFn that generates gripper masks on-the-fly during inference.

Injected before DroidInputs in the transform pipeline so that the mask
appears in the data dict under the key DroidInputs already knows how to consume.
"""

from collections.abc import Callable
import logging
from typing import Literal

import cv2
import jax
import numpy as np

from openpi.constants import YAM_CAM_TO_GRIPPER, YAM_WRIST_INTRINSIC_FY
from openpi.policies.gripper_mask_renderer import get_renderer
from openpi.policies.gripper_mask_renderer import get_sharpa_renderer
from openpi.policies.gripper_mask_renderer import get_umi_renderer
from openpi.policies.gripper_mask_renderer import get_yam_renderer
from openpi.shared.image_tools import resize_with_pad

logger = logging.getLogger("openpi")

_MASK_KEY = "observation/wrist_image_left_gripper_mask"
_MASK_KEY_UMI = "observation/wrist_image_left_gripper_mask_umi"
_MASK_KEY_SHARPA = "observation/wrist_image_left_gripper_mask_sharpa"
_WRIST_IMAGE_KEY = "observation/wrist_image"
_EXTRINSIC_KEY = "observation/camera_extrinsic"
_INTRINSIC_KEY = "observation/camera_intrinsic"
_HAND_JOINT_KEY = "observation/hand_joint_position"
# Optional [x_frac, y_frac, w_frac, h_frac] crop the client applied to the
# wrist image before sending. When present, the rendered mask is cropped with
# the same window so it stays aligned with the (cropped) image.
_CROP_KEY = "observation/wrist_crop_xywh_frac"
# When present, value is treated as cam-to-gripper and the server FK-composes
# T_cam_to_base = T_ee_attachment_site @ T_cam_to_gripper before passing it to
# the renderer. Same composition the training-time mask generator uses
# (examples/droid/preprocess_data.py:extract_extrinsics). FrankaSharpaEnv
# clients should send this — their cartesian_position is unreliable so the
# legacy _EXTRINSIC_KEY they send is actually cam-to-gripper, not cam-to-base.
_GRIPPER_EXT_KEY = "observation/wrist_extrinsic_cam_to_gripper"


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a bool (H, W) mask using the same resize_with_pad as RGB images."""
    mask3 = np.stack([mask.astype(np.uint8)] * 3, axis=-1)
    resized = np.asarray(
        resize_with_pad(mask3, height, width, method=jax.image.ResizeMethod.NEAREST)
    )
    return resized[..., 0] > 0


def _crop_with_pad(image: np.ndarray, crop_xywh_frac: tuple) -> np.ndarray:
    """Crop image to the given fractional window (top-left origin, normalized
    by source W/H). Pixels outside the source are padded with zeros.

    Mirrors the client-side wrist-image crop convention.
    Works on 2D (H, W) arrays (e.g. boolean masks) and 3D (H, W, C) arrays alike.
    """
    H, W = image.shape[:2]
    fx, fy, fw, fh = crop_xywh_frac
    x0 = int(round(fx * W))
    y0 = int(round(fy * H))
    cw = int(round(fw * W))
    ch = int(round(fh * H))
    out_shape = (ch, cw) if image.ndim == 2 else (ch, cw, image.shape[2])
    out = np.zeros(out_shape, dtype=image.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + cw), min(H, y0 + ch)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return out


class InjectGripperMask:
    """Generate and inject a gripper pixel mask from joint state + camera params.

    Requires ``observation/camera_extrinsic`` or ``observation/wrist_extrinsic_cam_to_gripper``
    in the data dict — raises otherwise.
    """

    def __init__(
        self,
        *,
        patch_masking_strategy: str | None = None,
        renderer_factory: Callable[[], "object"] | None = None,
        gripper_dilation_kernel_size: int = 16,
        default_cam_to_gripper: np.ndarray | None = None,
        default_intrinsic_fy: float | None = None,
    ) -> None:
        self._patch_masking_strategy = patch_masking_strategy
        self._logged_first_call = False
        # If None, falls back to module-level get_renderer() (Robotiq scene).
        self._renderer_factory = renderer_factory
        self._gripper_dilation_kernel_size = gripper_dilation_kernel_size
        # Server-side fallbacks for clients that send no camera params (YAM).
        # ``default_cam_to_gripper`` is a 6-vec [xyz, euler-xyz] composed the same
        # way as a client-sent one; ``default_intrinsic_fy`` is the wrist fy (px).
        # None for client-sourced robots (Robotiq/UMI/Sharpa).
        self._default_cam_to_gripper = default_cam_to_gripper
        self._default_intrinsic_fy = default_intrinsic_fy

    @staticmethod
    def combine_masks(gripper_mask: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Training-equivalent combine: mirrors `preprocess_data.compute_mask`
        (dilate the base mount with kernel 16, OR with the gripper). Subclasses
        override for robots whose training combine is different (e.g. Sharpa:
        plain OR — base is a subset of the hand). Static so calibration tuners
        can call it per robot (``InjectSharpaMask.combine_masks(...)``) without
        an instance.

        Not used by ``__call__`` — inference uses ``_build_inference_mask``,
        which applies a kernel-size-tunable dilation to ``OR(gripper, base)``
        directly (avoids dilating an already-dilated base).
        """
        kernel = np.ones((16, 16), dtype=np.uint8)
        dilated_base = cv2.dilate(base_mask.astype(np.uint8), kernel).astype(bool)
        return np.logical_or(gripper_mask, dilated_base)

    def _build_inference_mask(
        self, gripper_mask: np.ndarray, base_mask: np.ndarray
    ) -> np.ndarray:
        """Inference combine: ``dilate(OR(gripper, base), k)`` where ``k`` is
        ``self._gripper_dilation_kernel_size``. One dilation over the union,
        not a dilation of an already-dilated base — equivalent to dilating
        gripper and base separately by ``k`` and OR'ing (dilation distributes
        over union). At ``k <= 0``, reduces to a plain OR.
        """
        union = np.logical_or(gripper_mask, base_mask)
        k = self._gripper_dilation_kernel_size
        if k <= 0:
            return union
        kernel = np.ones((k, k), dtype=np.uint8)
        return cv2.dilate(union.astype(np.uint8), kernel).astype(bool)

    def __call__(self, data: dict) -> dict:
        assert _MASK_KEY not in data, (
            f"{_MASK_KEY} already present in data; InjectGripperMask is an inference "
            "transform and assumes the mask is not yet in the data."
        )
        has_client_ext = _EXTRINSIC_KEY in data or _GRIPPER_EXT_KEY in data
        if not has_client_ext and self._default_cam_to_gripper is None:
            raise ValueError(
                f"Masking is enabled but the client did not send '{_EXTRINSIC_KEY}' or "
                f"'{_GRIPPER_EXT_KEY}', and no server-side default_cam_to_gripper is set. "
                "Ensure the client has camera calibration set up and is sending extrinsics."
            )

        renderer = (self._renderer_factory or get_renderer)()

        if _INTRINSIC_KEY in data:
            intrinsic = np.asarray(data[_INTRINSIC_KEY])
            renderer.set_intrinsics(fy=float(intrinsic[1]))
        elif self._default_intrinsic_fy is not None:
            renderer.set_intrinsics(fy=float(self._default_intrinsic_fy))

        joint_position = data["observation/joint_position"]
        gripper_pos = data["observation/gripper_position"]
        assert gripper_pos.shape == (1,), (
            f"expected observation/gripper_position shape (1,), got {gripper_pos.shape}"
        )
        gripper_position = gripper_pos.item()

        # Settle once, then derive cam-to-base from whichever extrinsic the client
        # sent. Three protocols:
        #  - New: client sends cam-to-gripper -> FK-compose with the settled EE
        #    attachment site to get cam-to-base.
        #  - Legacy: client sends cam-to-base directly.
        #  - Server default (YAM): client sends nothing -> use
        #    ``default_cam_to_gripper`` as cam-to-gripper and FK-compose.
        settled_qpos = renderer.forward_qpos(joint_position, gripper_position)
        if _GRIPPER_EXT_KEY in data:
            camera_extrinsic = renderer.compose_cam_to_base(np.asarray(data[_GRIPPER_EXT_KEY]))
            used_gripper_ext = True
        elif _EXTRINSIC_KEY in data:
            camera_extrinsic = np.asarray(data[_EXTRINSIC_KEY])
            used_gripper_ext = False
        else:
            camera_extrinsic = renderer.compose_cam_to_base(np.asarray(self._default_cam_to_gripper))
            used_gripper_ext = True
        gripper_mask, base_mask = renderer.render_mask_at(settled_qpos, camera_extrinsic)
        combined = self._build_inference_mask(gripper_mask, base_mask)

        # Single source of truth for image+mask resize is `model.preprocess_observation`.
        # Leave the mask at the renderer's source resolution; the model resizes it
        # to image_resolution with the same aspect-ratio padding as the RGB image.

        if not self._logged_first_call:
            self._logged_first_call = True
            has_intrinsic = _INTRINSIC_KEY in data
            has_wrist_image = _WRIST_IMAGE_KEY in data
            mask_pixels = int(combined.sum())
            total_pixels = int(combined.size)
            logger.info(
                "Gripper masking active — extrinsic: %s, intrinsic: %s, wrist_image: %s, "
                "mask coverage: %d/%d pixels (%.1f%%)",
                f"{_GRIPPER_EXT_KEY} (FK-composed cam-to-base)" if used_gripper_ext
                else f"{_EXTRINSIC_KEY} (used as cam-to-base)",
                "yes" if has_intrinsic else "no (using default)",
                "yes" if has_wrist_image else "no",
                mask_pixels,
                total_pixels,
                100.0 * mask_pixels / max(total_pixels, 1),
            )

        return {**data, _MASK_KEY: combined[..., None]}


# ---------------------------------------------------------------------------
# Module-level singleton so serve_policy.py can access the transform instance.
# ---------------------------------------------------------------------------

_mask_transform_instance: InjectGripperMask | None = None


def get_mask_transform(
    *,
    patch_masking_strategy: str | None = None,
    gripper_dilation_kernel_size: int = 16,
) -> InjectGripperMask:
    """Return a shared InjectGripperMask instance (created on first call)."""
    global _mask_transform_instance
    if _mask_transform_instance is None:
        _mask_transform_instance = InjectGripperMask(
            patch_masking_strategy=patch_masking_strategy,
            gripper_dilation_kernel_size=gripper_dilation_kernel_size,
        )
    return _mask_transform_instance


class InjectSharpaMask(InjectGripperMask):
    """Generate and inject a Sharpa hand pixel mask from joint state + camera params.

    Subclasses InjectGripperMask — only overrides ``__call__`` to read
    ``hand_joint_position`` (22 DOF) instead of ``gripper_position`` (1 scalar).
    """

    @staticmethod
    def combine_masks(gripper_mask: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Sharpa training-equivalent combine: plain OR. The hand geoms already
        form a complete silhouette and the base (mount + palm) is a subset of
        the hand geoms, so a plain OR is the full mask. Matches Sharpa
        preprocess where ``base_masks`` is empty.

        Not used by ``__call__`` — inference uses the inherited
        ``_build_inference_mask`` from ``InjectGripperMask``.
        """
        return np.logical_or(gripper_mask, base_mask)

    def __call__(self, data: dict) -> dict:
        if _MASK_KEY in data:
            if not self._logged_first_call:
                self._logged_first_call = True
                logger.info("Hand mask already present in data, skipping mask generation.")
            return data
        if _EXTRINSIC_KEY not in data and _GRIPPER_EXT_KEY not in data:
            raise ValueError(
                f"Masking is enabled but the client did not send '{_EXTRINSIC_KEY}' or "
                f"'{_GRIPPER_EXT_KEY}'. Ensure the client has camera calibration set up "
                "and is sending extrinsics."
            )

        renderer = get_sharpa_renderer()

        if _INTRINSIC_KEY in data:
            intrinsic = np.asarray(data[_INTRINSIC_KEY])
            renderer.set_intrinsics(fy=float(intrinsic[1]))

        joint_position = np.asarray(data["observation/joint_position"])
        hand_joint_position = np.asarray(data[_HAND_JOINT_KEY])

        # Settle once, derive cam-to-base from whichever extrinsic the client
        # sent, then render from the settled state. Two extrinsic protocols:
        #  - New: client sends cam-to-gripper -> FK-compose with the settled EE
        #    attachment site to get cam-to-base.
        #  - Legacy: client sends cam-to-base directly.
        settled_qpos = renderer.forward_qpos(joint_position, hand_joint_position)
        used_gripper_ext = _GRIPPER_EXT_KEY in data
        if used_gripper_ext:
            cam_to_gripper = np.asarray(data[_GRIPPER_EXT_KEY])
            camera_extrinsic = renderer.compose_cam_to_base(cam_to_gripper)
        else:
            camera_extrinsic = np.asarray(data[_EXTRINSIC_KEY])
        hand_mask, base_mask = renderer.render_mask_at(settled_qpos, camera_extrinsic)

        combined = self._build_inference_mask(hand_mask, base_mask)
        # Single source of truth for image+mask resize is `model.preprocess_observation`.
        # Leave the mask at source resolution; the model resizes it to
        # image_resolution with the same aspect-ratio padding as the RGB image.

        if not self._logged_first_call:
            self._logged_first_call = True
            has_intrinsic = _INTRINSIC_KEY in data
            has_wrist_image = _WRIST_IMAGE_KEY in data
            mask_pixels = int(combined.sum())
            total_pixels = int(combined.size)
            logger.info(
                "Sharpa hand masking active — extrinsic: %s, intrinsic: %s, wrist_image: %s, "
                "crop: %s, mask coverage: %d/%d pixels (%.1f%%)",
                f"{_GRIPPER_EXT_KEY} (FK-composed cam-to-base)" if used_gripper_ext
                else f"{_EXTRINSIC_KEY} (used as cam-to-base)",
                "yes" if has_intrinsic else "no (using default)",
                "yes" if has_wrist_image else "no",
                "yes" if _CROP_KEY in data else "no",
                mask_pixels,
                total_pixels,
                100.0 * mask_pixels / max(total_pixels, 1),
            )

        return {**data, _MASK_KEY: combined[..., None]}


_sharpa_mask_transform_instance: InjectSharpaMask | None = None


def get_sharpa_mask_transform(
    *,
    patch_masking_strategy: str | None = None,
    gripper_dilation_kernel_size: int = 16,
) -> InjectSharpaMask:
    """Return a shared InjectSharpaMask instance (created on first call)."""
    global _sharpa_mask_transform_instance
    if _sharpa_mask_transform_instance is None:
        _sharpa_mask_transform_instance = InjectSharpaMask(
            patch_masking_strategy=patch_masking_strategy,
            gripper_dilation_kernel_size=gripper_dilation_kernel_size,
        )
    return _sharpa_mask_transform_instance


# ---------------------------------------------------------------------------
# UMI / YAM mask transforms. Both are parallel jaws that send a scalar
# ``gripper_position`` (like Robotiq), so the base ``InjectGripperMask.__call__``
# handles them directly — only the renderer differs. No subclass needed (unlike
# Sharpa, which reads the 22-DOF hand vector).
# ---------------------------------------------------------------------------

_umi_mask_transform_instance: InjectGripperMask | None = None


def get_umi_mask_transform(
    *,
    patch_masking_strategy: str | None = None,
    gripper_dilation_kernel_size: int = 16,
) -> InjectGripperMask:
    """Return a shared UMI mask-inject transform (InjectGripperMask + UMI renderer)."""
    global _umi_mask_transform_instance
    if _umi_mask_transform_instance is None:
        _umi_mask_transform_instance = InjectGripperMask(
            patch_masking_strategy=patch_masking_strategy,
            renderer_factory=get_umi_renderer,
            gripper_dilation_kernel_size=gripper_dilation_kernel_size,
        )
    return _umi_mask_transform_instance


_yam_mask_transform_instance: InjectGripperMask | None = None


def get_yam_mask_transform(
    *,
    patch_masking_strategy: str | None = None,
    gripper_dilation_kernel_size: int = 16,
) -> InjectGripperMask:
    """Return a shared YAM mask-inject transform (InjectGripperMask + YAM renderer).

    The YAM client sends no camera params, so the cam-to-gripper extrinsic and
    wrist intrinsic come from server-side defaults (``YAM_CAM_TO_GRIPPER`` /
    ``YAM_WRIST_INTRINSIC_FY``).
    """
    global _yam_mask_transform_instance
    if _yam_mask_transform_instance is None:
        _yam_mask_transform_instance = InjectGripperMask(
            patch_masking_strategy=patch_masking_strategy,
            renderer_factory=get_yam_renderer,
            gripper_dilation_kernel_size=gripper_dilation_kernel_size,
            default_cam_to_gripper=YAM_CAM_TO_GRIPPER,
            default_intrinsic_fy=YAM_WRIST_INTRINSIC_FY,
        )
    return _yam_mask_transform_instance


class MaskAugmentation:
    """Train-only wrist-mask augmentation.

    Per sample, in order: `roll_image` (roll the wrist image under the
    original gripper mask so no unmasked pixel can leak gripper content),
    then the mask-building step, then `remove_circles`. The mask-building
    step depends on `augmentation_type`: "blob" runs `add_capsules`,
    "embodiment" runs `add_embodiment_mask`. See per-method docstrings for
    sampling details.
    """

    _MASK_KEY = _MASK_KEY
    _MASK_KEY_UMI = _MASK_KEY_UMI
    _MASK_KEY_SHARPA = _MASK_KEY_SHARPA

    def __init__(
        self,
        *,
        augmentation_type: Literal["blob", "embodiment"] = "blob",
        seed: int | None = None,
    ) -> None:
        self._augmentation_type = augmentation_type
        self._rng = np.random.default_rng(seed)

    def _boundary_pixels(self, mask_bin: np.ndarray) -> np.ndarray:
        """(N, 2) array of (y, x) pixels on the inside boundary of `mask_bin`."""
        eroded = cv2.erode(mask_bin, np.ones((3, 3), np.uint8))
        return np.argwhere((mask_bin & (1 - eroded)) > 0)

    def _sample_capsule(
        self,
        anchor_pts: np.ndarray,
        *,
        r_min: int,
        r_max: int,
        len_min: int,
        len_max: int,
        angle_max_deg: float = 30.0,
    ) -> tuple[tuple[int, int], tuple[int, int], int] | None:
        """One tall-biased capsule anchored at a uniformly-sampled pixel from
        `anchor_pts`. Returns `((cx0, cy0), (cx1, cy1), r)`; axis tilts by
        `±angle_max_deg` from vertical with random up/down sign."""
        if len(anchor_pts) == 0:
            return None
        cy0, cx0 = anchor_pts[self._rng.integers(0, len(anchor_pts))]
        r = int(self._rng.integers(r_min, r_max + 1))
        L = int(self._rng.integers(len_min, len_max + 1))
        angle_rad = float(np.deg2rad(self._rng.uniform(-angle_max_deg, angle_max_deg)))
        sign = int(self._rng.choice([-1, 1]))
        dy = sign * L * np.cos(angle_rad)
        dx = L * np.sin(angle_rad)
        return (int(cx0), int(cy0)), (int(cx0 + dx), int(cy0 + dy)), r

    def add_capsules(self, mask: np.ndarray) -> np.ndarray:
        """K ~ U{1, 2} tall-biased capsules OR-ed onto `mask`, anchored at the boundary."""
        K = int(self._rng.integers(1, 3))
        r_min, r_max = 12, 16
        len_min, len_max = 32, 64
        angle_max_deg = 30.0

        m = (np.asarray(mask) > 0).astype(np.uint8)
        if not m.any():
            return m

        boundary_pts = self._boundary_pixels(m)
        for _ in range(K):
            cap = self._sample_capsule(
                boundary_pts,
                r_min=r_min, r_max=r_max,
                len_min=len_min, len_max=len_max,
                angle_max_deg=angle_max_deg,
            )
            if cap is None:
                continue
            pt1, pt2, r = cap
            cv2.line(m, pt1, pt2, color=1, thickness=2 * r)
        return m

    def add_embodiment_mask(
        self, gripper_mask: np.ndarray, umi_mask: np.ndarray, sharpa_mask: np.ndarray,
    ) -> np.ndarray:
        """Per sample, union the gripper mask with one randomly chosen
        embodiment mask, or none. With equal probability the result is
        `gripper`, `gripper ∪ umi`, or `gripper ∪ sharpa`. All inputs are
        (H, W); returns (H, W) uint8. Kept per-sample (not batched): the
        ~1/3 `none` early-return and small-slice ops touch far less memory
        than the full-batch passes a vectorized version would do."""
        m = (np.asarray(gripper_mask) > 0).astype(np.uint8)
        choice = int(self._rng.integers(0, 3))     # 0=none, 1=umi, 2=sharpa
        if choice == 1:
            extra = umi_mask
        elif choice == 2:
            extra = sharpa_mask
        else:
            return m
        m |= (np.asarray(extra) > 0).astype(np.uint8)
        return m

    def roll_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Roll each wrist image by a random fraction of its size and paste
        the rolled pixels back wherever `mask` is on, so no unmasked pixel
        can leak gripper content. Batched: `image` is (B, H, W, C), `mask`
        the (B, H, W) original gripper mask. Samples whose mask is empty are
        returned unchanged (the `np.where` keeps the original pixels)."""
        image = np.asarray(image)
        B, H, W = image.shape[:3]
        assert mask.shape == (B, H, W), f"mask {mask.shape} must match image {(B, H, W)}"
        shift_h = self._rng.integers(H // 3, 2 * H // 3 + 1, size=B)
        shift_w = self._rng.integers(W // 3, 2 * W // 3 + 1, size=B)
        # np.roll is two contiguous slice copies per sample — much faster than
        # a fancy-index gather (scattered reads + large index arrays), so the
        # per-sample loop stays.
        rolled = np.empty_like(image)
        for b in range(B):
            rolled[b] = np.roll(image[b], shift=(int(shift_h[b]), int(shift_w[b])), axis=(0, 1))
        inside4 = (mask > 0)[..., None]                            # (B, H, W, 1) bool
        return np.where(inside4, rolled, image)

    def remove_circles(self, mask: np.ndarray) -> np.ndarray:
        """Subtract K ~ U{1, 2} small-medium circles from `mask`. Centers
        sampled uniformly from pixels inside the *current* mask (post-add),
        so a circle may land on top of and remove from a newly-added
        capsule."""
        K = int(self._rng.integers(1, 3))
        r_min, r_max = 8, 16

        m = (np.asarray(mask) > 0).astype(np.uint8)
        inside_pts = np.argwhere(m)
        if len(inside_pts) == 0:
            return m
        for _ in range(K):
            cy, cx = inside_pts[self._rng.integers(0, len(inside_pts))]
            r = int(self._rng.integers(r_min, r_max + 1))
            cv2.circle(m, (int(cx), int(cy)), r, color=0, thickness=-1)
        return m

    def __call__(self, data: dict) -> dict:
        raw = np.asarray(data[self._MASK_KEY])
        image = np.asarray(data[_WRIST_IMAGE_KEY])
        assert raw.ndim == 4 and raw.shape[-1] == 1 and raw.dtype == np.uint8, (
            f"MaskAugmentation expects mask shape (B, H, W, 1) uint8; "
            f"got shape={raw.shape}, dtype={raw.dtype}"
        )
        assert image.ndim == 4 and image.shape[:3] == raw.shape[:3], (
            f"MaskAugmentation expects image shape (B, H, W, C) with matching (B, H, W); "
            f"got image shape={image.shape}, mask shape={raw.shape}"
        )
        B = raw.shape[0]
        masks = (raw[..., 0] > 0).astype(np.uint8)                  # (B, H, W)

        umi = sharpa = None
        if self._augmentation_type == "embodiment":
            umi_raw = np.asarray(data[self._MASK_KEY_UMI])
            sharpa_raw = np.asarray(data[self._MASK_KEY_SHARPA])
            assert umi_raw.shape == raw.shape and sharpa_raw.shape == raw.shape, (
                f"embodiment masks must match gripper mask shape {raw.shape}; "
                f"got umi={umi_raw.shape}, sharpa={sharpa_raw.shape}"
            )
            umi = (umi_raw[..., 0] > 0).astype(np.uint8)               # (B, H, W)
            sharpa = (sharpa_raw[..., 0] > 0).astype(np.uint8)         # (B, H, W)

        images_out = self.roll_image(image, masks)

        # Per-sample: blob adds cv2 capsules; embodiment unions one randomly
        # chosen embodiment mask. Both are cheaper per-sample than batched
        # (memory-bound: small slices + early-outs beat full-batch passes).
        masks_out = np.empty_like(masks)
        for b in range(B):
            if self._augmentation_type == "blob":
                masks[b] = self.add_capsules(masks[b])
            else:
                masks[b] = self.add_embodiment_mask(masks[b], umi[b], sharpa[b])

            masks_out[b] = self.remove_circles(masks[b])

        # The umi/sharpa masks are consumed here (merged into the gripper
        # mask); nothing downstream reads them, so drop them so they aren't
        # carried through the rest of the transform chain.
        out = {**data, self._MASK_KEY: masks_out[..., None], _WRIST_IMAGE_KEY: images_out}
        out.pop(self._MASK_KEY_UMI, None)
        out.pop(self._MASK_KEY_SHARPA, None)
        return out
