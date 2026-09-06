import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger("openpi")


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # 3D + leading dim 3 = CHW single image. (Don't trigger on batched 4D inputs where shape[0] happens to be 3.)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # If true, qpos (joint_position + gripper_position) is included in state/action.
    use_qpos: bool = True

    def __call__(self, data: dict) -> dict:
        if not self.use_qpos:
            raise ValueError("DroidInputs has no state components: enable use_qpos.")

        # Asserts: every input must be batched. Rest of the function assumes this.
        gp = data["observation/gripper_position"]
        jp = data["observation/joint_position"]
        base = data["observation/exterior_image_1_left"]
        wrist = data["observation/wrist_image"]
        assert gp.ndim == 2 and gp.shape[-1] == 1, f"expected (B, 1) gripper_position, got {gp.shape}"
        assert jp.ndim == 2 and jp.shape[-1] == 7, f"expected (B, 7) joint_position, got {jp.shape}"
        assert base.ndim == 4 and base.shape[-1] == 3, f"expected (B, H, W, 3) exterior image, got {base.shape}"
        assert wrist.ndim == 4 and wrist.shape[-1] == 3, f"expected (B, H, W, 3) wrist image, got {wrist.shape}"

        if "observation/wrist_image_left_gripper_mask" in data:
            m = data["observation/wrist_image_left_gripper_mask"]
            assert m.ndim == 4 and m.shape[-1] == 1, f"expected (B, H, W, 1) gripper mask, got {m.shape}"
            assert m.shape[1:3] == wrist.shape[1:3], (
                f"gripper mask shape {m.shape[1:3]} != wrist image shape {wrist.shape[1:3]}; "
                "they must share the same source resolution."
            )

        gripper_pos = np.asarray(gp)
        joint_pos = np.asarray(jp)
        batch_size = joint_pos.shape[0]

        state = np.concatenate([joint_pos, gripper_pos], axis=-1)

        base_image = _parse_image(base)
        wrist_image = _parse_image(wrist)

        ones = np.ones(batch_size, dtype=np.bool_)
        zeros = np.zeros(batch_size, dtype=np.bool_)

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (ones, ones, zeros)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Pass through pixel-level gripper masks for wrist cameras if available.
        # The model handles resizing and original_image_dims recording in preprocess_observation.
        gripper_mask_key = "observation/wrist_image_left_gripper_mask"
        if gripper_mask_key in data:
            pixel_mask = np.asarray(data[gripper_mask_key])[..., 0].astype(bool)
            inputs["pixel_mask"] = {"left_wrist_0_rgb": pixel_mask}

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Pass through per-example identifier for downstream eval tagging.
        if "step_id" in data:
            inputs["step_id"] = data["step_id"]

        # Pass through episode identifier for unique-episode tracking in the train loop.
        if "file_path" in data:
            inputs["file_path"] = data["file_path"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    # If true, qpos is included in the action target.
    use_qpos: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if self.use_qpos:
            return {"actions": actions[..., :8]}
        raise NotImplementedError(f"DroidOutputs(use_qpos={self.use_qpos}) is not yet wired up.")
