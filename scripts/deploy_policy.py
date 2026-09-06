# ruff: noqa

import configparser
import contextlib
import dataclasses
import datetime
import faulthandler
import itertools
import re
import select
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Make the vendored `droid/` package (repo root) and `src/` (constants) importable
# when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import tyro
from moviepy import ImageSequenceClip
from openpi_client import websocket_client_policy
from PIL import Image

from constants import DEFAULT_WRIST_INTRINSICS
from droid.robot_env import RobotEnv

faulthandler.enable()

# Terminal colors for status prints.
GREEN = "\033[0;32m"
BLUE = "\033[1;34m"
RESET = "\033[0m"

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

# RLDS images are 320x180; ZED HD native is 1280x720.
_CAM_SCALE = 320 / 1280  # 0.25

_intrinsics_cache: dict[tuple[str, str], np.ndarray] = {}


def _short_config_name(config: str) -> str:
    """e.g. 'pi05_full_droid_finetune_v3-1_ik' -> 'v3-1_ik'."""
    if not config:
        return ""
    if "finetune_" in config:
        return config.split("finetune_", 1)[1]
    return config


def _short_checkpoint_name(checkpoint_dir: str) -> str:
    """e.g. '.../finetune_v0/exp/10000'   -> 'v0_10k';
            '.../finetune_v3-1_100000'    -> 'v3-1_100k';
            'gs://.../pi05_droid'         -> 'pi05_droid'."""
    if not checkpoint_dir:
        return ""
    parts = checkpoint_dir.rstrip("/").split("/")
    basename = parts[-1]
    # Step = trailing run of digits in the basename, anchored at the start or after '_'.
    m = re.search(r"(?:^|_)(\d+)$", basename)
    if m is None:
        return basename
    step = int(m.group(1))
    step_str = f"{step // 1000}k" if step >= 1000 and step % 1000 == 0 else str(step)
    # Version = portion after 'finetune_' in any path component, with any trailing
    # '_<step>' stripped off (covers single-dir layouts like '.../finetune_v3-1_100000').
    version = ""
    for p in parts:
        if "finetune_" in p:
            version = re.sub(r"_\d+$", "", p.split("finetune_", 1)[1])
            break
    if not version:
        # Fallback: basename minus its trailing _<step>, then drop any 'checkpoints' leading parts.
        version = re.sub(r"_\d+$", "", basename) or (parts[-3] if len(parts) >= 3 else parts[0])
    return f"{version}_{step_str}"


def fetch_intrinsics(serial: str, lens: str = "left") -> np.ndarray:
    """Return [fx, fy, cx, cy] scaled to RLDS resolution. Queries Stereolabs calibration API."""
    cache_key = (serial, lens)
    if cache_key not in _intrinsics_cache:
        conf_path = f"/usr/local/zed/settings/SN{serial}.conf"
        cfg = configparser.ConfigParser()
        cfg.read(conf_path)
        section = f"{lens.upper()}_CAM_HD"
        if section not in cfg:
            raise RuntimeError(f"ZED calibration {section} not found at {conf_path}")
        hd = cfg[section]
        _intrinsics_cache[cache_key] = np.array(
            [
                float(hd["fx"]) * _CAM_SCALE,
                float(hd["fy"]) * _CAM_SCALE,
                float(hd["cx"]) * _CAM_SCALE,
                float(hd["cy"]) * _CAM_SCALE,
            ]
        )
    return _intrinsics_cache[cache_key]


def _resize_no_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Plain resize (no pad) to the training source resolution (180x320), so the
    image and the server-rendered gripper mask share the same source resolution."""
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


@dataclasses.dataclass
class Args:
    # ZED camera serial numbers (required) — set these in deploy_policy.sh to match
    # your cameras. Serials are the SN<serial> files under /usr/local/zed/settings/.
    external_camera_id: str
    wrist_camera_id: str

    # Which robot to drive: "sharpa" (arm + Sharpa hand) or "robotiq" (arm +
    # Robotiq 2F gripper). Must match the served checkpoint's embodiment.
    embodiment: str = "sharpa"  # "sharpa" or "robotiq"

    # Debug/mock mode: no robot, no cameras — default robot state + blank images.
    # Exercises the full client↔server loop against a running server, no hardware.
    debug: bool = False

    # Which physical lens of the wrist ZED to read (+ its intrinsics/extrinsics).
    # The wire key is "observation/wrist_image" regardless of the lens.
    wrist_cam_lens: str = "left"  # "left" or "right"

    # Rollout parameters
    max_timesteps: int = 900  # rollout length cap; <= 0 means run until Ctrl+C
    # Actions executed per predicted chunk before re-querying (8 ≈ 0.5s).
    open_loop_horizon: int = 8

    # Sharpa hand parameters
    hand_spec: str = "sharpa_R"  # "sharpa_R" or "sharpa_L"

    # Action space: "joint_position" (default) or "cartesian_position" (pose).
    action_space: str = "joint_position"

    # Remote server parameters
    remote_host: str = "127.0.1.1"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# Ctrl+C during a server call kills the connection, so delay it until the call
# returns. Lets the operator still stop a rollout early.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _poll_new_instruction() -> Optional[str]:
    """Non-blocking: return the latest instruction line typed on stdin, or None.

    Drains any complete lines waiting on stdin (so we never block the control
    loop) and returns the last non-empty one. Lets the operator retarget the
    policy mid-rollout.
    """
    new_instruction = None
    while select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        line = line.strip()
        if line:
            new_instruction = line
    return new_instruction


def _prompt_instruction(on_reset=None, on_cycle_gripper=None) -> Optional[str]:
    """Free-form startup prompt. Returns the typed instruction, or None iff the
    operator quit. Single-letter commands fire and re-prompt instead of returning:
      - ``r`` -> ``on_reset`` (re-home the arm)
      - ``g`` -> ``on_cycle_gripper`` (close→open the gripper/hand)
    ``q`` quits; empty input re-prompts; any other text is the instruction.
    """
    hints = ["q=quit"]
    if on_reset is not None:
        hints.append("r=reset")
    if on_cycle_gripper is not None:
        hints.append("g=cycle-gripper")
    options = ", ".join(hints)
    while True:
        instruction = input(f"Enter instruction [{options}]: ").strip()
        low = instruction.lower()
        if low == "q":
            return None
        if low == "r" and on_reset is not None:
            on_reset()
            continue
        if low == "g" and on_cycle_gripper is not None:
            on_cycle_gripper()
            continue
        if instruction:
            return instruction


def main(args: Args):
    assert args.embodiment in ("sharpa", "robotiq"), (
        f"--embodiment must be 'sharpa' or 'robotiq', got {args.embodiment}"
    )
    assert args.wrist_cam_lens in ("left", "right"), (
        f"--wrist-cam-lens must be 'left' or 'right', got {args.wrist_cam_lens}"
    )

    # Initialize the robot environment for the chosen embodiment.
    debug_camera_ids = [args.external_camera_id, args.wrist_camera_id]
    if args.embodiment == "sharpa":
        from droid.franka_sharpa_env import FrankaSharpaEnv

        env = FrankaSharpaEnv(
            hand_spec=args.hand_spec,
            action_space=args.action_space,
            use_cameras=True,
            do_reset=True,
            hand_camera_id=args.wrist_camera_id,
            debug=args.debug,
            debug_camera_ids=debug_camera_ids,
        )
        print(f"Created the FrankaSharpaEnv (arm + Sharpa hand) with action_space={args.action_space}!")
    else:
        env = RobotEnv(
            action_space=args.action_space,
            gripper_action_space="position",
            use_cameras=True,
            do_reset=True,
            hand_camera_id=args.wrist_camera_id,
            debug=args.debug,
            debug_camera_ids=debug_camera_ids,
        )
        print(f"Created the RobotEnv (arm + Robotiq gripper) with action_space={args.action_space}!")
    if args.debug:
        print(f"{BLUE}=== DEBUG MODE: mock robot + blank cameras, no hardware ==={RESET}")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    server_meta = policy_client.get_server_metadata() or {}
    config_short = _short_config_name(server_meta.get("config", ""))
    ckpt_short = _short_checkpoint_name(server_meta.get("checkpoint_dir", ""))
    deploy_tag = "_".join(t for t in (config_short, ckpt_short) if t)
    if deploy_tag:
        print(f"Serving deploy: {deploy_tag}")

    video_dir = Path("./test_videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Startup-prompt callbacks: re-home the arm, or close→open the gripper/hand so
    # the operator can confirm it responds before a rollout.
    on_reset = env.reset
    on_cycle_gripper = env.cycle_gripper

    while True:
        env.reset()
        instruction = _prompt_instruction(on_reset=on_reset, on_cycle_gripper=on_cycle_gripper)
        if instruction is None:
            break

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        step_iter = itertools.count() if args.max_timesteps <= 0 else range(args.max_timesteps)
        print("Running rollout... press Ctrl+C to stop early.")
        print("Type a new instruction + Enter at any time to retarget the policy mid-rollout.")
        for t_step in step_iter:
            start_time = time.time()
            try:
                # Pick up a new instruction the operator typed mid-rollout
                # (non-blocking). Takes effect on the next chunk.
                typed = _poll_new_instruction()
                if typed is not None:
                    instruction = typed
                    print(f"New instruction: {instruction!r}")

                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                # Top-bottom frame: external camera over wrist camera
                side_by_side = np.concatenate(
                    [curr_obs["external_image"], curr_obs["wrist_image"]], axis=0
                )
                video.append(side_by_side)

                # Predict a new action chunk when it's time
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # Resize to 180x320 (training source res) to cut latency, no pad
                    # so image and server-side gripper mask share the same source res.
                    # _extract_observation asserts the camera params, so fill unconditionally.
                    request_data = {
                        "observation/exterior_image_1_left": _resize_no_pad(
                            curr_obs["external_image"], 180, 320
                        ),
                        "observation/wrist_image": _resize_no_pad(curr_obs["wrist_image"], 180, 320),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        # Camera intrinsics for the server-side gripper mask. The
                        # cam-to-gripper extrinsic is added below; the server
                        # FK-composes cam-to-base from joint_position.
                        "observation/camera_intrinsic": curr_obs["camera_intrinsic"],
                        "prompt": instruction,
                    }
                    # Hand joint position for the Sharpa embodiment.
                    if curr_obs["hand_joint_position"] is not None:
                        request_data["observation/hand_joint_position"] = curr_obs["hand_joint_position"]
                    # Raw cam-to-gripper; the server FK-composes it with the EE
                    # pose to get cam-to-base.
                    request_data["observation/wrist_extrinsic_cam_to_gripper"] = (
                        np.asarray(curr_obs["camera_extrinsic_cam_to_gripper"], dtype=np.float64)
                    )

                    # Delay Ctrl+C until the server call returns (see contextmanager).
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                # Select current action to execute from chunk. Sharpa hand emits
                # 29-dim actions (7 arm + 22 hand qpos) — passed through as-is.
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # Tell the server the episode is over so it flushes its per-episode state
        # (IK warm-starts) before we prompt/encode.
        policy_client.signal_episode_done()

        # === Save video ===
        # Prompt default-no so junk rollouts don't clutter ./test_videos.
        should_save_video = input("Save video? [y/N]: ").strip().lower() == "y"
        if should_save_video:
            deploy_part = f"{deploy_tag}_" if deploy_tag else ""
            label = input("Label for video filename: ").strip().replace(" ", "_")
            label_part = f"{label}_" if label else ""
            save_filename = f"{deploy_part}{label_part}{timestamp}"
            save_path = str(video_dir / f"{save_filename}.mp4")
            ImageSequenceClip(list(np.stack(video)), fps=10).write_videofile(save_path, codec="libx264")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    external_image, wrist_image = None, None
    for key in image_observations:
        # "left" = left lens of the stereo pair; the model only trains on those.
        if args.external_camera_id in key and "left" in key:
            external_image = image_observations[key]
        elif args.wrist_camera_id in key and f"_{args.wrist_cam_lens}" in key:
            wrist_image = image_observations[key]

    assert wrist_image is not None, (
        f"Wrist camera {args.wrist_camera_id} (lens {args.wrist_cam_lens}) not found in observation"
    )
    assert external_image is not None, (
        f"External camera {args.external_camera_id} not found in observation"
    )

    # Drop the alpha dimension and convert to RGB
    external_image = external_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    # Proprioceptive state
    robot_state = obs_dict["robot_state"]
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Hand joint position from Sharpa hand (when using FrankaSharpaEnv)
    hand_joint_position = None
    if "hand_state" in obs_dict:
        hand_joint_position = np.array(obs_dict["hand_state"]["joint_positions"])

    # Save a combined image to disk for live viewing while the robot runs.
    if save_to_disk:
        combined_image = np.concatenate([external_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    # Static wrist cam-to-gripper extrinsic for the server-side gripper mask.
    # get_camera_extrinsics exposes it under the '{serial}_{lens}_gripper_offset'
    # suffix; the server FK-composes cam-to-base from joint_position.
    camera_extrinsic_cam_to_gripper = None
    if "camera_extrinsics" in obs_dict:
        for key, val in obs_dict["camera_extrinsics"].items():
            if (args.wrist_camera_id in key
                    and key.endswith(f"_{args.wrist_cam_lens}_gripper_offset")):
                camera_extrinsic_cam_to_gripper = np.array(val)
    assert camera_extrinsic_cam_to_gripper is not None, (
        f"Wrist cam-to-gripper extrinsic for {args.wrist_camera_id}_{args.wrist_cam_lens} not found"
    )

    # Intrinsics from the Stereolabs calibration API (same source as preprocess_data.py).
    # In debug mode there's no ZED conf on disk, so use the DROID-mean default.
    if args.debug:
        camera_intrinsic = DEFAULT_WRIST_INTRINSICS
    else:
        camera_intrinsic = fetch_intrinsics(args.wrist_camera_id, args.wrist_cam_lens)

    return {
        "external_image": external_image,
        "wrist_image": wrist_image,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
        "camera_extrinsic_cam_to_gripper": camera_extrinsic_cam_to_gripper,
        "camera_intrinsic": camera_intrinsic,
        "hand_joint_position": hand_joint_position,
    }


if __name__ == "__main__":
    print(f"{GREEN}Running: python {' '.join(sys.argv)}{RESET}")
    args: Args = tyro.cli(Args)
    main(args)
