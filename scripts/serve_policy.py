import dataclasses
import logging
import socket
import sys

import tyro

GREEN = "\033[92m"
RESET = "\033[0m"

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.policies import sharpa_ik_transform as _sharpa_ik
from openpi.policies import umi_ik_transform as _umi_ik
from openpi.policies import yam_ik_transform as _yam_ik
from openpi.policies.gripper_mask_transform import MaskAugmentation
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Per-embodiment IK module (for eagerly seeding the singleton with chunk_advance).
# robotiq is native (no IK), so it has no entry.
_IK_MODULES = {"sharpa": _sharpa_ik, "umi": _umi_ik, "yam": _yam_ik}


@dataclasses.dataclass(frozen=True)
class _DropExtras(_transforms.DataTransformFn):
    """Drop non-array passthrough keys before Policy.infer's jnp.asarray.

    TokenizePrompt now keeps the raw `prompt` string in the dict so the
    training data loader can surface it via `_EXTRA_KEYS` for offline
    analysis (commit 0f41268). On the serve path this string survives into
    Policy.infer and breaks `jax.tree.map(jnp.asarray, ...)`. Append this
    transform after the policy's input chain to drop those extras.
    """

    keys: tuple[str, ...] = ("prompt", "step_id")

    def __call__(self, data: dict) -> dict:
        return {k: v for k, v in data.items() if k not in self.keys}


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_full_droid_finetune_v3-1_ik").
    config: str
    # Checkpoint directory (e.g., "checkpoints/.../exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Serve a random-weight policy for the given config (no checkpoint).

    Testing only — actions are meaningless, but the full transform + IK + serve
    pipeline runs.
    """

    # Training config name.
    config: str = "pi05_full_droid_finetune_v3-1_ik"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Deploy embodiment: which robot this checkpoint is served to. Selects the
    # wrist mask renderer + IK retargeting (see policy_config). "robotiq" is
    # native (no retargeting). Must match the client (deploy_policy.py).
    embodiment: str = "robotiq"

    # The client's open_loop_horizon (= actions per chunk it executes before
    # requerying). Forwarded to the IK transform so its warm-start aligns with the
    # robot's real pose at chunk boundaries. Must match deploy_policy.py. Only used
    # when the config applies IK (use_ik) for a non-robotiq embodiment.
    open_loop_horizon: int = 8

    # Optional override for the Robotiq FK XML used by the IK transforms. None ->
    # module-level default in sharpa_ik_transform.py.
    robotiq_xml: str | None = None

    # Sharpa-only: if True, the output IK pins the 22-DOF hand qpos to a lerp of
    # SHARPA_HAND_QPOS_OPEN/CLOSED keyed by the per-step gripper scalar, and only
    # solves the 7 arm DOFs against the fingertip targets. See the `fixed_hand_ik`
    # docstring on SharpaIKTransform. Ignored for other embodiments.
    fixed_hand_sharpa_ik: bool = False

    # Specifies how to load the policy. If not provided, a random-weight policy is served (testing).
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# --- IK setup helpers ------------------------------------------------------
# The Sharpa IK transform self-seeds its warm-start and posture targets from the
# agreed reset pose via its constructor + lazy first-call ``_init_last_q``, so no
# explicit setup is needed beyond constructing the singleton with the right
# ``chunk_advance``.

def main(args: Args) -> None:
    # Resolve the train config once. All dispatch downstream (mask, IK) reads
    # ``train_config.inference`` + ``args.embodiment`` — no name parsing.
    config_name = args.policy.config
    train_config = _config.get_config(config_name)
    inf = train_config.inference

    if args.embodiment not in _policy_config.EMBODIMENTS:
        raise ValueError(
            f"Unknown --embodiment {args.embodiment!r}; expected one of {_policy_config.EMBODIMENTS}."
        )

    # Eagerly create the IK transform singleton before ``create_policy`` so it's
    # born with chunk_advance matched to the client's open_loop_horizon (the
    # data-config path also calls get_ik_transform() during create_policy;
    # first-call-wins on the singleton). Seeding from the canonical reset pose
    # means action[0] of the first chunk lands near the robot's actual pose
    # without any per-call adapter state. See src/openpi/constants.py. robotiq is
    # native (no IK) and has no _IK_MODULES entry.
    if inf.use_ik and args.embodiment in _IK_MODULES:
        ik_kwargs: dict = {"chunk_advance": args.open_loop_horizon}
        if args.robotiq_xml is not None:
            ik_kwargs["robotiq_xml"] = args.robotiq_xml
        if args.embodiment == "sharpa":
            ik_kwargs["fixed_hand_ik"] = args.fixed_hand_sharpa_ik
        _IK_MODULES[args.embodiment].get_ik_transform(**ik_kwargs)

    is_random = isinstance(args.policy, Default)
    policy = _policy_config.create_policy(
        train_config,
        checkpoint_dir=None if is_random else args.policy.dir,
        embodiment=args.embodiment,
        random_weights=is_random,
        default_prompt=args.default_prompt,
    )

    assert not any(
        isinstance(t, MaskAugmentation) for t in policy._input_transform.transforms
    ), "MaskAugmentation must not run at serve time."

    policy._input_transform = _transforms.compose([policy._input_transform, _DropExtras()])
    # Surface config + checkpoint identity to the client so it can tag its
    # debug videos / logs with which deploy produced them.
    policy_metadata = {
        **policy.metadata,
        "config": config_name,
        "checkpoint_dir": "<random-weights>" if is_random else args.policy.dir,
    }

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    print(f"{GREEN}Running: python {' '.join(sys.argv)}{RESET}")
    main(tyro.cli(Args))
