# NUC package reference

Follow the [setup guide](https://tml.stanford.edu/cloak/setup/) for assembly and deployment. This reference describes the container dependency contract.

## Base image requirements

Supply `BASE_IMAGE` explicitly when building. Use the existing compatible image from a working DROID installation. The historical `ghcr.io/droid-dataset/droid_nuc:panda` tag returned unauthorized during the release check; it is not a verified public download. Local image IDs in `backend-manifest.json` identify our tested environment but cannot be pulled from a registry.

The base must supply:

- Linux real-time robot support and libfranka 0.9.0 for the tested Panda setup; verify compatibility for your controller firmware.
- DROID at `/app/droid`, with Fairo/Polymetis at `/app/droid/fairo/polymetis/polymetis`.
- Conda at `/root/miniconda3`, environment `polymetis-local`, with torch, numpy, grpc, zerorpc, Polymetis and DROID modules.
- Executables `launch_robot.py`, `launch_gripper.py`, the compiled Franka arm and native-hand clients, and the Robotiq Python driver source targeted by the patch.
- The `patch` utility. Dockerfile patch application fails if the base source is incompatible; do not bypass that failure.

The Compose mounts supply the Cloak bridge and configuration without modifying the base image. Robot address and serial device are supplied as environment settings. Only one controller container should run against the robot at a time.

## Validation status

The replacement package was functionally exercised with Robotiq and UMI, and arm-only mode with Sharpa. The final reconciled release has offline coverage; final hardware regression is pending. Intermittent controller communication faults remain under investigation. The saved native-hand source and binary provenance have not been reconciled into a clean base-image build. A reproducible fresh-base distribution remains future work.

The optional `check_hand_contact.py` requires the native hand driver already running. It prompts before finger movement and never creates an arm controller. `scripts/check_umi_contact.sh` accepts `NUC_SSH_HOST` and optional `NUC_CONTAINER`; it is an operator tool, not an automated test.
