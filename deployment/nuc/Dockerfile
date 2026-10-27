# Build on your existing verified DROID image. See README.md for prerequisites.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY patches/robotiq-activation-status.patch /opt/cloak/robotiq-activation-status.patch
RUN cd /app/droid/fairo/polymetis/polymetis/python/polymetis/robot_client/robotiq_gripper && patch --batch --forward -p1 < /opt/cloak/robotiq-activation-status.patch
