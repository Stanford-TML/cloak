import os

# IP of the NUC running the Franka controller, read from the environment so no
# site-specific address is committed. If unset (None), RobotEnv launches the
# Franka controller locally instead of connecting to a NUC over the network.
# See the "Deployment" section of the README.
nuc_ip = os.environ.get("CLOAK_NUC_IP")
