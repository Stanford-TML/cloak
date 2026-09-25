import os

# Off-screen MuJoCo rendering; must be set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
