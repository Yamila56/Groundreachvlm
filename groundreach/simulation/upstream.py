"""Resolve assets supplied by the separate PhysVLM checkout."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("PHYSVLM_UPSTREAM", str(ROOT))).resolve()
SIMULATOR = UPSTREAM / "EQA-phys-simulator"
URDF = SIMULATOR / "my_robot_urdf" / "panda" / "panda_arm.urdf"
TOOL_LINK = "panda_link7"
TOLERANCE = 0.01
