"""Replay the reachability certificates in a frozen fixture selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as bullet

from run_paired_pilot import TOOL_LINK, TOLERANCE, URDF, workspace_bound


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    hashes = set()
    reached_errors = []
    negative_margins = []
    client = bullet.connect(bullet.DIRECT)
    try:
        for entry in selection["selection"]:
            path = selection_path.parent / entry["fixture"]
            pair = json.loads(path.read_text(encoding="utf-8"))
            rgb_hash = digest(path.parent / pair["rgb"])
            if rgb_hash != pair["rgb_sha256"] or rgb_hash in hashes:
                raise ValueError(f"Missing or reused image identity at {entry['scene']}")
            hashes.add(rgb_hash)
            if pair["scene_id"] != entry["scene"]:
                raise ValueError(f"Unexpected scene ID at {entry['scene']}")
            target, truth = next(iter(pair["pairs"].items()))
            if truth != entry["label"]:
                raise ValueError(f"Unexpected label at {entry['scene']}")
            point = np.array(pair["targets"][target]["position_m"])
            map_digests = set()
            for side, label in zip(("A", "B"), truth, strict=True):
                config = pair["configurations"][side]
                map_digests.add(digest(path.parent / config["sp_map"]))
                base = np.array(config["base_m"])
                evidence = config["targets"][target]
                if label != evidence["label"]:
                    raise ValueError(f"Label disagreement at {entry['scene']}/{side}")
                if label == "N":
                    origin, radius = workspace_bound(base)
                    margin = np.linalg.norm(point - origin) - radius
                    if margin <= TOLERANCE:
                        raise ValueError(f"Insufficient outer-bound margin at {entry['scene']}/{side}")
                    negative_margins.append(float(margin))
                    continue
                robot = bullet.loadURDF(str(URDF), basePosition=base, useFixedBase=True)
                info = [bullet.getJointInfo(robot, i) for i in range(bullet.getNumJoints(robot))]
                tool = next(joint[0] for joint in info if joint[12].decode() == TOOL_LINK)
                actuated = [joint for joint in info if joint[2] != bullet.JOINT_FIXED]
                for joint, angle in zip(actuated, evidence["joints"], strict=True):
                    if not joint[8] - 1e-6 <= angle <= joint[9] + 1e-6:
                        raise ValueError(f"Joint limit exceeded at {entry['scene']}/{side}")
                    bullet.resetJointState(robot, joint[0], angle)
                fk = bullet.getLinkState(robot, tool, computeForwardKinematics=True)[4]
                error = np.linalg.norm(np.asarray(fk) - point)
                if error > TOLERANCE:
                    raise ValueError(f"FK mismatch at {entry['scene']}/{side}")
                reached_errors.append(float(error))
                bullet.removeBody(robot)
            if len(map_digests) != 2:
                raise ValueError(f"Identical workspace maps at {entry['scene']}")
    finally:
        bullet.disconnect(client)
    return {
        "selection_sha256": digest(selection_path), "pairs": len(hashes),
        "reachable_witnesses": len(reached_errors), "max_fk_error_m": max(reached_errors, default=None),
        "unreachable_bounds": len(negative_margins), "minimum_bound_margin_m": min(negative_margins, default=None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.selection), indent=2))
