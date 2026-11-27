"""Replay the reachability certificates in a frozen fixture selection."""

import json

import numpy as np
import pybullet as bullet

from groundreach.geometry.reachability import workspace_bound
from groundreach.io import sha256_file as digest


def audit(selection_path, urdf, tool_link, tolerance):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    hashes = set()
    reached_errors = []
    negative_margins = []
    uncertified = 0
    identical_stable = 0
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
            target = entry["target"] if "target" in entry else next(iter(pair["pairs"]))
            truth = pair["pairs"][target]
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
                if label == "U":
                    origin, radius = workspace_bound(base, urdf, tool_link)
                    error = evidence["fk_error_m"]
                    if np.linalg.norm(point-origin) > radius+tolerance or (error is not None and error <= tolerance):
                        raise ValueError(f"Certified point marked unknown at {entry['scene']}/{side}")
                    uncertified += 1
                    continue
                if label == "N":
                    origin, radius = workspace_bound(base, urdf, tool_link)
                    margin = np.linalg.norm(point - origin) - radius
                    if margin <= tolerance:
                        raise ValueError(f"Insufficient outer-bound margin at {entry['scene']}/{side}")
                    negative_margins.append(float(margin))
                    continue
                robot = bullet.loadURDF(str(urdf), basePosition=base, useFixedBase=True)
                info = [bullet.getJointInfo(robot, i) for i in range(bullet.getNumJoints(robot))]
                tool = next(joint[0] for joint in info if joint[12].decode() == tool_link)
                actuated = [joint for joint in info if joint[2] != bullet.JOINT_FIXED]
                for joint, angle in zip(actuated, evidence["joints"], strict=True):
                    if not joint[8] - 1e-6 <= angle <= joint[9] + 1e-6:
                        raise ValueError(f"Joint limit exceeded at {entry['scene']}/{side}")
                    bullet.resetJointState(robot, joint[0], angle)
                fk = bullet.getLinkState(robot, tool, computeForwardKinematics=True)[4]
                error = np.linalg.norm(np.asarray(fk) - point)
                if error > tolerance:
                    raise ValueError(f"FK mismatch at {entry['scene']}/{side}")
                reached_errors.append(float(error))
                bullet.removeBody(robot)
            if len(map_digests) != 2:
                if not entry.get("allow_identical_stable_maps") or truth not in ("YY", "NN"):
                    raise ValueError(f"Identical workspace maps at {entry['scene']}")
                identical_stable += 1
    finally:
        bullet.disconnect(client)
    return {
        "selection_sha256": digest(selection_path), "pairs": len(hashes),
        **({"identical_stable_maps": identical_stable} if identical_stable else {}),
        "reachable_witnesses": len(reached_errors), "max_fk_error_m": max(reached_errors, default=None),
        "unreachable_bounds": len(negative_margins), "minimum_bound_margin_m": min(negative_margins, default=None),
        **({"unknown": uncertified} if uncertified else {}),
    }


def certify(base, targets, urdf, tool_link, tolerance):
    robot = bullet.loadURDF(str(urdf), basePosition=base, useFixedBase=True)
    joints = [bullet.getJointInfo(robot, i) for i in range(bullet.getNumJoints(robot))]
    tool = next(joint[0] for joint in joints if joint[12].decode() == tool_link)
    moving = [joint for joint in joints if joint[2] != bullet.JOINT_FIXED]
    lower = np.array([joint[8] for joint in moving])
    upper = np.array([joint[9] for joint in moving])
    rng = np.random.default_rng(17)
    origin, radius = workspace_bound(base, urdf, tool_link)
    starts = np.vstack([(lower + upper) / 2, rng.uniform(lower, upper, size=(15, len(moving)))])

    def reset(values):
        for joint, value in zip(moving, values, strict=True):
            bullet.resetJointState(robot, joint[0], value)
        return np.array(bullet.getLinkState(robot, tool, computeForwardKinematics=True)[4])

    for start in starts:
        if np.linalg.norm(reset(start) - origin) > radius + 1e-6:
            raise RuntimeError("FK witness violates the certified upper bound")
    results = {}
    for name, target in targets.items():
        point = np.asarray(target["position_m"])
        distance = np.linalg.norm(point - origin)
        evidence = {"bound_origin_m": origin.tolist(), "radius_m": radius, "distance_m": distance}
        if distance > radius + tolerance:
            results[name] = {"label": "N", "source": "URDF outer bound", **evidence}
            continue
        best = {"label": "U", "source": "IK not certified", "fk_error_m": None}
        for start in starts:
            reset(start)
            angles = np.array(bullet.calculateInverseKinematics(
                robot, tool, point, lowerLimits=lower.tolist(), upperLimits=upper.tolist(),
                jointRanges=(upper - lower).tolist(), restPoses=start.tolist(),
                maxNumIterations=400, residualThreshold=1e-5,
            ))
            if np.any(angles < lower - 1e-6) or np.any(angles > upper + 1e-6):
                continue
            error = np.linalg.norm(reset(angles) - point)
            if best["fk_error_m"] is None or error < best["fk_error_m"]:
                best = {"label": "Y" if error <= tolerance else "U",
                        "source": "joint-limited IK and FK", "fk_error_m": error, "joints": angles.tolist()}
        results[name] = {**evidence, **best}
    bullet.removeBody(robot)
    return results
