"""Generate same-image development pairs and independent Panda reach certificates."""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pybullet as bullet

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("PHYSVLM_UPSTREAM", str(ROOT))).resolve()
SIMULATOR = UPSTREAM / "EQA-phys-simulator"
URDF = SIMULATOR / "my_robot_urdf" / "panda" / "panda_arm.urdf"
TOOL_LINK = "panda_link7"
TOLERANCE = 0.01


def workspace_bound(base):
    joints = {
        joint.find("child").get("link"): joint for joint in ET.parse(URDF).getroot().findall("joint")
    }
    chain = []
    link = TOOL_LINK
    while link in joints:
        joint = joints[link]
        if joint.get("type") not in {"revolute", "fixed"}:
            raise ValueError("Panda bound requires a chain without prismatic joints")
        origin = joint.find("origin")
        chain.append(np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "))
        link = joint.find("parent").get("link")
    return base + chain[-1], sum(np.linalg.norm(offset) for offset in chain[:-1])


def create_box(center, half_extents, color):
    visual = bullet.createVisualShape(bullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=center)


def render_scene(output, objects_layout=None, *, camera_yaw=0.0):
    width, height = 640, 480
    near, far = 0.2, 3.0
    eye = np.array([0.65, 0.0, 1.55] if objects_layout is None else [0.26, 0.0, 2.45])
    # Camera coordinates as used by the upstream depth processor.
    rotation = np.diag([-1.0, 1.0, -1.0])
    fov = 58
    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsics = {"fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2}
    up = [0, 1, 0]
    if camera_yaw:
        theta = np.deg2rad(camera_yaw)
        c, s = np.cos(theta), np.sin(theta)
        yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotation = yaw_rotation @ rotation
        up = (yaw_rotation @ up).tolist()
    view = bullet.computeViewMatrix(eye, [eye[0], eye[1], 0], up)
    projection = bullet.computeProjectionMatrixFOV(fov, width / height, near, far)
    table_center = [0.70, 0, 0.18] if objects_layout is None else [0.26, 0.0, 0.18]
    table_size = [0.51, 0.47, 0.02] if objects_layout is None else [1.32, 0.51, 0.02]
    create_box(table_center, table_size, [0.62, 0.60, 0.55, 1])
    if objects_layout is None:
        objects_layout = {
            "red block": ([1.00, 0.08, 0.245], [0.85, 0.15, 0.12, 1]),
            "blue block": ([0.54, -0.17, 0.245], [0.13, 0.30, 0.81, 1]),
        }
    objects = {
        name: create_box(center, [0.04, 0.045, 0.045], color)
        for name, (center, color) in objects_layout.items()
    }
    _, _, rgba, depth_buffer, segmentation = bullet.getCameraImage(
        width, height, view, projection, renderer=bullet.ER_TINY_RENDERER,
    )
    depth_m = near * far / (far - (far - near) * np.asarray(depth_buffer))
    for filename, image in (("rgb.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)),
                            ("depth.png", np.rint(depth_m * 1000).astype(np.uint16))):
        if not cv2.imwrite(str(output / filename), image):
            raise OSError(filename)
    targets = {}
    for name, body in objects.items():
        pixels = np.argwhere(np.asarray(segmentation) == body)
        y, x = pixels[np.argmin(np.sum((pixels - pixels.mean(axis=0)) ** 2, axis=1))]
        z = depth_m[y, x]
        point = eye + rotation @ np.array([(width / 2 - x) * z / focal, (height / 2 - y) * z / focal, z])
        local = np.abs(point - np.array(objects_layout[name][0]))
        half_extents = np.array([0.04, 0.045, 0.045])
        pixel_tolerance = max(0.002, 2 * z / focal)
        if np.any(local > half_extents + pixel_tolerance) or np.min(np.abs(local - half_extents)) > pixel_tolerance:
            raise RuntimeError(f"Camera backprojection missed {name}: {point}")
        targets[name] = {"pixel_xy": [int(x), int(y)], "position_m": point.tolist()}
    return eye, rotation, intrinsics, targets


def certify(base, targets):
    robot = bullet.loadURDF(str(URDF), basePosition=base, useFixedBase=True)
    joints = [bullet.getJointInfo(robot, i) for i in range(bullet.getNumJoints(robot))]
    tool = next(joint[0] for joint in joints if joint[12].decode() == TOOL_LINK)
    moving = [joint for joint in joints if joint[2] != bullet.JOINT_FIXED]
    lower = np.array([joint[8] for joint in moving])
    upper = np.array([joint[9] for joint in moving])
    rng = np.random.default_rng(17)
    origin, radius = workspace_bound(base)
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
        if distance > radius + TOLERANCE:
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
                best = {"label": "Y" if error <= TOLERANCE else "U",
                        "source": "joint-limited IK and FK", "fk_error_m": error, "joints": angles.tolist()}
        results[name] = {**evidence, **best}
    bullet.removeBody(robot)
    return results


def generate_pair(output, objects_layout=None, target_name=None, scene_id=None, bases=None, *, camera_yaw=0.0):
    import generate_sp_map
    from generate_sp_map import DepthMapProcessor

    client = bullet.connect(bullet.DIRECT)
    try:
        eye, rotation, intrinsics, targets = render_scene(output, objects_layout, camera_yaw=camera_yaw)
        if target_name is not None:
            targets = {target_name: targets[target_name]}
        configurations = {}
        if bases is None:
            bases = ([0, 0, 0.20], [0.30, 0, 0.20])
        for key, base in zip(("A", "B"), map(np.asarray, bases), strict=True):
            labels = certify(base, targets)
            entry = {
                "raw_path": str(output / "depth.png"), "rgb_path": str(output / "rgb.png"),
                "sp_map_save_path": str(output / f"sp_map_{key}.png"),
                "rgb_map_save_path": str(output / f"overlay_{key}.png"),
                "pcb_file": str(SIMULATOR / "arm_ws_pcb" / "PANDA_ws.pcd"),
                "rotation_matrix": rotation.tolist(), "translation_vector": (eye - base).tolist(),
                "camera_intrinsics": intrinsics, "workspace_radius": 0.05,
            }
            processor = DepthMapProcessor([entry])
            if generate_sp_map.global_kdtree is None:
                processor.initialize_kdtree()
            processor.process_image_single(entry)
            rules = {}
            for name, target in targets.items():
                relative = np.asarray(target["position_m"]) - base
                nearest, _ = generate_sp_map.global_kdtree.query(relative)
                rules[name] = {"workspace_distance_m": float(nearest),
                               "workspace_label": "Y" if nearest <= 0.05 else "N",
                               "radial_label": "Y" if np.linalg.norm(relative) <= 0.905 else "N"}
            configurations[key] = {"base_m": base.tolist(), "sp_map": f"sp_map_{key}.png",
                                   "targets": labels, "rules": rules}
        pairs = {name: "".join(config["targets"][name]["label"] for config in configurations.values())
                 for name in targets}
        report = {
            "scope": "single development scene, no VLM inference", "rgb": "rgb.png",
            "rgb_sha256": hashlib.sha256((output / "rgb.png").read_bytes()).hexdigest(),
            "camera": {"intrinsics": intrinsics, "origin_m": eye.tolist(), "rotation": rotation.tolist()},
            "targets": targets, "configurations": configurations, "pairs": pairs,
        }
        if scene_id is not None:
            report["scene_id"] = scene_id
            report["scope"] = "simulation with paired workspace poses"
        (output / "paired_pilot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return pairs
    finally:
        bullet.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SIMULATOR))
    pairs = generate_pair(output)
    if pairs != {"red block": "NY", "blue block": "YY"}:
        raise RuntimeError(f"Expected one flip and one invariant, obtained {pairs}")
    print(json.dumps(pairs))


if __name__ == "__main__":
    main()
