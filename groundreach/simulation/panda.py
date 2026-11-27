"""Render paired Panda scenes with the pinned PhysVLM simulator."""

import hashlib
import json

import cv2
import numpy as np
import pybullet as bullet

from groundreach.geometry.certificates import certify
from groundreach.vision.physvlm_map import render_sp_map
from groundreach.simulation.upstream import SIMULATOR, TOOL_LINK, TOLERANCE, URDF


def create_box(center, half_extents, color):
    visual = bullet.createVisualShape(bullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=center)


def render_scene(output, objects_layout=None, *, camera_yaw=0.0, shapes=None, point_quantile=None):
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
    half_extents = np.array([0.04, 0.045, 0.045])
    objects = {}
    for name, (center, color) in objects_layout.items():
        shape = shapes.get(name, "box") if shapes else "box"
        if shape == "box":
            objects[name] = create_box(center, half_extents.tolist(), color)
        else:
            options = {"sphere": (bullet.GEOM_SPHERE, dict(radius=.045)),
                       "cylinder": (bullet.GEOM_CYLINDER, dict(radius=.045, length=.09))}
            kind, kwargs = options[shape]
            visual = bullet.createVisualShape(kind, rgbaColor=color, **kwargs)
            objects[name] = bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=center)
    _, _, rgba, depth_buffer, segmentation = bullet.getCameraImage(
        width, height, view, projection, renderer=bullet.ER_TINY_RENDERER,
    )
    depth_m = near * far / (far - (far - near) * np.asarray(depth_buffer))
    for filename, image in (("rgb.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)),
                            ("depth.png", np.rint(depth_m * 1000).astype(np.uint16))):
        if not cv2.imwrite(str(output / filename), image):
            raise OSError(filename)
    targets = {}
    for index, (name, body) in enumerate(objects.items()):
        mask = np.asarray(segmentation) == body
        pixels = np.argwhere(mask)
        center_pixel = pixels.mean(axis=0)
        if point_quantile is not None:
            center_pixel[1] = np.quantile(pixels[:, 1], point_quantile)
        y, x = pixels[np.argmin(np.sum((pixels - center_pixel) ** 2, axis=1))]
        z = depth_m[y, x]
        point = eye + rotation @ np.array([(width / 2 - x) * z / focal, (height / 2 - y) * z / focal, z])
        local = np.abs(point - np.array(objects_layout[name][0]))
        pixel_tolerance = max(0.002, 2 * z / focal)
        if not shapes and (np.any(local > half_extents + pixel_tolerance)
                           or np.min(np.abs(local - half_extents)) > pixel_tolerance):
            raise RuntimeError(f"Camera backprojection missed {name}: {point}")
        targets[name] = {"pixel_xy": [int(x), int(y)], "position_m": point.tolist()}
        if shapes:
            top, left = pixels.min(axis=0)
            bottom, right = pixels.max(axis=0) + 1
            filename = f"mask-{index}.png"
            if not cv2.imwrite(str(output / filename), mask.astype(np.uint8) * 255):
                raise OSError(filename)
            targets[name].update(box=[int(left), int(top), int(right), int(bottom)], mask=filename)
    return eye, rotation, intrinsics, targets


def generate_pair(output, objects_layout=None, target_name=None, scene_id=None, bases=None, *,
                  camera_yaw=0.0, shapes=None, point_quantile=None):
    if not URDF.is_file():
        raise FileNotFoundError("Set PHYSVLM_UPSTREAM to the pinned PhysVLM simulator checkout")
    client = bullet.connect(bullet.DIRECT)
    try:
        eye, rotation, intrinsics, targets = render_scene(
            output, objects_layout, camera_yaw=camera_yaw, shapes=shapes, point_quantile=point_quantile,
        )
        if target_name is not None:
            targets = {target_name: targets[target_name]}
        configurations = {}
        if bases is None:
            bases = ([0, 0, 0.20], [0.30, 0, 0.20])
        for key, base in zip(("A", "B"), map(np.asarray, bases), strict=True):
            labels = certify(base, targets, URDF, TOOL_LINK, TOLERANCE)
            workspace = render_sp_map(output, key, eye, rotation, intrinsics, base, SIMULATOR)
            rules = {}
            for name, target in targets.items():
                relative = np.asarray(target["position_m"]) - base
                nearest, _ = workspace.query(relative)
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
