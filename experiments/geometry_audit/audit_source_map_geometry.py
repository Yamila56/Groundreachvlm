"""Replay source scenes to compare radial labels and true visible-target workspace pixels."""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree

from experiments.relational_grounding.grounding_readout import THRESHOLD
from experiments.physvlm_reproduction.prepare_official_box_probe import PREFIX, SUFFIX, normalized
from experiments.physvlm_reproduction.run_cpu_pilot import SIMULATOR

WORKSPACE_RADIUS = .05


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_distances(depth, camera, processor, tree):
    valid_depth = np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid_depth)
    intrinsics = camera["camera_intrinsics"]
    z = depth[y, x].astype(np.float32)
    points = np.stack(((intrinsics["cx"]-x)*z/intrinsics["fx"],
                       (intrinsics["cy"]-y)*z/intrinsics["fy"], z), axis=1)/1000
    points = processor.transform_points(points, np.asarray(camera["camera_to_base_rotation"]),
                                       np.asarray(camera["camera_to_base_translation"]))
    finite = np.isfinite(points).all(axis=1)
    radial = np.full(depth.shape, np.nan)
    nearest = np.full(depth.shape, np.nan)
    radial[y[finite], x[finite]] = np.linalg.norm(points[finite], axis=1)
    nearest[y[finite], x[finite]] = tree.query(points[finite])[0]
    return radial, nearest


def compare_map(reconstructed, jpeg, valid):
    coded = jpeg >= THRESHOLD
    return dict(valid_pixels=int(valid.sum()),
                true_positive=int((reconstructed & coded & valid).sum()),
                true_negative=int((~reconstructed & ~coded & valid).sum()),
                false_positive=int((~reconstructed & coded & valid).sum()),
                false_negative=int((reconstructed & ~coded & valid).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.manifest.resolve().parent, args.output.resolve()
    scenes = defaultdict(list)
    for row in json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]:
        scenes[row["image_sha256"]].append(row)
    os.chdir(SIMULATOR)
    sys.path.insert(0, str(SIMULATOR))
    from generate_sp_map import DepthMapProcessor
    from main import DH_PARAMETERS, URDFDataset, run_simulation
    from simulation_utils import MAX_REACH, ObjectManager, Robot
    import pybullet as bullet

    dataset = URDFDataset()
    if not dataset.load_dataset():
        raise RuntimeError("Could not load upstream objects")
    render, load_objects, reachable = bullet.getCameraImage, ObjectManager.load_objects, Robot.can_reach_position
    clouds = {robot: SIMULATOR/"arm_ws_pcb"/f"{robot}_ws.pcd" for robot in {r[0]["robot"] for r in scenes.values()}}
    workspace = {robot: cKDTree(DepthMapProcessor.load_pcb_point_cloud(str(path))) for robot, path in clouds.items()}
    generator = SIMULATOR/"offline_calculate_robot_ws.py"
    trace = {}

    def record_render(*args, **kwargs):
        result = render(*args, **kwargs)
        trace["segmentation"] = np.asarray(result[4]).reshape(result[1], result[0])
        return result

    def record_objects(manager, *args, **kwargs):
        result = load_objects(manager, *args, **kwargs)
        trace["objects"] = list(zip(*result, strict=True))
        return result

    def record_reach(robot, position, tolerance=.05):
        relative = (np.asarray(position)-robot.base_position) @ robot.get_base_orientation_matrix()
        result = reachable(robot, position, tolerance)
        trace["labels"].append((bool(result), relative, MAX_REACH[robot.robot_type]+tolerance))
        return result

    results, map_checks = [], []
    active_robot = None
    upstream_processor = DepthMapProcessor([])
    with (
        TemporaryDirectory() as temp,
        patch.object(bullet, "getCameraImage", record_render),
        patch.object(ObjectManager, "load_objects", record_objects),
        patch.object(Robot, "can_reach_position", record_reach),
    ):
        for queries in scenes.values():
            first = queries[0]
            robot = first["robot"]
            index = int(first["image"].removesuffix(".jpg").rsplit("_", 1)[1])
            random.seed(first["scene_seed"])
            np.random.seed(first["scene_seed"])
            trace.clear()
            trace["labels"] = []
            replay = run_simulation(index, robot, dataset, str(SIMULATOR/"URDF_100models"),
                                    DH_PARAMETERS[robot], temp)
            if not replay or file_hash(Path(temp)/first["image"]) != first["image_sha256"]:
                raise RuntimeError(f"RGB replay mismatch for {first['id']}")
            if file_hash(source/first["sp_map"]) != first["map_sha256"]:
                raise RuntimeError(f"Map hash mismatch for {first['id']}")
            depth_name = first["image"].replace("_rgb_", "_depth_").replace(".jpg", ".png")
            replay_depth = Path(temp)/depth_name
            if file_hash(replay_depth) != file_hash(source/depth_name):
                raise RuntimeError(f"Depth replay mismatch for {first['id']}")
            depth = cv2.imread(str(replay_depth), cv2.IMREAD_UNCHANGED)
            camera_name = first["image"].replace("_rgb_", "_data_").replace(".jpg", ".json")
            camera = json.loads((Path(temp)/camera_name).read_text(encoding="utf-8"))
            chroma = np.ptp(cv2.imread(str(source/first["sp_map"])).astype(float), axis=-1)
            radial, nearest = workspace_distances(depth, camera, DepthMapProcessor, workspace[robot])
            valid = np.isfinite(nearest)
            inside = nearest <= WORKSPACE_RADIUS
            regenerated = Path(temp)/first["sp_map"]
            entry = dict(raw_path=str(replay_depth), rgb_path=None, sp_map_save_path=str(regenerated),
                         pcb_file=str(clouds[robot]), rotation_matrix=camera["camera_to_base_rotation"],
                         translation_vector=camera["camera_to_base_translation"],
                         camera_intrinsics=camera["camera_intrinsics"], workspace_radius=WORKSPACE_RADIUS)
            if robot != active_robot:
                upstream_processor.datas = [entry]
                upstream_processor.initialize_kdtree()
                active_robot = robot
            upstream_processor.process_image_single(entry)
            map_checks.append(dict(image_sha256=first["image_sha256"], total_pixels=depth.size,
                                   byte_identical=file_hash(regenerated) == first["map_sha256"],
                                   **compare_map(inside, chroma, valid)))
            object_ids = trace["segmentation"] & ((1 << 24)-1)
            objects = list(zip(trace["objects"], trace["labels"], strict=True))
            for row in queries:
                phrase = row["question"][len(PREFIX):-len(SUFFIX)]
                matches = [(obj, record) for (obj, name), record in objects
                           if normalized(name) == phrase and record[0] == (row["truth"] == "Y")]
                if len(matches) != 1:
                    raise RuntimeError(f"Ambiguous replay target for {row['id']}")
                obj, (_, base, radius) = matches[0]
                instance = object_ids == obj
                mask = instance & valid
                distances = nearest[mask]
                origin_distance, _ = workspace[robot].query(base)
                radial_margin = radius-float(np.linalg.norm(base))
                if (radial_margin >= 0) != (row["truth"] == "Y"):
                    raise RuntimeError(f"Radial label mismatch for {row['id']}")
                ys, xs = np.nonzero(mask)
                point = None
                if len(xs):
                    i = int(np.argmin((xs-np.median(xs))**2 + (ys-np.median(ys))**2))
                    y, x = ys[i], xs[i]
                    point = dict(x=int(x), y=int(y), workspace_distance_m=float(nearest[y, x]),
                                 sampled_workspace_label="Y" if inside[y, x] else "N",
                                 jpeg_chroma_label="Y" if chroma[y, x] >= THRESHOLD else "N")
                results.append(dict(id=row["id"], truth=row["truth"], radial_label=row["truth"],
                                    point=point, instance_pixels=int(instance.sum()), pixels=len(distances),
                                    visible_pixels_check=compare_map(inside, chroma, mask),
                                    radial_margin_m=radial_margin,
                                    visible_median_radial_margin_m=radius-float(np.median(radial[mask]))
                                    if len(distances) else None,
                                    origin_workspace_distance_m=float(origin_distance),
                                    closest_visible_workspace_distance_m=float(distances.min()) if len(distances) else None,
                                    visible_in_workspace_fraction=float(inside[mask].mean()) if len(distances) else None,
                                    visible_median_chroma=float(np.median(chroma[mask])) if len(distances) else None))
            print(json.dumps(dict(replayed=len(results))), flush=True)
    labeled = defaultdict(dict)
    for label in ("Y", "N"):
        group = [r for r in results if r["truth"] == label]
        visible = [r for r in group if r["pixels"]]
        labeled[label] = dict(
            queries=len(group), visible=len(visible),
            origins_within_5cm=sum(r["origin_workspace_distance_m"] <= .05 for r in group),
            any_visible_within_5cm=sum(r["visible_in_workspace_fraction"] > 0 for r in visible),
            majority_visible_within_5cm=sum(r["visible_in_workspace_fraction"] >= .5 for r in visible),
            high_median_chroma=sum(r["visible_median_chroma"] >= THRESHOLD for r in visible),
            visible_median_inside_radial_label_radius=sum(r["visible_median_radial_margin_m"] >= 0 for r in visible),
            sampled_workspace_point_positive=sum(r["point"]["sampled_workspace_label"] == "Y" for r in visible),
            sampled_workspace_point_jpeg_agreement=sum(r["point"]["sampled_workspace_label"] ==
                                                       r["point"]["jpeg_chroma_label"] for r in visible),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    documented_generation = dict(generator_sha256=file_hash(generator), random_samples=1000000,
                                 joint_ranges_degrees=[-180, 180], voxel_size_m=.05,
                                 source="pinned upstream script defaults")
    report = dict(manifest_sha256=file_hash(source/args.manifest.name), scene_count=len(scenes),
                  workspace_pcd={robot: dict(sha256=file_hash(path), points=workspace[robot].n)
                                 for robot, path in clouds.items()},
                  documented_generation=documented_generation, workspace_radius_m=WORKSPACE_RADIUS,
                  chroma_threshold=THRESHOLD, by_label=labeled, map_checks=map_checks, rows=results)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
