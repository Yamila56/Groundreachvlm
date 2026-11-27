"""Regenerate a balanced control set with the pinned upstream simulator recipe."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from experiments.physvlm_reproduction.run_cpu_pilot import SIMULATOR


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-start", type=int, default=11)
    parser.add_argument("--per-label", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=50)
    parser.add_argument("--robot", nargs="+", choices=("PANDA", "UR5", "FR5", "CR5"),
                        default=("PANDA", "UR5", "FR5", "CR5"))
    args = parser.parse_args()
    if args.per_label < 1 or args.max_scenes < 1 or args.seed_start < 0:
        parser.error("Positive sample sizes and a nonnegative seed are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(SIMULATOR)
    sys.path.insert(0, str(SIMULATOR))
    from generate_sp_map import DepthMapProcessor
    from main import DH_PARAMETERS, URDFDataset, run_simulation

    dataset = URDFDataset()
    if not dataset.load_dataset():
        raise RuntimeError("Could not load upstream objects")
    selected = {"Y": [], "N": []}
    for robot in args.robot:
        processor = DepthMapProcessor([])
        for index in range(args.max_scenes):
            if all(len(rows) >= args.per_label for rows in selected.values()):
                break
            seed = args.seed_start + index
            random.seed(seed)
            np.random.seed(seed)
            label = run_simulation(index, robot, dataset, str(SIMULATOR / "URDF_100models"),
                                   DH_PARAMETERS[robot], str(output))
            if label is None:
                continue
            take = [(key, sorted(set(label[source]))[0]) for key, source in (("Y", "true"), ("N", "false"))
                    if label[source] and len(selected[key]) < args.per_label]
            if not take:
                continue
            stem = f"{robot}_"
            rgb = output / f"{stem}rgb_{index}.jpg"
            spatial_map = output / f"{stem}sp_map_{index}.jpg"
            meta = json.loads((output / f"{stem}data_{index}.json").read_text())
            entry = dict(raw_path=str(output / f"{stem}depth_{index}.png"),
                         rgb_path=str(rgb), sp_map_save_path=str(spatial_map),
                         rgb_map_save_path=str(output / f"{stem}rgb_map_{index}.jpg"),
                         pcb_file=str(SIMULATOR / "arm_ws_pcb" / f"{robot}_ws.pcd"),
                         rotation_matrix=meta["camera_to_base_rotation"],
                         translation_vector=meta["camera_to_base_translation"],
                         camera_intrinsics=meta["camera_intrinsics"], workspace_radius=0.05)
            processor.datas = [entry]
            processor.initialize_kdtree()
            processor.process_image_single(entry)
            for truth, name in take:
                selected[truth].append(dict(id=f"{robot}-{index:02d}-{truth}", robot=robot,
                    image=rgb.name, sp_map=spatial_map.name, image_sha256=sha(rgb),
                    map_sha256=sha(spatial_map), question=f"Is the {name} in the robot's reachable space?",
                    truth=truth, scene_seed=seed))
    rows = sorted(selected["Y"] + selected["N"], key=lambda row: row["id"])
    manifest = dict(source="regenerated from pinned EQA-phys simulator recipe",
                    upstream="37e25fd83d8b9f0bdb425661e2dcdffa674a79c7",
                    counts={key: len(items) for key, items in selected.items()}, rows=rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["counts"], flush=True)
    if min(manifest["counts"].values()) < args.per_label:
        raise RuntimeError("Insufficient balanced recipe queries in the requested scene range")


if __name__ == "__main__":
    main()
