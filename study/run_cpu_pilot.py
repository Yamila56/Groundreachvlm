"""Generate one CPU-only PhysVLM fixture with the upstream PyBullet and S-P Map code."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("PHYSVLM_UPSTREAM", str(ROOT))).resolve()
SIMULATOR = UPSTREAM / "EQA-phys-simulator"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=["PANDA", "UR5", "CR5", "FR5"], default="PANDA")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    sys.dont_write_bytecode = True
    os.chdir(SIMULATOR)
    sys.path.insert(0, str(SIMULATOR))

    import cv2
    from generate_sp_map import DepthMapProcessor
    from main import DH_PARAMETERS, URDFDataset, run_simulation

    assets = SIMULATOR / "URDF_100models"
    dataset = URDFDataset()
    if not dataset.load_dataset():
        raise RuntimeError("The upstream URDF dataset could not be loaded.")

    robot = args.robot
    label = run_simulation(0, robot, dataset, str(assets), DH_PARAMETERS.get(robot, []), str(output))
    if label is None or not (label["true"] or label["false"]):
        raise RuntimeError("Upstream simulator produced no reachability labels.")

    rgb_path = output / f"{robot}_rgb_0.jpg"
    depth_path = output / f"{robot}_depth_0.png"
    map_path = output / f"{robot}_sp_map_0.jpg"
    metadata = json.loads((output / f"{robot}_data_0.json").read_text(encoding="utf-8"))
    entry = {
        "raw_path": str(depth_path),
        "rgb_path": str(rgb_path),
        "sp_map_save_path": str(map_path),
        "rgb_map_save_path": str(output / f"{robot}_rgb_map_0.jpg"),
        "pcb_file": str(SIMULATOR / "arm_ws_pcb" / f"{robot}_ws.pcd"),
        "rotation_matrix": metadata["camera_to_base_rotation"],
        "translation_vector": metadata["camera_to_base_translation"],
        "camera_intrinsics": metadata["camera_intrinsics"],
        "workspace_radius": 0.05,
    }
    processor = DepthMapProcessor([entry])
    processor.initialize_kdtree()
    processor.process_image_single(entry)

    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    sp_map = cv2.imread(str(map_path))
    if rgb is None or depth is None or sp_map is None:
        raise RuntimeError("At least one required input could not be decoded.")
    if rgb.shape != sp_map.shape or rgb.shape[:2] != depth.shape or depth.max() == depth.min():
        raise RuntimeError("The RGB, depth and S-P Map must align and depth must be nonconstant.")

    questions = [
        {
            "image": rgb_path.name,
            "sp_map": map_path.name,
            "question": f"Is the {name} in the robot's reachable space?",
            "answer": answer,
        }
        for key, answer in (("true", "Yes"), ("false", "No"))
        for name in label[key]
    ]
    result = {
        "robot": robot,
        "rgb_shape": list(rgb.shape),
        "depth_range_mm": [int(depth.min()), int(depth.max())],
        "label_source": "upstream Euclidean distance to robot base, thresholded by maximum reach plus 0.05 m",
        "questions": questions,
    }
    (output / "pilot.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
