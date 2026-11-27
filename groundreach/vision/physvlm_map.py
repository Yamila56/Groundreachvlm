"""Generate S-P Maps with the workspace cloud and renderer from PhysVLM."""

import sys


def render_sp_map(output, key, eye, rotation, intrinsics, base, simulator):
    if str(simulator) not in sys.path:
        sys.path.insert(0, str(simulator))
    import generate_sp_map
    from generate_sp_map import DepthMapProcessor

    entry = {
        "raw_path": str(output / "depth.png"), "rgb_path": str(output / "rgb.png"),
        "sp_map_save_path": str(output / f"sp_map_{key}.png"),
        "rgb_map_save_path": str(output / f"overlay_{key}.png"),
        "pcb_file": str(simulator / "arm_ws_pcb" / "PANDA_ws.pcd"),
        "rotation_matrix": rotation.tolist(), "translation_vector": (eye - base).tolist(),
        "camera_intrinsics": intrinsics, "workspace_radius": 0.05,
    }
    processor = DepthMapProcessor([entry])
    if generate_sp_map.global_kdtree is None:
        processor.initialize_kdtree()
    processor.process_image_single(entry)
    return generate_sp_map.global_kdtree
