"""Render crossed same-color targets with certified easy and near-boundary layouts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from audit_fixtures import audit
from run_paired_pilot import SIMULATOR, generate_pair


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, count, seed):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows = []
    colors = [("blue", [0.12, 0.28, 0.85, 1]), ("red", [0.84, 0.16, 0.15, 1]),
              ("green", [0.12, 0.67, 0.21, 1])]
    for layout in range(count):
        boundary = layout % 2 == 1
        center = np.array([rng.uniform(.18, .35), rng.uniform(-.08, .08)])
        theta = rng.uniform(-.53, .53)
        axis = np.array([np.cos(theta), np.sin(theta)])
        separation = rng.uniform(.35, .41)
        color, rgba = colors[layout % len(colors)]
        shapes = ("box", "sphere") if layout % 3 else ("cylinder", "sphere")
        names = tuple(f"{color} {shape}" for shape in shapes)
        yaw = int(rng.choice([-32, -17, 11, 26]))
        certified = None
        for radius in ((.82, .78, .74, .70, .66, .62) if boundary else (.47,)):
            provisional = []
            for ordering in (0, 1):
                scene = f"s{seed}-layout{layout:03}-order{ordering}"
                folder = output / f"{scene}-{int(radius*100)}"
                folder.mkdir()
                sign = -1 if ordering == 0 else 1
                objects = {
                    name: ((*(center + offset * separation * axis), .245), rgba)
                    for name, offset in zip(names, (sign, -sign), strict=True)
                }
                bases = [[*(center + offset * (separation + radius) * axis), .20] for offset in (-1, 1)]
                if layout % 4 >= 2:
                    bases.reverse()
                expected = dict(zip(names, ("YN", "NY") if (ordering == (layout % 4 >= 2))
                                    else ("NY", "YN"), strict=True))
                actual = generate_pair(folder, objects, scene_id=scene, bases=bases, camera_yaw=yaw,
                                       shapes=dict(zip(names, shapes, strict=True)))
                if actual != expected:
                    break
                fixture_path = folder / "paired_pilot.json"
                fixture = json.loads(fixture_path.read_text())
                for side, conf in fixture["configurations"].items():
                    for index, target in enumerate(names):
                        obj = fixture["targets"][target]
                        provisional.append(dict(
                            id=f"{scene}-{side}-{index}", scene=scene, layout=layout, ordering=ordering,
                            camera_yaw=yaw, target_index=index, stratum="boundary" if boundary else "interior",
                            base_offset_m=radius, side=side, target=target,
                            truth=conf["targets"][target]["label"],
                            workspace_distance_m=conf["rules"][target]["workspace_distance_m"],
                            question=f"Is the {target} in the robot's reachable space?",
                            image=f"{folder.name}/rgb.png", image_sha256=fixture["rgb_sha256"],
                            sp_map=f"{folder.name}/{conf['sp_map']}", map_sha256=sha(folder / conf["sp_map"]),
                            box=obj["box"], mask=f"{folder.name}/{obj['mask']}", point=obj["pixel_xy"],
                            certificate=f"{folder.name}/paired_pilot.json", certificate_sha256=sha(fixture_path),
                        ))
            if len(provisional) == 8:
                certified = provisional
                break
        if certified is None:
            raise RuntimeError(f"Layout {layout} did not pass both orderings")
        rows.extend(certified)
        print(json.dumps(dict(layout=layout, n=count, radius=radius)), flush=True)
    replays = []
    for target_index in (0, 1):
        index = {(r["scene"], r["side"]): r for r in rows if r["target_index"] == target_index}
        selection = [dict(scene=scene, target=r["target"], fixture=r["certificate"],
                          label=r["truth"] + index[(scene, "B")]["truth"])
                     for (scene, side), r in index.items() if side == "A"]
        manifest = output / f"certificate-selection-{target_index}.json"
        manifest.write_text(json.dumps(dict(selection=selection), indent=2) + "\n", encoding="utf-8")
        replays.append(audit(manifest))
    (output / "certificate-audit.json").write_text(json.dumps(replays, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(dict(seed=seed, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(SIMULATOR))
    build(args.output.resolve(), args.count, args.seed)
