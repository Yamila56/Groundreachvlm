"""Construct crossed target and workspace pairs with a paired camera rotation."""
import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from experiments.geometry_audit.audit_fixtures import audit
from experiments.geometry_audit.audit_matched_pairs import locate
from experiments.physvlm_reproduction.generate_paired_dataset import COLORS
from groundreach.simulation.panda import generate_pair
from groundreach.simulation.upstream import SIMULATOR


SEED = 20261125
YAWS = (0, 27)
THRESHOLD = 83  # Locked on the earlier matched experiment's training split.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_scene(output, layout_id, colors, rng, *, yaws=YAWS):
    center = np.array([rng.uniform(.10, .45), rng.uniform(-.10, .10)])
    theta = rng.uniform(-.64, .64)
    axis = np.array([np.cos(theta), np.sin(theta)])
    offsets = rng.uniform(.34, .40, size=2)
    layout = {name: ((*tuple(center + sign * offset * axis), .245), COLORS[name])
              for name, sign, offset in zip(colors, (-1, 1), offsets, strict=True)}
    radii = offsets + rng.uniform(.42, .50, size=2)
    bases = [[*list(center + sign * radius * axis), .20]
             for sign, radius in zip((-1, 1), radii, strict=True)]
    swap = layout_id % 2
    if swap:
        bases.reverse()
    expected = dict(zip(colors, ("NY", "YN") if swap else ("YN", "NY"), strict=True))
    rows = []
    for yaw in yaws:
        scene = f"target-{layout_id:03d}-yaw{yaw}"
        directory = output / scene
        directory.mkdir()
        actual = generate_pair(directory, layout, scene_id=scene, bases=bases, camera_yaw=yaw)
        if actual != expected:
            raise RuntimeError(f"Cannot certify {scene}: {actual} versus {expected}")
        fixture = json.loads((directory / "paired_pilot.json").read_text())
        rgb = cv2.imread(str(directory / fixture["rgb"]))
        masks = {color: locate(rgb, color).astype(bool) for color in colors}
        for side, conf in fixture["configurations"].items():
            sp_map = directory / conf["sp_map"]
            chroma = np.ptp(cv2.imread(str(sp_map)).astype(np.float32), axis=-1)
            map_hash = sha(sp_map)
            for color in colors:
                target = fixture["targets"][color]
                mask = masks[color]
                x, y = target["pixel_xy"]
                target_chroma = float(np.median(chroma[mask])) if mask.any() else None
                gold = conf["targets"][color]["label"]
                certificate = conf["targets"][color]
                rows.append(dict(
                    id=f"{scene}-{side}-{color.split()[0]}", scene=scene, layout=layout_id,
                    camera_yaw=yaw, side=side, target=color, truth=gold,
                    question=f"Is the {color} in the robot's reachable space?",
                    image=f"{scene}/{fixture['rgb']}", image_sha256=fixture["rgb_sha256"],
                    sp_map=f"{scene}/{conf['sp_map']}", map_sha256=map_hash,
                    certificate=f"{scene}/paired_pilot.json", certificate_sha256=sha(directory / "paired_pilot.json"),
                    fk_error_m=certificate.get("fk_error_m"),
                    outer_bound_margin_m=certificate["distance_m"] - certificate["radius_m"] if gold == "N" else None,
                    center_localized=bool(mask[y, x]), target_chroma=target_chroma,
                    whole_map_median_chroma=float(np.median(chroma)),
                    geometry="U" if target_chroma is None else "Y" if target_chroma >= THRESHOLD else "N",
                    workspace=conf["rules"][color]["workspace_label"],
                ))
    return rows


def audit_manifest(root, earlier):
    source = root / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in manifest["rows"]:
        for field, digest in (("image", "image_sha256"), ("sp_map", "map_sha256"),
                              ("certificate", "certificate_sha256")):
            if sha(root / row[field]) != row[digest]:
                raise ValueError(f"Unexpected file hash: {row[field]}")
        grouped[row["scene"]].append(row)
    if earlier:
        prior_hashes = {r["rgb_sha256"] for split in ("train", "dev", "test")
                        for r in json.loads((earlier / split / "selection.json").read_text())["selection"]}
        current_hashes = {r["image_sha256"] for r in manifest["rows"]}
        if len(current_hashes) != len(grouped) or current_hashes & prior_hashes:
            raise ValueError("Repeated RGB between the new and earlier layouts")

    by_target = [[], []]
    for scene, group in grouped.items():
        targets = sorted({row["target"] for row in group})
        if len(group) != 4 or len(targets) != 2 or len({row["image_sha256"] for row in group}) != 1:
            raise ValueError(f"Malformed crossed fixture: {scene}")
        if len({row["map_sha256"] for row in group}) != 2:
            raise ValueError(f"Expected different workspaces: {scene}")
        for side in ("A", "B"):
            side_rows = [r for r in group if r["side"] == side]
            if {r["truth"] for r in side_rows} != {"Y", "N"} or len({r["map_sha256"] for r in side_rows}) != 1:
                raise ValueError(f"Invalid same-map target switch: {scene}/{side}")
        for index, target in enumerate(targets):
            pair = sorted((row for row in group if row["target"] == target), key=lambda row: row["side"])
            if {row["truth"] for row in pair} != {"Y", "N"}:
                raise ValueError(f"Invalid same-target map switch: {scene}/{target}")
            by_target[index].append(dict(scene=scene, target=target,
                fixture=str((root / pair[0]["certificate"]).resolve()),
                label="".join(row["truth"] for row in pair)))

    with TemporaryDirectory(prefix="physvlm-target-audit-") as temporary:
        selection = Path(temporary) / "selection.json"
        results = []
        for records in by_target:
            selection.write_text(json.dumps(dict(selection=records)), encoding="utf-8")
            results.append(audit(selection))
    return dict(layouts=manifest["layout_count"], views=len(grouped), queries=len(manifest["rows"]),
                checked_earlier_rgb=bool(earlier), manifest_sha256=sha(source),
                max_replay_error_m=max(r["max_fk_error_m"] for r in results),
                min_bound_margin_m=min(r["minimum_bound_margin_m"] for r in results),
                reachable_witnesses=sum(r["reachable_witnesses"] for r in results),
                unreachable_bounds=sum(r["unreachable_bounds"] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--yaws", type=int, nargs="+", default=YAWS)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--matched-root", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_manifest(args.output, args.matched_root), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SIMULATOR))
    if args.repeats < 1 or len(set(args.yaws)) != len(args.yaws):
        parser.error("Provide a positive repeat count and distinct yaws")
    seed = args.seed + int(args.pilot)
    rng = np.random.default_rng(seed)
    combinations = list(itertools.permutations(COLORS, 2))
    schedules = combinations[:2] if args.pilot else combinations * args.repeats
    rng.shuffle(schedules)
    rows = []
    for layout_id, colors in enumerate(schedules):
        rows.extend(make_scene(args.output, layout_id, colors, rng, yaws=args.yaws))
    if len({r["image_sha256"] for r in rows}) != len(schedules) * len(args.yaws):
        raise RuntimeError("Repeated RGB across distinct scenes")
    manifest = dict(seed=seed, yaws=args.yaws, geometry_threshold=THRESHOLD,
                    layout_count=len(schedules), rows=rows)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(layouts=len(schedules), queries=len(rows),
                          geometry_correct=sum(r["truth"] == r["geometry"] for r in rows),
                          workspace_correct=sum(r["truth"] == r["workspace"] for r in rows),
                          centers_localized=sum(r["center_localized"] for r in rows))))


if __name__ == "__main__":
    main()
