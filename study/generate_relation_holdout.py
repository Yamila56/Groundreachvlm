"""Render paired reachability queries with visually identical targets and a distinct anchor."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from audit_fixtures import audit
from generate_grounding_study import sha
from point_markers import marked_image
from run_paired_pilot import SIMULATOR, generate_pair


def save_split(output, seed, rows, detector, references):
    audits = []
    for i, selection in enumerate(references):
        path = output / f"certificate-selection-{i}.json"
        path.write_text(json.dumps(dict(selection=selection), indent=2) + "\n", encoding="utf-8")
        audits.append(audit(path))
    for filename, value in (("manifest.json", dict(seed=seed, rows=rows)), ("detector-inputs.json", dict(rows=detector)),
                            ("certificate-audit.json", audits)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def detector_row(phrase, image):
    return dict(**image, question=f"Is the {phrase} in the robot's reachable space?")


def collect(output, scene, layout, yaw, names, queries, fixture):
    directory = output / scene
    fixture_path = directory / "paired_pilot.json"
    targets = fixture["targets"]
    marked = directory / "marked-rgb.png"
    if not cv2.imwrite(str(marked), marked_image(cv2.imread(str(directory / "rgb.png")),
                                                 [targets[name]["pixel_xy"] for name in names])):
        raise OSError(marked)
    marked_sha, certificate_sha = sha(marked), sha(fixture_path)
    image = dict(image=f"{scene}/rgb.png", image_sha256=fixture["rgb_sha256"])
    rows, detectors, references = [], {}, []
    for idx, (name, noun, anchor, relation, axis, direction, nearest) in enumerate(queries):
        target = targets[name]
        anchor_box = targets[anchor]["box"]
        anchor_center = (anchor_box[axis] + anchor_box[axis + 2]) / 2
        ordered = []
        for instance in names:
            box = targets[instance]["box"]
            distance = direction * ((box[axis] + box[axis + 2]) / 2 - anchor_center)
            if abs(distance) <= max(box[axis + 2] - box[axis], 13):
                raise ValueError(f"Ambiguous visible relation for {scene}/{instance}")
            if distance > 0:
                ordered.append((distance, instance))
        if not ordered or min(ordered)[1] != name or (not nearest and len(ordered) != 1):
            raise ValueError(f"Incorrect referent for {scene}/{name}")
        phrase = f"{'nearest ' if nearest else ''}{noun} {relation} the {anchor}"
        for query in (noun, anchor, phrase):
            detectors[query] = detector_row(query, image)
        references.append(dict(scene=scene, target=name, fixture=f"{scene}/paired_pilot.json",
                               label=fixture["pairs"][name]))
        for side, conf in fixture["configurations"].items():
            rows.append(dict(
                **detectors[phrase], id=f"{scene}-{side}-{idx}", scene=scene, layout=layout, side=side, target=phrase,
                noun=noun, anchor=anchor, relation=relation, axis=axis, direction=direction, nearest=nearest,
                camera_yaw=yaw, box=target["box"], anchor_box=anchor_box, point=target["pixel_xy"],
                marker_count=len(names), truth=conf["targets"][name]["label"],
                sp_map=f"{scene}/{conf['sp_map']}", map_sha256=sha(directory / conf["sp_map"]),
                marked_image=f"{scene}/{marked.name}", marked_sha256=marked_sha,
                certificate=f"{scene}/paired_pilot.json", certificate_sha256=certificate_sha,
            ))
    return rows, list(detectors.values()), references


def build(output, count, seed):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    assets = (
        ("blue cylinders", "cylinder", [.12, .28, .85, 1], "yellow box", "box", [.9, .73, .09, 1]),
        ("red boxes", "box", [.84, .16, .15, 1], "cyan sphere", "sphere", [.11, .71, .8, 1]),
        ("green spheres", "sphere", [.12, .67, .21, 1], "purple cylinder", "cylinder", [.57, .18, .8, 1]),
        ("purple boxes", "box", [.57, .18, .8, 1], "yellow sphere", "sphere", [.9, .73, .09, 1]),
    )
    rows, detector = [], []
    references = [[], []]
    sides = (("to the left of", "to the right of"), ("above", "below"))
    for layout in range(count):
        axis = layout % 2
        yaw = int(rng.choice([-44, -24, 5, 31, 47]))
        angle = np.deg2rad(yaw)
        horizontal = np.array([np.cos(angle), np.sin(angle)])
        vertical = np.array([np.sin(angle), -np.cos(angle)])
        basis = np.vstack((horizontal, vertical))
        center = rng.uniform([.13, -.05], [.39, .05])
        separation = rng.uniform(.29, .37, 2)
        # Vary the absolute position and the perpendicular offset of the referent.
        anchor_xy = center + basis.T @ rng.uniform(-.095, .095, 2)
        pos = [anchor_xy + sign * separation[i] * basis[axis] + rng.uniform(-.06, .06) * basis[1-axis]
               for i, sign in enumerate((-1, 1))]
        plural, shape, color, anchor, anchor_shape, anchor_color = assets[(layout // 2) % len(assets)]
        noun = plural.removesuffix("es") if shape == "box" else plural.removesuffix("s")
        scene = f"rel-{seed}-{layout:03}"
        directory = output / scene
        directory.mkdir()
        names = ("target-0", "target-1")
        objects = {name: ((float(x), float(y), .245), color) for name, (x, y) in zip(names, pos, strict=True)}
        objects[anchor] = ((*anchor_xy, .245), anchor_color)
        shapes = {name: shape for name in names} | {anchor: anchor_shape}
        radii = (.47, .54) if (layout // 2) % 2 == 0 else (.73, .80)
        bases = [[*(point + sign * radius * basis[axis]), .20]
                 for point, sign, radius in zip(pos, (-1, 1), radii, strict=True)]
        if rng.integers(2):
            bases.reverse()
        generate_pair(directory, objects_layout=objects, scene_id=scene, bases=bases,
                      camera_yaw=yaw, shapes=shapes, point_quantile=float(rng.choice([.32, .5, .68])))
        fixture_path = directory / "paired_pilot.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        targets = fixture["targets"]
        anchor_box = targets[anchor]["box"]
        anchor_center = (anchor_box[axis] + anchor_box[axis + 2]) / 2
        marked = directory / "marked-rgb.png"
        if not cv2.imwrite(str(marked), marked_image(cv2.imread(str(directory / "rgb.png")),
                                                     [targets[name]["pixel_xy"] for name in names])):
            raise OSError(marked)
        marked_sha = sha(marked)
        certificate_sha = sha(fixture_path)
        image = dict(image=f"{scene}/rgb.png", image_sha256=fixture["rgb_sha256"])
        for phrase in (noun, anchor):
            detector.append(dict(**image, question=f"Is the {phrase} in the robot's reachable space?"))
        for idx, (name, relation) in enumerate(zip(names, sides[axis], strict=True)):
            target = targets[name]
            box = target["box"]
            displacement = (box[axis] + box[axis + 2]) / 2 - anchor_center
            if (displacement if idx else -displacement) <= max(box[axis + 2] - box[axis], 13):
                raise ValueError(f"Ambiguous visible relation for {scene}/{name}")
            phrase = f"{noun} {relation} the {anchor}"
            query = dict(**image, question=f"Is the {phrase} in the robot's reachable space?")
            detector.append(query)
            references[idx].append(dict(scene=scene, target=name, fixture=f"{scene}/paired_pilot.json",
                                        label=fixture["pairs"][name]))
            for side, conf in fixture["configurations"].items():
                rows.append(dict(
                    **query, id=f"{scene}-{side}-{idx}", scene=scene, layout=layout, side=side, target=phrase,
                    noun=noun, anchor=anchor, relation=relation, axis=axis, direction=2 * idx - 1,
                    camera_yaw=yaw, box=box, anchor_box=anchor_box, point=target["pixel_xy"],
                    truth=conf["targets"][name]["label"],
                    sp_map=f"{scene}/{conf['sp_map']}", map_sha256=sha(directory / conf["sp_map"]),
                    marked_image=f"{scene}/{marked.name}", marked_sha256=marked_sha,
                    certificate=f"{scene}/paired_pilot.json", certificate_sha256=certificate_sha,
                ))
        print(f"Rendered {layout + 1}/{count} layouts", flush=True)
    save_split(output, seed, rows, detector, references)


def build_anchor_switch(output, count, seed, confirmation, physical_probe=False, balanced_points=False,
                        resume=False, vary_pattern_direction=False):
    if resume and (output / "manifest.json").exists():
        raise ValueError("Resume requires an unfinished output directory")
    output.mkdir(parents=True, exist_ok=resume)
    rng = np.random.default_rng(seed)
    rows, detector = [], []
    references = [[], []]
    assets = (
        ("orange spheres", "sphere", [.9, .36, .09, 1], ("green box", "cyan cylinder")),
        ("cyan boxes", "box", [.11, .71, .8, 1], ("orange sphere", "purple cylinder")),
        ("yellow cylinders", "cylinder", [.9, .73, .09, 1], ("blue box", "red sphere")),
        ("red cylinders", "cylinder", [.84, .16, .15, 1], ("cyan sphere", "yellow box")),
    )
    if confirmation:
        assets = (
            ("blue boxes", "box", [.12, .28, .85, 1], ("orange box", "orange sphere")),
            ("green cylinders", "cylinder", [.12, .67, .21, 1], ("purple box", "purple sphere")),
            ("red spheres", "sphere", [.84, .16, .15, 1], ("cyan box", "green cylinder")),
            ("yellow spheres", "sphere", [.9, .73, .09, 1], ("yellow box", "blue sphere")),
        )
    if physical_probe:
        assets = (
            ("purple boxes", "box", [.57, .18, .8, 1], ("cyan box", "red cylinder")),
            ("blue cylinders", "cylinder", [.12, .28, .85, 1], ("red sphere", "purple box")),
            ("cyan spheres", "sphere", [.11, .71, .8, 1], ("yellow cylinder", "green sphere")),
            ("green boxes", "box", [.12, .67, .21, 1], ("blue sphere", "cyan cylinder")),
        )
    colors = dict(green=[.12, .67, .21, 1], cyan=[.11, .71, .8, 1], orange=[.9, .36, .09, 1],
                  purple=[.57, .18, .8, 1], blue=[.12, .28, .85, 1], red=[.84, .16, .15, 1],
                  yellow=[.9, .73, .09, 1])
    target_names = ("instance-0", "instance-1", "instance-2")
    terms = (("to the left of", "to the right of"), ("above", "below"))
    for layout in range(count):
        axis, direction = layout % 2, 2 * ((layout // 2) % 2) - 1
        yaw = int(rng.choice([-48, -28, -9, 14, 37, 52]))
        angle = np.deg2rad(yaw)
        basis = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), -np.cos(angle)]])
        center = rng.uniform([.12, -.055], [.40, .055])
        scene = f"anchor-{seed}-{layout:03}"
        directory = output / scene
        directory.mkdir(exist_ok=resume)
        plural, shape, color, names = assets[(layout // 4) % len(assets)]
        noun = plural.removesuffix("es") if shape == "box" else plural.removesuffix("s")
        anchors = names if rng.integers(2) else names[::-1]
        objects, shapes = {}, {}
        positions = rng.uniform([-.44, -.27, -.05, .17, .40], [-.40, -.17, .05, .27, .44])
        names_by_position = (target_names[0], anchors[0], target_names[1], anchors[1], target_names[2])
        for i, name in enumerate(names_by_position):
            pos = center + positions[i] * basis[axis] + rng.uniform(-.075, .075) * basis[1-axis]
            if name in target_names:
                objects[name], shapes[name] = ((*pos, .245), color), shape
            else:
                tint, geometry = name.split()
                objects[name], shapes[name] = ((*pos, .245), colors[tint]), geometry
        pattern = "legacy"
        if balanced_points:
            pattern_index = (layout + layout//4) % 4 if vary_pattern_direction else layout % 4
            pattern = ("cross", "both_reachable", "both_unreachable", "workspace_flip")[pattern_index]
            along, across = basis[axis], basis[1-axis]
            side = float(rng.choice([-1, 1]))
            offsets = {
                "cross": (-.74*along, .74*along),
                "both_reachable": (-.06*along, .08*along),
                "both_unreachable": (side*1.45*across, -side*1.45*across),
                "workspace_flip": (.02*along, side*1.45*across),
            }[pattern]
            bases = [[*(center+offset), .20] for offset in offsets]
        else:
            bases = [[*(center + offset * basis[axis]), .20] for offset in (-.64, .64)]
        if rng.integers(2):
            bases.reverse()
        fixture_path = directory / "paired_pilot.json"
        quantile = float(rng.choice([.38, .5, .62]))
        if not resume or not fixture_path.is_file():
            generate_pair(directory, objects_layout=objects, scene_id=scene, bases=bases, camera_yaw=yaw,
                          shapes=shapes, point_quantile=quantile)
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not np.array_equal(bases, [c["base_m"] for c in fixture["configurations"].values()]):
            raise ValueError(f"Existing workspace poses differ for {scene}")
        positive = int(direction > 0)
        queries = [(target_names[i + positive], noun, anchor, terms[axis][positive], axis, direction, True)
                   for i, anchor in enumerate(anchors)]
        if balanced_points:
            queries = [(target_names[2*i], noun, anchor, terms[axis][i], axis, 2*i-1, True)
                       for i, anchor in enumerate(anchors)]
        paired_rows, detector_rows, certificates = collect(output, scene, layout, yaw, target_names, queries, fixture)
        if pattern == "both_unreachable":
            for certificate in certificates:
                certificate["allow_identical_stable_maps"] = True
        chromatic = "same-anchor-color" if anchors[0].split()[0] == anchors[1].split()[0] else (
            "target-anchor-color-match" if noun.split()[0] in (anchor.split()[0] for anchor in anchors) else
            "distinct-colors")
        rows.extend(dict(row, anchor_palette=chromatic, **({"layout_pattern": pattern} if balanced_points else {}))
                    for row in paired_rows)
        detector.extend(detector_rows)
        for collection, certificate in zip(references, certificates, strict=True):
            collection.append(certificate)
        print(f"Rendered {layout + 1}/{count} anchor-switch layouts", flush=True)
    save_split(output, seed, rows, detector, references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor-switch", action="store_true")
    parser.add_argument("--balanced-points", action="store_true")
    parser.add_argument("--vary-pattern-direction", action="store_true")
    parser.add_argument("--resume-balanced-points", action="store_true")
    asset_set = parser.add_mutually_exclusive_group()
    asset_set.add_argument("--confirmation", action="store_true")
    asset_set.add_argument("--physical-probe", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SIMULATOR))
    if (args.confirmation or args.physical_probe) and not args.anchor_switch:
        parser.error("A custom anchor asset set requires --anchor-switch")
    if args.balanced_points and not args.anchor_switch:
        parser.error("--balanced-points requires --anchor-switch")
    if args.resume_balanced_points and not args.balanced_points:
        parser.error("--resume-balanced-points requires --balanced-points")
    if args.vary_pattern_direction and not args.balanced_points:
        parser.error("--vary-pattern-direction requires --balanced-points")
    if args.anchor_switch:
        build_anchor_switch(args.output.resolve(), args.count, args.seed, args.confirmation,
                            args.physical_probe, args.balanced_points, args.resume_balanced_points,
                            args.vary_pattern_direction)
    else:
        build(args.output.resolve(), args.count, args.seed)
