"""Evaluate point reachability with visible markers and held-out geometric certificates."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_target_alignment import load_rows
from fit_regional_readout import fit, load_split, summarize_region
from grounding_readout import THRESHOLD, cluster_delta, read_predictions, rectangle, region
from point_markers import find_markers, locate_marker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_image(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input hash changed: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def candidate_index(path):
    source = read(path)
    entries = {(r["image_sha256"], r["phrase"]): r["boxes"] for r in source}
    if len(entries) != len(source):
        raise ValueError(f"Duplicate candidate groups in {path}")
    return entries


def point_chroma(chroma, point, radius):
    if point is None:
        return None
    x, y = point
    h, w = chroma.shape
    return float(np.median(chroma[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)]))


def prepare(args):
    source = args.manifest.resolve().parent
    candidates = {name: candidate_index(Path(path)) for name, path in (value.split("=",1) for value in args.boxes)}
    args.output.mkdir(parents=True, exist_ok=False)
    global_rows, crop_rows, features = [], defaultdict(list), []
    for row in read(args.manifest)["rows"]:
        rgb = checked_image(source/row["marked_image"], row["marked_sha256"])
        spatial_map = checked_image(source/row["sp_map"], row["map_sha256"])
        if rgb.shape != spatial_map.shape:
            raise ValueError("RGB and map must share original pixel coordinates")
        truth_point = tuple(row["point"])
        visible = find_markers(rgb)
        if truth_point not in visible or len(visible) != 2:
            raise ValueError(f"Expected both object markers in {row['scene']}")
        chroma = np.ptp(spatial_map, axis=-1)
        query = dict(row, question=f"Is the magenta cross on the {row['target']} in the robot's reachable space?")
        global_rows.append(dict(query, image=str(source/row["marked_image"]), image_sha256=row["marked_sha256"],
                                sp_map=str(source/row["sp_map"])))
        record = dict(id=row["id"], scene=row["scene"], layout=row["layout"], target=row["target"], side=row["side"],
                      stratum=row["stratum"], truth=row["truth"], point=row["point"],
                      sampled_workspace_label=row["sampled_workspace_label"],
                      workspace_distance_m=row["workspace_distance_m"],
                      exact_pixel=point_chroma(chroma, truth_point, 0),
                      exact_neighborhood=point_chroma(chroma, truth_point, 2), boxes={})
        boxes = {"oracle": row["box"]}
        for name, index in candidates.items():
            choices = index[row["image_sha256"], row["target"]]
            boxes[name] = choices[0]["box"] if choices else None
        for name, candidate in boxes.items():
            box = np.asarray(candidate, dtype=float) if candidate is not None else None
            center = rectangle(box, chroma.shape, .5) if box is not None else None
            marker = locate_marker(visible, candidate)
            record["boxes"][name] = dict(marker_matches=marker == truth_point,
                                          center=float(np.median(region(chroma, center))) if center else None,
                                          marker_chroma=point_chroma(chroma, marker, 0),
                                          marker_neighborhood=point_chroma(chroma, marker, 2),
                                          features=summarize_region(chroma, box) if center else None)
            if center is None or name == "detector":
                continue
            crop = rectangle(box, chroma.shape, 2)
            folder = args.output/name
            folder.mkdir(exist_ok=True)
            tag = f"{row['id']}"
            image_path, map_path = folder/f"{tag}-rgb.png", folder/f"{tag}-map.png"
            if not (cv2.imwrite(str(image_path), region(rgb, crop)) and
                    cv2.imwrite(str(map_path), region(spatial_map, crop))):
                raise OSError(tag)
            crop_rows[name].append(dict(query, image=image_path.name, image_sha256=digest(image_path),
                                        sp_map=map_path.name, map_sha256=digest(map_path)))
        features.append(record)
    write(args.output/"global.json", dict(rows=global_rows))
    for name, rows in crop_rows.items():
        write(args.output/name/"manifest.json", dict(rows=rows))
    write(args.output/"regions.json", features)


def classify(value):
    return "U" if value is None else "Y" if value >= THRESHOLD else "N"


def evaluate(rows, values, frozen):
    known = [r for r in rows if r["truth"] != "U"]
    layouts, targets, workspaces = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        layouts[row["scene"]].append(row)
        targets[row["scene"], row["side"]].append(row)
        workspaces[row["scene"], row["target"]].append(row)

    def joint(groups, size, opposite=False):
        selected = [group for group in groups.values()
                    if len(group) == size and all(r["truth"] != "U" for r in group)
                    and (not opposite or {r["truth"] for r in group} == {"Y", "N"})]
        return dict(total=len(groups), certified=len(selected),
                    correct=sum(all(values[r["id"]] == r["truth"] for r in group) for group in selected))

    return dict(certified=len(known), correct=sum(values[r["id"]] == r["truth"] for r in known),
                unanswered=sum(values[r["id"]] == "U" for r in known),
                confusion=dict(Counter(f"{r['truth']}-{values[r['id']]}" for r in rows)),
                target_flip=joint(targets, 2, opposite=True), workspace_flip=joint(workspaces, 2, opposite=True),
                four=joint(layouts, 4), versus_frozen=cluster_delta(known, values, frozen))


def fixed_readouts(training, development):
    train = [r for r in load_split(training) if r["features"] is not None]
    dev = [r for r in load_split(development) if r["features"] is not None]
    values = torch.tensor([r["features"] for r in train], dtype=torch.float32)
    center, scale = values.mean(0), values.std(0, unbiased=False).clamp_min(.01)
    models, selected = {}, {}
    for depth, name in ((0, "logistic"), (1, "mlp")):
        trials = [(fit(train, dev, depth, seed, center, scale), seed) for seed in (17, 29)]
        (model, validation), seed = max(trials, key=lambda result: result[0][1]["correct"])
        models[name], selected[name] = model, dict(seed=seed, **validation)
    return models, center, scale, selected


def score(args):
    torch.set_num_threads(2)
    rows = read(args.regions)
    keys = {r["id"] for r in rows}
    frozen = read_predictions(args.frozen, "explicit")
    if frozen.keys() != keys:
        raise ValueError("Global model lacks evaluation rows")
    models, center, scale, selection = fixed_readouts(args.train, args.dev)
    all_methods = {name: {r["id"]: classify(r[key]) for r in rows}
                   for name, key in (("exact_pixel", "exact_pixel"), ("exact_neighborhood", "exact_neighborhood"))}
    all_methods.update(frozen=frozen, pcd={r["id"]: r["sampled_workspace_label"] for r in rows})
    identity = {}
    for method in rows[0]["boxes"]:
        boxes = {r["id"]: r["boxes"][method] for r in rows}
        hits = {(r["scene"], r["target"]): boxes[r["id"]]["marker_matches"] for r in rows}
        identity[method] = dict(n=len(hits), exact_marker=sum(hits.values()))
        for feature in ("center", "marker_chroma", "marker_neighborhood"):
            all_methods[f"{method}-{feature}"] = {key: classify(box[feature]) for key, box in boxes.items()}
        available = [(key, box["features"]) for key, box in boxes.items() if box["features"] is not None]
        for name, model in models.items():
            tensor = torch.tensor([feat for _, feat in available], dtype=torch.float32).reshape(-1, center.numel())
            inputs = (tensor-center)/scale
            with torch.no_grad():
                values = model(inputs).flatten().tolist()
            choices = {key: "Y" if score >= 0 else "N" for (key, _), score in zip(available, values, strict=True)}
            all_methods[f"{method}-{name}"] = {key: choices.get(key, "U") for key in keys}
    first_mention = {}
    for source in args.projector:
        name, path = source.split("=", 1)
        box_method = name.split("-")[0]
        raw = load_rows(Path(path))
        local = {r["id"]: r["explicit"]["free"] for r in raw}
        if local.keys() != {r["id"] for r in rows if r["boxes"][box_method]["features"] is not None}:
            raise ValueError(f"Unexpected local projector coverage: {name}")
        all_methods[name] = {key: local.get(key, "U") for key in keys}
        first_mention[name] = {
            r["id"]: match[1][0].upper() for r in raw
            if (match := re.search(r"\b(yes|no)\b", r["explicit"]["response"], re.IGNORECASE))
        }
    oracle = ("exact_pixel", "exact_neighborhood", "pcd", *(m for m in all_methods if m.startswith("oracle-")))
    subsets = {"all": rows, **{key: [r for r in rows if r["stratum"] == key] for key in ("interior", "boundary")}}
    metrics = {name: {key: evaluate(subset, values, frozen) for key, subset in subsets.items()}
               for name, values in all_methods.items()}
    known = [r for r in rows if r["truth"] != "U"]
    contrasts = {}
    for prefix in ("ranked17", "ranked29"):
        comparisons = {name: cluster_delta(known, all_methods[f"{prefix}-{name}"],
                                           all_methods[f"{prefix}-center"])
                       for name in ("logistic", "mlp", "marker_chroma", "marker_neighborhood", "projector")
                       if f"{prefix}-{name}" in all_methods}
        contrasts[prefix] = comparisons
    marker_errors = {name: dict(certified_with_correct_marker=sum(r["boxes"][name]["marker_matches"] for r in known),
                                missed_or_wrong_marker=sum(not r["boxes"][name]["marker_matches"] for r in known),
                                read_errors_with_correct_marker=sum(
                                    r["boxes"][name]["marker_matches"] and
                                    all_methods[f"{name}-marker_chroma"][r["id"]] != r["truth"] for r in known))
                     for name in identity}
    write(args.output, dict(n=len(rows), physical_layouts=len({r["layout"] for r in rows}),
                            labels=dict(Counter(r["truth"] for r in rows)), identity=identity,
                            readout_selection=selection, oracle_methods=oracle, metrics=metrics,
                            marker_errors=marker_errors,
                            first_mention_sensitivity={
                                name: sum(values.get(r["id"]) == r["truth"] for r in known)
                                for name, values in first_mention.items()
                            },
                            versus_same_box_center=contrasts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("manifest", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--boxes", nargs="+", required=True)
    ev = commands.add_parser("score")
    for name in ("regions", "train", "dev", "frozen", "output"):
        ev.add_argument(name, type=Path)
    ev.add_argument("--projector", nargs="+", default=[])
    args = parser.parse_args()
    (prepare if args.command == "prepare" else score)(args)


if __name__ == "__main__":
    main()
