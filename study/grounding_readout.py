"""Build and score location interventions without changing labels or source prompts."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from evaluate_target_alignment import load_rows
from prepare_official_box_probe import PREFIX, SUFFIX, normalized

THRESHOLD = 83
METHODS = ("oracle", "predicted", "wrong", "shift10", "shift20", "shift30", "scale15")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(1, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - intersection))


def rectangle(box, shape, scale=1):
    h, w = shape[:2]
    center = (box[:2]+box[2:])/2
    delta = (box[2:]-box[:2])*scale/2
    ends = np.clip([*np.floor(center-delta), *np.ceil(center+delta)], 0, [w, h, w, h]).astype(int)
    return ends.tolist() if np.all(ends[2:] > ends[:2]) else None


def region(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def official_boxes(source, row, phrase, shape):
    stem = row["image"].replace("_rgb_", "_data_").removesuffix(".jpg")
    meta = json.loads((source / f"{stem}.json").read_text(encoding="utf-8"))
    boxes = meta["bounding_boxes"]
    targets = [name for name in boxes if normalized(name) == phrase]
    scale = np.tile(shape[1::-1], 2)
    oracle = np.array(boxes[targets[0]])*scale if len(targets) == 1 else None
    other = [np.array(box)*scale for name, box in sorted(boxes.items()) if normalized(name) != phrase]
    return oracle, other[0] if other and oracle is not None else None


def chroma(values):
    return float(np.median(values)) if values.size else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    source = args.manifest.parent
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    predictions = json.loads(args.boxes.read_text(encoding="utf-8"))
    predicted = {(r["image_sha256"], r["phrase"]): r["boxes"] for r in predictions}
    args.output.mkdir(parents=True, exist_ok=False)
    crops = defaultdict(list)
    meta = []
    for row in rows:
        phrase = row["question"][len(PREFIX):-len(SUFFIX)]
        p = predicted[row["image_sha256"], phrase]
        rgb_path, map_path = source / row["image"], source / row["sp_map"]
        if digest(rgb_path) != row["image_sha256"] or digest(map_path) != row["map_sha256"]:
            raise ValueError(f"Source hash mismatch for {row['id']}")
        rgb, sp_map = cv2.imread(str(rgb_path)), cv2.imread(str(map_path))
        if rgb is None or sp_map is None or rgb.shape != sp_map.shape:
            raise ValueError(f"Unaligned RGB and map for {row['id']}")
        mask = point = None
        if "box" in row:
            oracle = np.array(row["box"], dtype=float)
            other = [r for r in rows if r["scene"] == row["scene"] and r["side"] == row["side"]
                     and r["target"] != row["target"]]
            wrong = np.array(other[0]["box"], dtype=float)
            mask = cv2.imread(str(source / row["mask"]), cv2.IMREAD_GRAYSCALE) > 0
            point = row.get("point")
            if point is None:
                fixture = json.loads((source / row["certificate"]).read_text(encoding="utf-8"))
                point = fixture["targets"][row["target"]]["pixel_xy"]
            if not mask[point[1], point[0]]:
                raise ValueError(f"Certificate point lies outside the visible target for {row['id']}")
        else:
            oracle, wrong = official_boxes(source, row, phrase, rgb.shape)
        predicted_box = np.array(p[0]["box"]) if p else None
        boxes = dict(oracle=oracle, wrong=wrong, predicted=predicted_box)
        if predicted_box is not None:
            px_width = predicted_box[2]-predicted_box[0]
            for factor in (10, 20, 30):
                # Reverse the deterministic horizontal direction across layouts.
                parity = row["layout"] if "layout" in row else int(row["image_sha256"][-1], 16)
                sign = -1 if parity % 2 else 1
                boxes[f"shift{factor}"] = predicted_box + sign*px_width*factor/100*np.array([1, 0, 1, 0])
            scaled = rectangle(predicted_box, rgb.shape, 1.5)
            boxes["scale15"] = np.array(scaled) if scaled else None
        arr = np.ptp(sp_map.astype(np.float32), axis=-1)
        log = dict(id=row["id"], scene=row.get("scene", row["image_sha256"]),
                   layout=row.get("layout", row["image_sha256"]), truth=row["truth"],
                   target=phrase, side=row.get("side"),
                   stratum=row.get("stratum", "source"), score=p[0]["score"] if p else None,
                   iou=iou(oracle, predicted_box) if oracle is not None and predicted_box is not None else None,
                   mask_chroma=chroma(arr[mask]) if mask is not None else None,
                   point_chroma=float(arr[point[1], point[0]]) if point is not None else None,
                   mask_reachable_fraction=float((arr[mask] >= THRESHOLD).mean()) if mask is not None else None,
                   regions={})
        for method, box in boxes.items():
            if box is None:
                continue
            bbox, center = rectangle(box, rgb.shape), rectangle(box, rgb.shape, .5)
            if bbox is None or center is None:
                continue
            log["regions"][method] = dict(full=chroma(region(arr, bbox)), center=chroma(region(arr, center)),
                                          foreground=float(region(mask, bbox).mean()) if mask is not None else None)
            if method not in args.crops:
                continue
            crop = rectangle(box, rgb.shape, 2)
            folder = args.output / method
            folder.mkdir(exist_ok=True)
            image = folder / f"{row['id']}-rgb.png"
            map_image = folder / f"{row['id']}-map.png"
            if not cv2.imwrite(str(image), region(rgb, crop)) or not cv2.imwrite(str(map_image), region(sp_map, crop)):
                raise OSError(f"Could not write crop {row['id']}/{method}")
            crops[method].append(dict(row, image=image.name, sp_map=map_image.name,
                                      image_sha256=digest(image), map_sha256=digest(map_image)))
        meta.append(log)
    for method, subset in crops.items():
        save(args.output / method / "manifest.json", dict(rows=subset))
    save(args.output / "regions.json", meta)


def correct(value, truth):
    return value == truth


def cluster_delta(records, a, b):
    groups = defaultdict(list)
    for row in records:
        groups[row["layout"]].append(int(correct(a[row["id"]], row["truth"])) -
                                     int(correct(b[row["id"]], row["truth"])))
    rng = random.Random(73595)
    clusters = list(groups.values())
    samples = sorted(mean(value for group in rng.choices(clusters, k=len(clusters)) for value in group)
                     for _ in range(2000))
    return dict(delta=mean(x for group in clusters for x in group), interval=[samples[50], samples[1949]])


def read_predictions(path, key):
    return {r["id"]: r[key]["free"] for r in load_rows(path)}


def score(args):
    rows = json.loads(args.regions.read_text(encoding="utf-8"))
    key = "original" if args.official else "explicit"
    frozen = read_predictions(args.frozen, key)
    if frozen.keys() != {r["id"] for r in rows}:
        raise ValueError("Frozen baseline coverage differs from evaluation rows")
    answers = {"frozen": frozen}
    for method in METHODS:
        for field in ("full", "center"):
            result = {}
            for row in rows:
                identifier = row["id"]
                entry = row["regions"].get(method)
                allowed = entry is not None and (method in {"oracle", "wrong"} or row["score"] >= args.confidence)
                result[identifier] = ("Y" if entry[field] >= THRESHOLD else "N") if allowed else frozen[identifier]
            answers[f"{method}-{field}"] = result
    if any(row["mask_chroma"] is not None for row in rows):
        answers["oracle-mask"] = {r["id"]: "Y" if r["mask_chroma"] >= THRESHOLD else "N" for r in rows}
    if any(row.get("point_chroma") is not None for row in rows):
        answers["certificate-point"] = {r["id"]: "Y" if r["point_chroma"] >= THRESHOLD else "N" for r in rows}
    for option in args.prediction:
        name, path = option.split("=", 1)
        readout = read_predictions(Path(path), key)
        method = name.split("-")[0]
        expected = {r["id"] for r in rows if method in r["regions"]}
        if readout.keys() != expected:
            raise ValueError(f"Unexpected {name} crop coverage")
        answers[name] = {r["id"]: readout.get(r["id"], frozen[r["id"]]) if
                         method in {"oracle", "wrong"} or (r["score"] or 0) >= args.confidence
                         else frozen[r["id"]] for r in rows}
    metrics = {}
    for name, result in answers.items():
        metrics[name] = {}
        for stratum in (None, *sorted({r["stratum"] for r in rows})):
            select = [r for r in rows if stratum is None or r["stratum"] == stratum]
            groups = defaultdict(list)
            scores = defaultdict(list)
            target_pairs, workspace_pairs = defaultdict(list), defaultdict(list)
            for row in select:
                hit = correct(result[row["id"]], row["truth"])
                groups[row["scene"]].append(hit)
                scores[row["truth"]].append(hit)
                if not args.official:
                    target_pairs[row["scene"], row["side"]].append(hit)
                    workspace_pairs[row["scene"], row["target"]].append(hit)
            complete = [all(v) for v in groups.values() if len(v) == 4] if not args.official else []
            four_view_interval = None
            if complete:
                view_rows = {}
                frozen_groups = defaultdict(list)
                for row in select:
                    scene = row["scene"]
                    view_rows[scene] = dict(id=scene, layout=row["layout"], truth=True)
                    frozen_groups[scene].append(correct(frozen[row["id"]], row["truth"]))
                four_view_interval = cluster_delta(
                    list(view_rows.values()), {k: all(v) for k, v in groups.items()},
                    {k: all(v) for k, v in frozen_groups.items()},
                )
            metrics[name][stratum or "all"] = dict(
                n=len(select), correct=sum(map(sum, scores.values())),
                by_label={k: dict(correct=sum(v), n=len(v)) for k, v in scores.items()},
                complete=sum(complete), four_question_views=len(complete),
                target_pairs=dict(correct=sum(all(v) for v in target_pairs.values()), n=len(target_pairs)),
                workspace_pairs=dict(correct=sum(all(v) for v in workspace_pairs.values()), n=len(workspace_pairs)),
                versus_frozen=cluster_delta(select, result, frozen),
                four_question_versus_frozen=four_view_interval,
            )
    diagnosed = []
    for row in rows:
        key = row["id"]
        if correct(answers["predicted-full"][key], row["truth"]):
            continue
        point = answers.get("certificate-point", {}).get(key)
        mask = answers.get("oracle-mask", {}).get(key)
        diagnosed.append(dict(id=key, truth=row["truth"], iou=row["iou"], point_correct=point == row["truth"]
                              if point is not None else None,
                              mask_correct=mask == row["truth"] if mask is not None else None,
                              oracle_box_correct=answers["oracle-full"][key] == row["truth"],
                              predicted_center_correct=answers["predicted-center"][key] == row["truth"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, dict(confidence=args.confidence, threshold=THRESHOLD, metrics=metrics,
                           detector=dict(available=sum("predicted" in r["regions"] for r in rows),
                                         iou_at_least_half=sum((r["iou"] or 0) >= .5 for r in rows),
                                         iou_eligible=sum(r["iou"] is not None for r in rows)),
                           predictions=answers, predicted_rule_errors=diagnosed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("manifest", type=Path)
    prepare_cmd.add_argument("boxes", type=Path)
    prepare_cmd.add_argument("output", type=Path)
    prepare_cmd.add_argument("--crops", nargs="+", choices=METHODS, default=METHODS)
    report = commands.add_parser("score")
    report.add_argument("regions", type=Path)
    report.add_argument("frozen", type=Path)
    report.add_argument("output", type=Path)
    report.add_argument("--prediction", nargs="*", default=[])
    report.add_argument("--official", action="store_true")
    report.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else score)(args)


if __name__ == "__main__":
    main()
