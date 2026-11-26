"""Compare frozen region ranking with a fixed image-coordinate relation rule."""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from detect_grounding_targets import target_phrase
from evaluate_marked_points import candidate_index, checked_image, classify, read, write
from grounding_readout import cluster_delta, iou
from point_markers import find_markers, locate_marker
from replay_marked_points import counts


COLOR_RGB = {
    "red": (225, 45, 40), "orange": (235, 112, 28), "yellow": (230, 190, 30),
    "green": (40, 175, 55), "cyan": (28, 187, 222), "blue": (38, 79, 220),
    "purple": (146, 45, 205),
}


def hue_vector(rgb):
    vector = np.asarray(rgb, dtype=np.float64) - np.mean(rgb)
    return vector / max(np.linalg.norm(vector), 1e-6)


def anchor_score(image, candidate, phrase, use_shape):
    tint, geometry = phrase.split()
    height, width = image.shape[:2]
    x0, y0, x1, y1 = np.clip(candidate["box"], 0, [width, height, width, height])
    bgr = image[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
    if bgr.size == 0:
        return -np.inf
    foreground = (np.ptp(bgr, axis=2) > 32) & (bgr.max(2) > 45)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    if n <= 1:
        return -np.inf
    component = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    mask = labels == component
    observed = np.median(bgr[mask, ::-1], axis=0)
    color = float(hue_vector(observed) @ hue_vector(COLOR_RGB[tint]))
    if not use_shape:
        return color
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    width, height = cv2.minAreaRect(contour)[1]
    fill = cv2.contourArea(contour) / max(width * height, 1)
    expected = .99 if geometry == "box" else .79
    # A strong color mismatch should dominate shape and pretrained model tie-breaks.
    return color - .6 * abs(fill - expected) + .05 * candidate["semantic"] + .02 * candidate["score"]


def axis_center(box, axis):
    return (box[axis] + box[axis + 2]) / 2


def relation_box(pool, anchor, axis, direction, nearest, markers=None):
    if anchor is None:
        return None
    reference = axis_center(anchor, axis)
    options = []
    for index, candidate in enumerate(pool):
        box = candidate["box"]
        marker = locate_marker(markers, box) if markers is not None else None
        if markers is not None and marker is None:
            continue
        coordinate = marker[axis] if marker is not None else axis_center(box, axis)
        distance = direction * (coordinate - reference)
        if distance > 0 and iou(np.asarray(box), np.asarray(anchor)) < .5:
            options.append((distance if nearest else index, index, box))
    return min(options)[2] if options else None


def top_box(group):
    return group[0]["box"] if group else None


def evaluate(args):
    root = args.manifest.parent
    rows = read(args.manifest)["rows"]
    candidates = candidate_index(args.candidates)
    rankers = {key: candidate_index(Path(value)) for key, value in (x.split("=", 1) for x in args.ranker)}

    @lru_cache(maxsize=4)
    def load(name, sha256):
        return checked_image(root / name, sha256)

    per_item = []
    for row in rows:
        rgb = load(row["marked_image"], row["marked_sha256"])
        spatial = load(row["sp_map"], row["map_sha256"])
        markers = find_markers(rgb)
        point = tuple(row["point"])
        if len(markers) != row.get("marker_count", 2) or point not in markers or rgb.shape != spatial.shape:
            raise ValueError(f"Invalid RGB, marker, or map alignment for {row['id']}")
        digest = row["image_sha256"]
        full = digest, target_phrase(row["question"])
        pool = candidates[digest, row["noun"]]
        anchors = candidates[digest, row["anchor"]]
        anchor = top_box(anchors)
        original = load(row["image"], row["image_sha256"])
        clip_anchor = max(anchors, key=lambda c: c["semantic"], default=None)
        color_anchor = max(anchors, key=lambda c: anchor_score(original, c, row["anchor"], False), default=None)
        appearance_anchor = max(anchors, key=lambda c: anchor_score(original, c, row["anchor"], True), default=None)
        axis, direction, nearest = row["axis"], row["direction"], row.get("nearest", False)
        boxes = {
            "detector_full": top_box(candidates[full]),
            "detector_noun": top_box(pool),
            "relation_rule": relation_box(pool, anchor, axis, direction, nearest),
            "rule_oracle_anchor": relation_box(pool, row["anchor_box"], axis, direction, nearest),
            "relation_marker_rule": relation_box(pool, anchor, axis, direction, nearest, markers),
            "candidate_oracle": next((c["box"] for c in pool if locate_marker(markers, c["box"]) == point), None),
            "oracle_marker": row["box"],
        }
        anchor_options = {"detector": anchor, "clip": clip_anchor["box"] if clip_anchor else None,
                          "color": color_anchor["box"] if color_anchor else None,
                          "appearance": appearance_anchor["box"] if appearance_anchor else None}
        if nearest:
            extreme = max(pool, key=lambda c: direction * axis_center(c["box"], axis), default=None)
            boxes["anchor_ignored"] = extreme["box"] if extreme else None
            unmarked = [c for c in anchors if locate_marker(markers, c["box"]) is None]
            marked_filter = next(iter(unmarked), None)
            grounded_anchor = max(unmarked, key=lambda c: anchor_score(original, c, row["anchor"], True), default=None)
            for method, selected_anchor in (("anchor_clip", clip_anchor), ("anchor_color", color_anchor),
                                            ("anchor_appearance", appearance_anchor),
                                            ("unmarked_detector", marked_filter),
                                            ("unmarked_appearance", grounded_anchor)):
                boxes[method] = relation_box(pool, selected_anchor["box"] if selected_anchor else None,
                                             axis, direction, nearest, markers)
            anchor_options["unmarked_detector"] = marked_filter["box"] if marked_filter else None
            anchor_options["unmarked_appearance"] = grounded_anchor["box"] if grounded_anchor else None
        boxes.update({key: top_box(index[full]) for key, index in rankers.items()})
        methods = {}
        for name, box in boxes.items():
            marker = locate_marker(markers, box)
            if marker is None:
                value = None
            else:
                x, y = marker
                value = int(np.ptp(spatial[y, x]))
            methods[name] = dict(marker=marker, correct_marker=marker == point, prediction=classify(value))
        per_item.append(dict(row, methods=methods, anchor_identity={
            name: iou(np.asarray(selected), np.asarray(row["anchor_box"])) if selected is not None else 0
            for name, selected in anchor_options.items()
        }))
    metrics = {}
    for name in per_item[0]["methods"]:
        outcomes = {r["id"]: r["methods"][name]["prediction"] for r in per_item}
        reference = {r["id"]: r["methods"]["detector_full"]["prediction"] for r in per_item}
        summaries = {}
        subsets = {"all": per_item, "horizontal": [r for r in per_item if r["axis"] == 0],
                   "vertical": [r for r in per_item if r["axis"] == 1]}
        for palette in sorted({r.get("anchor_palette") for r in per_item} - {None}):
            subsets[palette] = [r for r in per_item if r.get("anchor_palette") == palette]
        for subset, selected in subsets.items():
            markers = [r["methods"][name]["correct_marker"] for r in selected if r["side"] == "A"]
            paired = {}
            for r in selected:
                if r["side"] == "A":
                    paired.setdefault(r["scene"], []).append(r["methods"][name]["correct_marker"])
            summaries[subset] = dict(queries=len(markers), correct_marker=sum(markers), binding_pairs=len(paired),
                                     both_markers=sum(all(values) for values in paired.values()),
                                     **counts(selected, outcomes))
        metrics[name] = dict(**summaries, versus_detector_full=cluster_delta(
            [r for r in per_item if r["truth"] != "U"], outcomes, reference))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "metrics.json", metrics)
    write(args.output / "predictions.json", per_item)
    anchor_rows = [r for r in per_item if r["side"] == "A"]
    write(args.output / "anchor-metrics.json", dict(
        n=len(anchor_rows), iou50={name: sum(r["anchor_identity"][name] >= .5 for r in anchor_rows)
                                   for name in anchor_rows[0]["anchor_identity"]},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidates", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--ranker", nargs="+", default=[])
    evaluate(parser.parse_args())
