"""Audit target identity and recall independently of reachability labels."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiments.relational_grounding.grounding_readout import THRESHOLD, iou, rectangle, region, save
from experiments.physvlm_reproduction.prepare_official_box_probe import PREFIX, SUFFIX, normalized

TOP_K = (1, 3, 5, 10)


def deduplicate(candidates):
    kept = []
    for candidate in sorted(candidates, key=lambda value: value.get("proposal_rank", -value["score"])):
        box = np.array(candidate["box"], dtype=float)
        if all(iou(box, existing["box"]) <= .7 for existing in kept):
            kept.append(dict(box=box, score=candidate["score"]))
    return kept


def reference_boxes(source, row, height, width):
    if "box" in row:
        return {entry["target"]: np.array(entry["box"], dtype=float)
                for entry in source["by_image"][row["image_sha256"]]}
    filename = row["image"].replace("_rgb_", "_data_").removesuffix(".jpg")
    metadata = json.loads((source["root"] / f"{filename}.json").read_text(encoding="utf-8"))
    return {key: np.array(value)*np.tile((width, height), 2) for key, value in metadata["bounding_boxes"].items()}


def identity(candidate, targets):
    overlaps = sorted(((iou(candidate["box"], box), key) for key, box in targets.items()), reverse=True)
    if not overlaps or overlaps[0][0] < .10 or len(overlaps) > 1 and overlaps[0][0] == overlaps[1][0]:
        return None
    return overlaps[0][1]


def inspect(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    detections = json.loads(args.boxes.read_text(encoding="utf-8"))
    indexed = {(r["image_sha256"], r["phrase"]): r["boxes"] for r in detections}
    source = dict(root=args.manifest.parent, by_image=defaultdict(list))
    for row in manifest["rows"]:
        if "box" in row and row["side"] == "A":
            source["by_image"][row["image_sha256"]].append(row)
    unique = {}
    detailed = []
    for row in manifest["rows"]:
        phrase = row["question"][len(PREFIX):-len(SUFFIX)]
        key = row["image_sha256"], phrase
        if key not in unique:
            img = cv2.imread(str(source["root"] / row["image"]))
            refs = reference_boxes(source, row, *img.shape[:2])
            matching = [name for name in refs if (name if "box" in row else normalized(name)) == phrase]
            target = matching[0] if len(matching) == 1 else None
            options = deduplicate(c for c in indexed[key] if c["score"] >= args.confidence)
            assessed = [dict(box=c["box"].tolist(), score=c["score"], matched=identity(c, refs),
                             iou=iou(c["box"], refs[target]) if target else None) for c in options[:10]]
            unique[key] = dict(image_sha256=key[0], phrase=phrase, target=target, candidates=assessed)
        audit = unique[key]
        choices = audit["candidates"]
        first = choices[0] if choices else None
        target = audit["target"]
        if target is None:
            category = "reference_unavailable"
        elif first is None:
            category = "no_prediction"
        elif first["matched"] == target:
            category = "target_iou_at_least_half" if first["iou"] >= .5 else "target_low_iou"
        else:
            category = "other_object" if first["matched"] is not None else "unmatched_region"
        candidates = [c for c in choices[:5] if target is not None and c["matched"] == target]
        oracle = max(candidates, key=lambda c: c["iou"]) if candidates else None
        map_image = cv2.imread(str(source["root"] / row["sp_map"]))
        chroma = np.ptp(map_image.astype(np.float32), axis=-1)

        def read(candidate):
            if candidate is None or (box := rectangle(np.array(candidate["box"]), map_image.shape)) is None:
                return None
            return "Y" if np.median(region(chroma, box)) >= THRESHOLD else "N"

        detailed.append(dict(id=row["id"], category=category, truth=row["truth"], top1=read(first),
                             oracle_top5=read(oracle), iou=first["iou"] if first else None))
    eligible = [r for r in unique.values() if r["target"] is not None]
    recalls = {str(k): dict(
        identity=sum(any(c["matched"] == r["target"] for c in r["candidates"][:k]) for r in eligible),
        overlap=sum(any(c["iou"] >= .5 for c in r["candidates"][:k]) for r in eligible),
    ) for k in TOP_K}
    breakdown = defaultdict(Counter)
    for row in detailed:
        b = breakdown[row["category"]]
        b["n"] += 1
        b["top1_correct"] += row["top1"] == row["truth"]
        b["top1_unanswered"] += row["top1"] is None
        b["oracle_top5_covered"] += row["oracle_top5"] is not None
        b["oracle_top5_correct"] += row["oracle_top5"] == row["truth"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, dict(n_unique=len(unique), n_unique_annotated=len(eligible), confidence=args.confidence,
                           manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                           recalls=recalls, query_readout=breakdown, rows=detailed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("boxes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confidence", type=float, default=.25)
    inspect(parser.parse_args())
