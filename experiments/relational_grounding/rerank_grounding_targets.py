"""Calibrate word-level box reranking on development images and freeze the choice."""
import argparse
import hashlib
import json
from collections import defaultdict
from itertools import product, zip_longest
from math import log
from pathlib import Path

import cv2
import numpy as np

from experiments.relational_grounding.grounding_readout import iou, official_boxes, rectangle, region, save
from experiments.physvlm_reproduction.prepare_official_box_probe import PREFIX, SUFFIX

RANKING_KEYS = ("score", "all_words", "last_word")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select(candidates, index, confidence):
    allowed = [candidate for candidate in candidates if candidate["score"] >= confidence]
    return max(allowed, key=lambda c: c[index]) if allowed else None


def assign_distinct(rows, confidence):
    """Ground exactly two known-distinct descriptions that share an RGB image."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_sha256"]].append(row)
    result = []
    for group in grouped.values():
        if len(group) != 2:
            raise ValueError("Distinct assignment requires two requested targets per RGB image")
        options = [[c for c in row["boxes"][:3] if c["score"] >= confidence] for row in group]
        feasible = [(a, b) for a, b in product(*options)
                    if iou(np.array(a["box"]), np.array(b["box"])) < .5]
        selected = max(feasible, key=lambda pair: sum(log(c["score"]) for c in pair)) if feasible else [
            select(row["boxes"], "score", confidence) for row in group
        ]
        result.extend(dict(row, boxes=[box] if box else []) for row, box in zip(group, selected, strict=True))
    return result


def rank_rgb_objectness(manifest, rows, confidence):
    paths = {row["image_sha256"]: row["image"] for row in read(manifest)["rows"]}
    result = []
    for row in rows:
        rgb = cv2.imread(str(manifest.parent / paths[row["image_sha256"]]))
        if rgb is None:
            raise ValueError(f"Missing RGB for {row['image_sha256']}")
        chroma = np.ptp(rgb.astype(np.float32), axis=-1) / 255
        choices = []
        for candidate in row["boxes"]:
            box = rectangle(np.array(candidate["box"]), rgb.shape)
            if box is not None and candidate["score"] >= confidence:
                score = candidate["score"] * float(region(chroma, box).mean())
                choices.append((score, candidate))
        selected = max(choices, key=lambda item: item[0])[1] if choices else None
        result.append(dict(row, boxes=[selected] if selected else []))
    return result


def union_proposals(full, final_word):
    alternative = {(row["image_sha256"], row["phrase"]): row["boxes"] for row in final_word}
    result = []
    for row in full:
        key = row["image_sha256"], row["phrase"]
        candidates = (candidate for pair in zip_longest(row["boxes"], alternative[key])
                      for candidate in pair if candidate is not None)
        result.append(dict(row, boxes=[dict(candidate, proposal_rank=rank) for rank, candidate in enumerate(candidates)]))
    return result


def select_semantic(rows, weight):
    result = []
    for row in rows:
        options = row["boxes"]
        if options:
            detector = np.array([option["score"] for option in options])
            semantic = np.array([option["semantic"] for option in options])
            combined = ((1-weight)*(detector-detector.mean())/max(detector.std(), 1e-6)
                        + weight*(semantic-semantic.mean())/max(semantic.std(), 1e-6))
            options = [options[int(np.argmax(combined))]]
        result.append(dict(row, boxes=options))
    return result


def calibration(manifest, candidates, confidence, ranking_keys=RANKING_KEYS):
    rows = read(manifest)["rows"]
    by_key = {(r["image_sha256"], r["phrase"]): r["boxes"] for r in candidates}
    observed = set()
    totals = {key: 0 for key in ranking_keys}
    oracle3 = 0
    for row in rows:
        phrase = row["question"][len(PREFIX):-len(SUFFIX)]
        key = row["image_sha256"], phrase
        if key in observed:
            continue
        targets = by_key[key]
        if "box" in row:
            oracle = np.array(row["box"])
        else:
            image = cv2.imread(str(manifest.parent / row["image"]))
            oracle, _ = official_boxes(manifest.parent, row, phrase, image.shape)
        if oracle is None:
            continue
        observed.add(key)
        for score in ranking_keys:
            prediction = select(targets, score, confidence)
            totals[score] += prediction is not None and iou(oracle, np.array(prediction["box"])) >= .5
        oracle3 += any(iou(oracle, np.array(b["box"])) >= .5 for b in targets[:3])
    return dict(n=len(observed), iou_at_least_half=totals, correct_in_top_three=oracle3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    develop = commands.add_parser("calibrate")
    for field in ("dev", "dev_candidates", "source_dev", "source_candidates", "output"):
        develop.add_argument(field, type=Path)
    develop.add_argument("--confidence", type=float, default=.25)
    apply = commands.add_parser("apply")
    for field in ("calibration", "candidates", "output"):
        apply.add_argument(field, type=Path)
    joint = commands.add_parser("distinct")
    joint.add_argument("candidates", type=Path)
    joint.add_argument("output", type=Path)
    joint.add_argument("--confidence", type=float, default=.25)
    color = commands.add_parser("rgb-objectness")
    color.add_argument("manifest", type=Path)
    color.add_argument("candidates", type=Path)
    color.add_argument("output", type=Path)
    color.add_argument("--confidence", type=float, default=.25)
    union = commands.add_parser("multi-query")
    union.add_argument("full", type=Path)
    union.add_argument("final_word", type=Path)
    union.add_argument("output", type=Path)
    semantic = commands.add_parser("semantic")
    semantic.add_argument("candidates", type=Path)
    semantic.add_argument("output", type=Path)
    semantic.add_argument("--weight", type=float, choices=(0, .25, .5, .75, 1), required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "calibrate":
        main_set = calibration(args.dev, read(args.dev_candidates), args.confidence)
        source_set = calibration(args.source_dev, read(args.source_candidates), args.confidence)
        key = max(RANKING_KEYS, key=lambda name: (main_set["iou_at_least_half"][name],
                                                  source_set["iou_at_least_half"][name]))
        save(args.output, dict(sort_key=key, confidence=args.confidence, controlled=main_set, upstream=source_set,
                               dev_manifest_sha256=hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                               source_dev_manifest_sha256=hashlib.sha256(args.source_dev.read_bytes()).hexdigest()))
    elif args.mode == "apply":
        config = read(args.calibration)
        reordered = []
        for row in read(args.candidates):
            selected = select(row["boxes"], config["sort_key"], config["confidence"])
            reordered.append(dict(row, boxes=[selected] if selected else []))
        save(args.output, reordered)
    elif args.mode == "distinct":
        save(args.output, assign_distinct(read(args.candidates), args.confidence))
    elif args.mode == "rgb-objectness":
        save(args.output, rank_rgb_objectness(args.manifest, read(args.candidates), args.confidence))
    elif args.mode == "semantic":
        save(args.output, select_semantic(read(args.candidates), args.weight))
    else:
        save(args.output, union_proposals(read(args.full), read(args.final_word)))


if __name__ == "__main__":
    main()
