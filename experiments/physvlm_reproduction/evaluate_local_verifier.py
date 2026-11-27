"""Calibrate a local-map verifier and evaluate it on fresh author-recipe scenes."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.physvlm_reproduction.evaluate_target_alignment import load_rows, write_json
from experiments.physvlm_reproduction.score_official_box_probe import paired, summarize

THRESHOLD = 83


def source_rows(source):
    value = json.loads(source.read_text(encoding="utf-8"))
    return {row["id"]: row for row in value["rows"]}


def inference(source, prediction):
    predictions = {row["id"]: row for row in load_rows(prediction)}
    if predictions.keys() != source.keys():
        raise ValueError("Wrong original-recipe prediction IDs")
    for key, row in predictions.items():
        reference = source[key]
        if any(row[field] != reference[field] for field in ("image_sha256", "map_sha256", "truth", "question")):
            raise ValueError(f"Recipe inputs or targets changed for {key}")
    return predictions


def choose_answer(row, item, margin):
    if item is None or abs(item["box_chroma"]-THRESHOLD) < margin:
        return row["original"]["free"]
    return item["geometry"]


def calibration(args):
    source = source_rows(args.source)
    boxes = {row["id"]: row for row in json.loads(args.boxes.read_text())}
    raw = {row["id"]: row for row in load_rows(args.prediction)}
    # Conflicting questions were already removed by matching simulator instances.
    dev = [raw[key] for key in boxes]
    if any(row["truth"] != source[row["id"]]["truth"] for row in dev):
        raise ValueError("Wrong source labels")
    margin_grid = (0, 5, 10, 15, 20, 30, 40, 256)
    candidates = []
    for margin in margin_grid:
        correct = sum(row["truth"] == choose_answer(row, boxes[row["id"]], margin) for row in dev)
        coverage = sum(abs(boxes[row["id"]]["box_chroma"]-THRESHOLD) >= margin for row in dev)
        candidates.append(dict(margin=margin, correct=correct, coverage=coverage))
    selected = max(candidates, key=lambda r: (r["correct"], -r["coverage"], -r["margin"]))
    write_json(args.output, dict(
        threshold=THRESHOLD, margin=selected["margin"], dev_n=len(dev), candidates=candidates,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_image_sha256s=sorted({row["image_sha256"] for row in source.values()}),
        tie_break="higher accuracy, then fewer overrides, then smaller margin",
    ))


def evaluation(args):
    source = source_rows(args.source)
    boxes = {row["id"]: row for row in json.loads(args.boxes.read_text())}
    frozen = inference(source, args.prediction)
    config = json.loads(args.calibration.read_text(encoding="utf-8"))
    if set(config["source_image_sha256s"]) & {row["image_sha256"] for row in source.values()}:
        raise ValueError("Development and test RGB overlap")
    patched, reference, local = [], [], []
    eligible = sorted(boxes)
    labels_by_input = {}
    for row in source.values():
        key = (row["image_sha256"], row["map_sha256"], row["question"])
        labels_by_input.setdefault(key, set()).add(row["truth"])
    ambiguous = {r["id"] for r in source.values()
                 if len(labels_by_input[r["image_sha256"], r["map_sha256"], r["question"]]) > 1}
    overrides = 0
    identity = {key: value["image_sha256"] for key, value in source.items()}
    for key, row in sorted(frozen.items()):
        if key in ambiguous:
            continue
        hint = boxes.get(key)
        answer = choose_answer(row, hint, config["margin"])
        reference.append(row)
        patched.append(dict(row, original=dict(free=answer)))
        overrides += hint is not None and abs(hint["box_chroma"]-THRESHOLD) >= config["margin"]
        if hint is not None:
            local.append(dict(row, original=dict(free=hint["geometry"])))
    baseline_local = [row for row in reference if row["id"] in boxes]
    write_json(args.output, dict(
        scope="Panda with upstream simulator boxes; fresh scene seeds, shared asset catalog",
        source_n=len(source), excluded_contradictory=len(ambiguous), box_eligible_n=len(eligible),
        configured_margin=config["margin"], policy_active=overrides,
        frozen=summarize(reference, "original"), gated=summarize(patched, "original"),
        gated_vs_frozen=paired(reference, patched, identity, "original"),
        frozen_box_eligible=summarize(baseline_local, "original"),
        fixed_threshold_box_eligible=summarize(local, "original"),
        fixed_threshold_vs_frozen=paired(baseline_local, local, identity, "original"),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("calibrate", "evaluate"):
        action = modes.add_parser(mode)
        for arg in ("source", "boxes", "prediction", "output"):
            action.add_argument(f"--{arg}", type=Path, required=True)
        if mode == "evaluate":
            action.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "calibrate":
        calibration(args)
    else:
        evaluation(args)


if __name__ == "__main__":
    main()
