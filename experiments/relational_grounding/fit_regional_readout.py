"""Fit map-only readouts on source-recipe training scenes and evaluate held-out scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from experiments.relational_grounding.grounding_readout import THRESHOLD, cluster_delta, official_boxes, read_predictions, rectangle, region
from experiments.physvlm_reproduction.prepare_official_box_probe import PREFIX, SUFFIX


def summarize_region(chroma, box):
    full = region(chroma, rectangle(box, chroma.shape))
    center = region(chroma, rectangle(box, chroma.shape, .5))
    result = []
    for view in (full, center):
        high = (view >= THRESHOLD).astype(np.uint8)
        _, _, components, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
        largest = components[1:, cv2.CC_STAT_AREA].max(initial=0)
        result.extend([*np.quantile(view, [.10, .5, .90])/255, high.mean(), largest/high.size])
    return result


def load_split(path, automatic=None):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    choices = {}
    if automatic:
        choices = {(r["image_sha256"], r["phrase"]): r["boxes"]
                   for r in json.loads(automatic.read_text(encoding="utf-8"))}
    result = []
    for row in rows:
        phrase = row["question"][len(PREFIX):-len(SUFFIX)]
        map_path = path.parent/row["sp_map"]
        if hashlib.sha256(map_path.read_bytes()).hexdigest() != row["map_sha256"]:
            raise ValueError(f"Unexpected S-P Map hash for {row['id']}")
        spatial_map = cv2.imread(str(map_path))
        if spatial_map is None:
            raise ValueError(f"Unreadable S-P Map for {row['id']}")
        if automatic:
            candidates = choices[row["image_sha256"], phrase]
            box = np.array(candidates[0]["box"]) if candidates else None
        else:
            box, _ = official_boxes(path.parent, row, phrase, spatial_map.shape[:2])
        chroma = np.ptp(spatial_map, axis=-1)
        features = None if box is None or rectangle(box, chroma.shape, .5) is None else summarize_region(chroma, box)
        result.append(dict(id=row["id"], truth=row["truth"], layout=row["image_sha256"], features=features))
    return result


def prepare(rows, center, scale):
    values = torch.tensor([row["features"] for row in rows], dtype=torch.float32).reshape(-1, center.numel())
    return (values-center)/scale


def fit(training, validation, depth, seed, center, scale):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 12), nn.GELU(), nn.Linear(12, 1)) if depth else nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.025, weight_decay=.05)
    x = prepare(training, center, scale)
    actual = torch.tensor([row["truth"] == "Y" for row in training], dtype=torch.float32)
    dev = prepare(validation, center, scale)
    gold = torch.tensor([row["truth"] == "Y" for row in validation])
    optimum = -1
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), actual).backward()
        optimizer.step()
        with torch.no_grad():
            correct = int(((model(dev).flatten() >= 0) == gold).sum())
            if correct > optimum:
                optimum, selected_step = correct, step+1
                selected = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(selected)
    return model.eval(), dict(step=selected_step, correct=optimum, n=len(validation))


@torch.no_grad()
def evaluate(rows, models, center, scale, references):
    actual = {r["id"]: r["truth"] for r in rows}
    covered = [r for r in rows if r["features"] is not None]
    threshold = THRESHOLD/255
    predictions = {
        "whole_box": {r["id"]: "Y" if r["features"][1] >= threshold else "N" for r in covered},
        "center_box": {r["id"]: "Y" if r["features"][6] >= threshold else "N" for r in covered},
    }
    x = prepare(covered, center, scale)
    for name, model in models.items():
        predictions[name] = {r["id"]: "Y" if value >= 0 else "N"
                             for r, value in zip(covered, model(x).flatten().tolist(), strict=True)}
    for p in predictions.values():
        p.update({r["id"]: "U" for r in rows if r["features"] is None})
    return dict(n=len(rows), covered=len(covered),
                methods={name: dict(
                    correct=sum(value == actual[key] for key, value in p.items()),
                    by_label={t: sum(p[k] == t for k, v in actual.items() if v == t) for t in ("Y", "N")},
                    paired_vs_center=cluster_delta(rows, p, predictions["center_box"]),
                    paired_vs_reference={key: cluster_delta(rows, p, values) for key, values in references.items()},
                ) for name, p in predictions.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("train", "dev", "test", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--predicted-boxes", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--detector-score", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    splits = {name: load_split(getattr(args,name)) for name in ("train", "dev", "test")}
    hashes = {name: {r["layout"] for r in rows} for name, rows in splits.items()}
    if hashes["train"] & hashes["dev"] or (hashes["train"] | hashes["dev"]) & hashes["test"]:
        raise ValueError("Regional readout splits share RGB scenes")
    training, dev = ([r for r in splits[k] if r["features"] is not None] for k in ("train", "dev"))
    if {r["truth"] for r in training} != {"Y", "N"} or not dev:
        raise ValueError("Readout fitting needs both training labels and a nonempty development split")
    values = torch.tensor([r["features"] for r in training], dtype=torch.float32)
    center, scale = values.mean(0), values.std(0, unbiased=False).clamp_min(.01)
    models, validation = {}, {}
    for depth, name in ((0, "logistic"), (1, "mlp")):
        for seed in (17, 29):
            key = f"{name}-{seed}"
            models[key], validation[key] = fit(training, dev, depth, seed, center, scale)
    references = {}
    if args.frozen:
        references["frozen"] = read_predictions(args.frozen, "original")
    if args.detector_score:
        references["detector_center"] = json.loads(args.detector_score.read_text(encoding="utf-8"))[
            "predictions"]["predicted-center"]
    if any(values.keys() != {r["id"] for r in splits["test"]} for values in references.values()):
        raise ValueError("Baseline and readout used different test questions")
    result = dict(train_n=len(training), dev_n=len(dev), validation=validation,
                  oracle=evaluate(splits["test"], models, center, scale, references))
    if args.predicted_boxes:
        result["automatic"] = evaluate(load_split(args.test, args.predicted_boxes), models, center, scale, references)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
