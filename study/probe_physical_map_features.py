"""Measure point-level workspace information in frozen Qwen3-VL map features."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import ImageOps

from evaluate_frozen_vlm import image, point_crop, read, save, shared_point_crop, shared_targets, workspace_mask
from replay_marked_points import counts

MAPS = ("original", "two_color", "binary")
COLOR_INSIDE = "#db4b52"
COLOR_OUTSIDE = "#2266ad"


def get_rows(path):
    rows = read(path)["rows"]
    if any(row["truth"] not in ("Y", "N") for row in rows):
        raise ValueError("The probe requires certified point labels")
    return rows


def collect(manifest, rows, processor, encoder, shared_target_crop):
    import torch

    features = {f"{encoding}_{pool}": [] for encoding in MAPS for pool in ("point", "global")}
    features["coordinate"] = []
    readout = {}
    target_points = shared_targets(rows) if shared_target_crop else {}
    merge = encoder.visual.spatial_merge_size
    for index, row in enumerate(rows):
        spatial = image(manifest.parent / row["sp_map"], row["map_sha256"])
        width, height = spatial.size
        x, y = row["point"]
        if shared_target_crop:
            region = shared_point_crop(target_points[row["scene"], row["map_sha256"]], width, height)
        else:
            region = point_crop(row, width, height)
        spatial = spatial.crop(region)
        x_local, y_local = x-region[0], y-region[1]
        mask = workspace_mask(spatial)
        readout[row["id"]] = "Y" if mask.getpixel((x_local, y_local)) else "N"
        u, v = (x_local+.5)/spatial.width, (y_local+.5)/spatial.height
        features["coordinate"].append([(x+.5)/width, (y+.5)/height, u, v])
        images = [spatial, ImageOps.colorize(mask, black=COLOR_OUTSIDE, white=COLOR_INSIDE), mask.convert("RGB")]
        inputs = processor.image_processor(images=images, return_tensors="pt").to(encoder.device)
        with torch.inference_mode():
            embeddings, _ = encoder.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
        for mode, embedding, (_, grid_h, grid_w) in zip(
            MAPS, embeddings, inputs.image_grid_thw.tolist(), strict=True
        ):
            h, w = grid_h//merge, grid_w//merge
            grid = embedding.float().cpu().numpy().reshape(h, w, -1)
            features[f"{mode}_point"].append(grid[int(v*h), int(u*w)])
            features[f"{mode}_global"].append(grid.mean(axis=(0, 1)))
        if (index+1) % 32 == 0:
            print(manifest.parent.name, index+1, "/", len(rows), flush=True)
    return {key: np.asarray(values) for key, values in features.items()}, readout


def margins(training, evaluation, labels):
    mean, scale = training.mean(axis=0), training.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = (training-mean)/scale/np.sqrt(training.shape[1])
    test = (evaluation-mean)/scale/np.sqrt(training.shape[1])
    weight = np.where(labels, .5/labels.mean(), .5/(~labels).mean()) ** .5
    design = np.c_[train, np.ones(len(train))] * weight[:, None]
    # A dual ridge fit keeps the solve small with thousands of frozen features.
    coef = design.T @ np.linalg.solve(design @ design.T + np.eye(len(train)), (2*labels-1)*weight)
    return np.c_[test, np.ones(len(test))] @ coef


def run(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    train, confirm = get_rows(args.train), get_rows(args.confirm)
    if {r["scene"] for r in train} & {r["scene"] for r in confirm}:
        raise ValueError("A scene appears in both splits")
    if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in confirm}:
        raise ValueError("An RGB image appears in both splits")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "protocol.json", dict(
        train=dict(path=str(args.train), sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(), n=len(train)),
        confirm=dict(path=str(args.confirm), sha256=hashlib.sha256(args.confirm.read_bytes()).hexdigest(), n=len(confirm)),
        model=str(args.model), model_sha256=args.model_sha256, two_color=[COLOR_OUTSIDE, COLOR_INSIDE],
        ridge=1.0, balanced_training=True, shared_target_crop=args.shared_target_crop,
    ))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    train_features, _ = collect(args.train, train, processor, model, args.shared_target_crop)
    test_features, pixel_readout = collect(args.confirm, confirm, processor, model, args.shared_target_crop)
    labels = np.array([r["truth"] == "Y" for r in train])
    predictions = {r["id"]: dict(id=r["id"], scene=r["scene"], truth=r["truth"],
                                 pixel_readout=pixel_readout[r["id"]]) for r in confirm}
    metrics = {"pixel_readout": counts(confirm, pixel_readout)}
    for name, values in train_features.items():
        probes = margins(values, test_features[name], labels)
        answers = {}
        for row, logit in zip(confirm, probes, strict=True):
            ident = row["id"]
            answers[ident] = "Y" if logit > 0 else "N"
            predictions[ident][name] = dict(margin=float(logit), prediction=answers[ident])
        metrics[name] = counts(confirm, answers)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as records:
        for row in predictions.values():
            records.write(json.dumps(row) + "\n")
    save(args.output / "metrics.json", metrics)
    np.savez_compressed(args.output / "frozen-features.npz",
                        **{f"train_{k}": v for k, v in train_features.items()},
                        **{f"confirm_{k}": v for k, v in test_features.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "confirm", "model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--shared-target-crop", action="store_true",
                        help="Use identical map pixels for the two target points of each scene and workspace.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
