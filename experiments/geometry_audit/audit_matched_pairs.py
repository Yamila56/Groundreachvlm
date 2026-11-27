"""Audit matched v2 reachability fixtures and build input-only evaluation rows."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(directory):
    manifest = json.loads((directory / "selection.json").read_text())
    return [(row, directory / row["fixture"]) for row in manifest["selection"]]


def audit(root):
    hashes, families = set(), set()
    result = {}
    for split in ("train", "dev", "test"):
        rows = data(root / split)
        split_hashes = {entry["rgb_sha256"] for entry, _ in rows}
        split_families = {entry["family"] for entry, _ in rows}
        if len(split_hashes) != len(rows) or hashes & split_hashes or families & split_families:
            raise ValueError(f"Split overlap: {split}")
        hashes |= split_hashes
        families |= split_families
        labels_by_pixel = {}
        margins, errors = [], []
        for row, path in rows:
            fixture = json.loads(path.read_text())
            rgb = path.parent / fixture["rgb"]
            if digest(path) != row["fixture_sha256"] or digest(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Fixture changed: {path}")
            target = next(iter(fixture["pairs"]))
            if fixture["pairs"][target] != row["label"]:
                raise ValueError(f"Pair label changed: {path}")
            labels_by_pixel.setdefault(tuple(row["pixel"]), Counter())[row["label"]] += 1
            for conf in fixture["configurations"].values():
                certificate = conf["targets"][target]
                if certificate["label"] == "Y":
                    errors.append(certificate["fk_error_m"])
                else:
                    margins.append(certificate["distance_m"] - certificate["radius_m"])
        # Exact equality proves that any deterministic pixel-only function gets 25% on
        # this split, including arbitrary classifiers trained on a different split.
        if any(len(c) != 4 or len(set(c.values())) != 1 for c in labels_by_pixel.values()):
            raise ValueError(f"Pixel distribution mismatch: {split}")
        result[split] = dict(n=len(rows), coordinate_only_accuracy=.25, families=sorted(split_families),
                             min_outer_margin_m=min(margins), max_fk_error_m=max(errors))
    return result


COLOR_INDEX = {"red block": 2, "green block": 1, "blue block": 0}


def locate(rgb, color):
    pix = rgb.astype(np.int16)
    if color == "yellow block":
        mask = (pix[:, :, 1] - pix[:, :, 0] > 30) & (pix[:, :, 2] - pix[:, :, 0] > 30)
    else:
        index = COLOR_INDEX[color]
        others = [i for i in range(3) if i != index]
        mask = pix[:, :, index] - pix[:, :, others].max(2) > 42
    return mask.astype(np.uint8)


def measure_geometry(root, include_test=False):
    import cv2

    datasets = {}
    for split in ("train", "dev", "test") if include_test else ("train", "dev"):
        samples = []
        for _, path in data(root / split):
            fixture = json.loads(path.read_text())
            target = next(iter(fixture["pairs"]))
            rgb = cv2.imread(str(path.parent / fixture["rgb"]))
            mask = locate(rgb, target) > 0
            labels, radii, chroma = [], [], []
            for conf in fixture["configurations"].values():
                sp_map = cv2.imread(str(path.parent / conf["sp_map"])).astype(np.float32)
                labels.append(conf["targets"][target]["label"] == "Y")
                radii.append(conf["rules"][target]["workspace_label"] == "Y")
                # The upstream map uses a grey blend of the same pseudo-color outside the workspace.
                chroma.append(float(np.median(np.ptp(sp_map[mask], axis=1))) if mask.any() else -1)
            samples.append((labels, radii, chroma))
        datasets[split] = samples
    train = datasets["train"]
    thresholds = sorted({v for _, _, values in train for v in values})
    threshold = max(thresholds, key=lambda t: sum((v >= t) == y for ys, _, vs in train
                                                  for v, y in zip(vs, ys, strict=True)))
    result = {}
    for split, samples in datasets.items():
        result[split] = dict(
            localization_failures=sum(any(v < 0 for v in vs) for _, _, vs in samples),
            image_pair_accuracy=sum([v >= threshold for v in vs] == ys for ys, _, vs in samples)/len(samples),
            oracle_pointcloud_pair_accuracy=sum(rs == ys for ys, rs, _ in samples)/len(samples),
            total_pairs=len(samples))
    return dict(threshold=threshold, results=result)


def rgb_baseline(root):
    import cv2
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeClassifier

    matrices = {}
    for split in ("train", "dev", "test"):
        rgb_features, labels = [], []
        for row, path in data(root / split):
            rgb = cv2.imread(str(path.parent / "rgb.png"))
            descriptors = []
            for color in (*COLOR_INDEX, "yellow block"):
                ys, xs = np.where(locate(rgb, color))
                descriptors.extend((len(xs), float(xs.mean()) if len(xs) else -1,
                                    float(ys.mean()) if len(ys) else -1))
            rgb_features.append([*row["pixel"], *descriptors, color_index(row["color"])])
            labels.append(row["label"])
        matrices[split] = (np.asarray(rgb_features), labels)
    train_x, train_y = matrices["train"]
    classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=31)
    classifier.fit(train_x, train_y)
    return {split: dict(accuracy=float(classifier.score(x, y)), n=len(y),
                       labels=classifier.classes_.tolist(),
                       confusion=confusion_matrix(y, classifier.predict(x),
                                                  labels=classifier.classes_).tolist())
            for split, (x, y) in matrices.items()}


def color_index(color):
    return (*COLOR_INDEX, "yellow block").index(color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rgb-baseline", action="store_true")
    args = parser.parse_args()
    result = dict(splits=audit(args.root), geometry=measure_geometry(args.root))
    if args.rgb_baseline:
        result["rgb_baseline"] = rgb_baseline(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
