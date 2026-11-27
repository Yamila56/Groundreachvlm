"""Generate paired scenes with exactly matched target-pixel distributions by label."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.physvlm_reproduction.generate_paired_dataset import COLORS
from groundreach.simulation.panda import generate_pair
from groundreach.simulation.upstream import SIMULATOR

STRATA = ("YN", "NY", "YY", "NN")
FAMILIES = {
    "train": (0, 30, 90, 150, 180, 240, 300, 330),
    "dev": (50, 125, 205, 280),
    "test": (15, 70, 110, 165, 225, 260, 315, 345),
}
COUNTS = {"train": 256, "dev": 64, "test": 160}
SEEDS = {"train": 31000, "dev": 41000, "test": 51000, "pilot": 61000}


def bases_for(rng, target, family, label):
    result = []
    swap = bool(rng.integers(2))
    for side, value in enumerate(label):
        bearing = family + 40 * (side ^ swap) + rng.uniform(-3, 3)
        angle = np.deg2rad(bearing)
        radius = rng.uniform(.42, .53) if value == "Y" else rng.uniform(1.05, 1.13)
        result.append([target[0] - radius * np.cos(angle), target[1] - radius * np.sin(angle), .20])
    return result


def generate(output, split, count):
    output.mkdir(parents=True, exist_ok=False)
    pool = FAMILIES["dev" if split == "pilot" else split]
    rng = np.random.default_rng(SEEDS[split])
    selection, rejected = [], []
    for candidate in range(count):
        block = len(selection) // 4
        family = pool[block % len(pool)]
        x, y = rng.uniform(-.30, .82), rng.uniform(-.27, .27)
        target = (x, y, .245)
        labels = rng.permutation(STRATA).tolist()
        colors = list(rng.permutation(list(COLORS)))
        quartet = []
        for slot, (label, target_color) in enumerate(zip(labels, colors, strict=True)):
            scene_id = f"{split}-{candidate:04d}-{slot}"
            distractor_color = colors[(slot + int(rng.integers(1, 4))) % 4]
            while True:
                dx, dy = rng.uniform(-.68, 1.21), rng.uniform(-.44, .44)
                if abs(dx - x) > .15 or abs(dy - y) > .16:
                    break
            layout = {target_color: (target, COLORS[target_color]),
                      distractor_color: ((dx, dy, .245), COLORS[distractor_color])}
            directory = output / scene_id
            directory.mkdir()
            actual = generate_pair(directory, layout, target_color, scene_id,
                                   bases_for(rng, target, family, label))[target_color]
            if actual != label:
                rejected.append({"scene": scene_id, "required": label, "observed": actual})
                break
            path = directory / "paired_pilot.json"
            fixture = json.loads(path.read_text())
            quartet.append(dict(scene=scene_id, fixture=f"{scene_id}/paired_pilot.json", label=label,
                                family=family, block=block, color=target_color,
                                pixel=fixture["targets"][target_color]["pixel_xy"],
                                rgb_sha256=fixture["rgb_sha256"],
                                fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(quartet) != 4:
            continue
        if len({tuple(row["pixel"]) for row in quartet}) != 1:
            raise ValueError(f"Target coordinate changed within {split}/{block}")
        selection.extend(quartet)
        print(json.dumps({"split": split, "accepted": len(selection), "target": count}), flush=True)
        if len(selection) == count:
            break
    payload = dict(split=split, seed=SEEDS[split], families=pool, target=count, selection=selection,
                   accepted=dict(Counter(row["label"] for row in selection)), rejected=rejected)
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    if len(selection) != count or len({r["rgb_sha256"] for r in selection}) != count:
        raise RuntimeError(f"Could not fill unique matched blocks: {split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(SIMULATOR))
    if args.pilot:
        generate(args.output / "pilot", "pilot", 32)
    else:
        for split, count in COUNTS.items():
            generate(args.output / split, split, count)


if __name__ == "__main__":
    main()
