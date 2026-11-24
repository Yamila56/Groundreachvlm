"""Use upstream simulator boxes as an independent target-localization upper bound."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PREFIX = "Is the "
SUFFIX = " in the robot's reachable space?"
THRESHOLD = 83


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(name):
    if len(name) > 2 and name[-2] == "_" and name[-1].isdigit():
        name = name[:-2]
    return name.replace("_", " ")


def mapped_box(box, width, height):
    x0, y0, x1, y1 = box
    x, y = (x0 + x1) * width / 2, (y0 + y1) * height / 2
    radius = max((x1-x0)*width, (y1-y0)*height)
    return (max(0, math.floor(x-radius)), max(0, math.floor(y-radius)),
            min(width, math.ceil(x+radius)), min(height, math.ceil(y+radius)))


def crop_file(path, output, box):
    data = cv2.imread(str(path))
    if data is None:
        raise OSError(f"Unreadable image {path}")
    left, top, right, bottom = box
    crop = data[top:bottom, left:right]
    if crop.size == 0 or not cv2.imwrite(str(output), crop):
        raise OSError(f"Unable to save target crop from {path}")
    return crop


def save(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.manifest.parent
    original = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    control, cropped, excluded = [], [], []
    instances = Counter((r["image_sha256"], r["map_sha256"], r["question"]) for r in original["rows"])
    for row in original["rows"]:
        question = row["question"]
        if not question.startswith(PREFIX) or not question.endswith(SUFFIX):
            raise ValueError(f"Unexpected question {question}")
        target = question[len(PREFIX):-len(SUFFIX)]
        stem = row["image"].replace("_rgb_", "_data_").removesuffix(".jpg")
        meta = json.loads((source / f"{stem}.json").read_text(encoding="utf-8"))
        identity = (row["image_sha256"], row["map_sha256"], question)
        matching = [name for name in meta["models"] if normalized(name) == target]
        if instances[identity] != 1 or len(matching) != 1 or matching[0] not in meta["bounding_boxes"]:
            excluded.append(dict(id=row["id"], target=target, truth=row["truth"], matching=matching))
            continue
        rgb_path, map_path = source / row["image"], source / row["sp_map"]
        if digest(rgb_path) != row["image_sha256"] or digest(map_path) != row["map_sha256"]:
            raise RuntimeError(f"Original recipe input hash changed: {row['id']}")
        rgb = cv2.imread(str(rgb_path))
        height, width = rgb.shape[:2]
        raw_box = meta["bounding_boxes"][matching[0]]
        box = mapped_box(raw_box, width, height)
        rgb_out = args.output / f"{row['id']}-rgb.png"
        map_out = args.output / f"{row['id']}-map.png"
        crop_file(rgb_path, rgb_out, box)
        cropped_map = crop_file(map_path, map_out, box)
        chroma = float(np.median(np.ptp(cropped_map.astype(np.float32), axis=-1)))
        record = dict(row, simulator_instance=matching[0], crop_box=box, box_chroma=chroma,
                      geometry="Y" if chroma >= THRESHOLD else "N")
        control.append(record)
        cropped.append(dict(record, image=rgb_out.name, sp_map=map_out.name,
                            image_sha256=digest(rgb_out), map_sha256=digest(map_out)))
    if not cropped:
        raise ValueError("No unambiguous recipe questions with an upstream box")
    save(args.output / "manifest.json", dict(source_manifest_sha256=digest(args.manifest),
         source="upstream projected object boxes, retrospective", n_original=len(original["rows"]),
         excluded=excluded, rows=cropped))
    save(args.output / "matched-rows.json", control)
    print(json.dumps(dict(eligible=len(cropped), excluded=len(excluded),
                          geometry_correct=sum(r["geometry"] == r["truth"] for r in cropped))))


if __name__ == "__main__":
    main()
