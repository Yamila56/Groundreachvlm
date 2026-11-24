"""Create query-local RGB and workspace crops, and paired projector fixtures."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from audit_matched_pairs import locate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def target_box(rgb, target):
    mask = locate(rgb, target)
    if target == "red block":
        mask &= 1 - locate(rgb, "yellow block")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError(f"Could not locate {target} in RGB")
    component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    ys, xs = np.where(labels == component)
    center_x = (int(xs.min()) + int(xs.max())) // 2
    center_y = (int(ys.min()) + int(ys.max())) // 2
    radius = max(int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
    return (max(0, center_x - radius), max(0, center_y - radius),
            min(rgb.shape[1], center_x + radius + 1), min(rgb.shape[0], center_y + radius + 1))


def within(point, box):
    x, y = point
    left, top, right, bottom = box
    return left <= x < right and top <= y < bottom


def read_bgr(path):
    result = cv2.imread(str(path))
    if result is None:
        raise OSError(f"Could not read {path}")
    return result


def save_crop(output, rgb, box):
    left, top, right, bottom = box
    if not cv2.imwrite(str(output), rgb[top:bottom, left:right]):
        raise OSError(f"Could not write {output}")


def prepare(source, output, *, full_image=False):
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    by_target = defaultdict(list)
    for row in original["rows"]:
        by_target[row["scene"], row["target"]].append(row)
    output.mkdir(parents=True, exist_ok=False)
    result, selected, clean = [], [], 0
    for (scene, target), group in by_target.items():
        rows = sorted(group, key=lambda row: row["side"])
        if len(rows) != 2 or {row["truth"] for row in rows} != {"Y", "N"}:
            raise ValueError(f"Incorrect paired layout {scene}/{target}")
        first = rows[0]
        name = f"{scene}-{target.split()[0]}"
        folder = output / name
        folder.mkdir()
        if full_image:
            rgb_path = os.path.relpath(source / first["image"], folder)
            maps = {row["side"]: os.path.relpath(source / row["sp_map"], folder) for row in rows}
        else:
            rgb = read_bgr(source / first["image"])
            box = target_box(rgb, target)
            fixture = json.loads((source / first["certificate"]).read_text(encoding="utf-8"))
            if not within(fixture["targets"][target]["pixel_xy"], box):
                raise ValueError(f"Crop failed to locate {scene}/{target}")
            clean += not any(within(value["pixel_xy"], box) for name, value in fixture["targets"].items()
                             if name != target)
            rgb_path = "rgb.png"
            save_crop(folder / rgb_path, rgb, box)
            maps = {}
            for row in rows:
                side = row["side"]
                filename = f"sp_map_{side}.png"
                maps[side] = filename
                save_crop(folder / filename, read_bgr(source / row["sp_map"]), box)
                result.append(dict(row, image=f"{name}/rgb.png", image_sha256=sha(folder / rgb_path),
                                   sp_map=f"{name}/{filename}", map_sha256=sha(folder / filename),
                                   uncropped_image_sha256=row["image_sha256"], crop_box=box))
        labels = "".join(row["truth"] for row in rows)
        fixture_name = "training-pair.json"
        write(folder / fixture_name, dict(
            rgb=rgb_path, pairs={target: labels},
            configurations={side: {"sp_map": path} for side, path in maps.items()}))
        selected.append(dict(scene=name, crossed_scene=scene, label=labels, block=first["layout"],
                             family=first["camera_yaw"], color=target, fixture=f"{name}/{fixture_name}"))
    write(output / "selection.json", {"selection": selected})
    if full_image:
        return
    ordered = {row["id"]: row for row in result}
    if len(ordered) != len(original["rows"]):
        raise ValueError("Cropped IDs were not preserved")
    write(output / "manifest.json", dict(original, localization="RGB color mask; identical crop for both streams",
          target_isolated_views=clean, target_views=len(by_target),
          rows=[ordered[row["id"]] for row in original["rows"]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-image", action="store_true", help="Link the original images for a matched training control.")
    args = parser.parse_args()
    prepare(args.source, args.output, full_image=args.full_image)


if __name__ == "__main__":
    main()
