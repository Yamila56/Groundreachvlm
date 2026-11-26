"""Evaluate a frozen generic VLM with RGB, a spatial map, and optional automatic boxes."""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops

LEGEND = (
    "Image 1 is the RGB scene. Image 2 is a spatial map aligned pixel-for-pixel with it. "
    "Small black marks in image 1 identify the points to check on the object surfaces. "
    "A point is reachable if its map color is strongly saturated "
    "(largest minus smallest RGB component is at least 83 on the 0–255 scale). "
    "Check the point on the object named by the question. Respond with exactly Yes or No."
)


def auto_boxes(rows, root, candidate_path, reference_path):
    from functools import lru_cache

    from evaluate_marked_points import candidate_index, checked_image
    from evaluate_relation_holdout import anchor_score, relation_box
    from point_markers import find_markers, locate_marker

    candidates = candidate_index(candidate_path)
    reference = {row["id"]: row["methods"]["unmarked_appearance"] for row in read(reference_path)}

    @lru_cache(maxsize=4)
    def load(path, digest):
        return checked_image(root / path, digest)

    positions = {}
    for row in rows:
        key = row["image_sha256"]
        original = load(row["image"], key)
        marked = load(row["marked_image"], row["marked_sha256"])
        markers = find_markers(marked)
        anchor = max((c for c in candidates[key, row["anchor"]] if locate_marker(markers, c["box"]) is None),
                     key=lambda c: anchor_score(original, c, row["anchor"], True), default=None)
        box = relation_box(candidates[key, row["noun"]], anchor["box"] if anchor else None,
                           row["axis"], row["direction"], row["nearest"], markers)
        actual = locate_marker(markers, box) if box is not None else None
        expected = reference[row["id"]]["marker"]
        if (list(actual) if actual is not None else None) != expected:
            raise ValueError(f"Autobox replay differs at {row['id']}")
        positions[row["id"]] = box
    return positions, {key: record["prediction"] for key, record in reference.items()}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def image(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Image hash changed: {path}")
    with Image.open(path) as source:
        return source.convert("RGB")


def workspace_mask(spatial):
    r, g, b = spatial.split()
    brightest = ImageChops.lighter(ImageChops.lighter(r, g), b)
    darkest = ImageChops.darker(ImageChops.darker(r, g), b)
    return ImageChops.subtract(brightest, darkest).point(
        [255 if value >= 83 else 0 for value in range(256)]
    )


def point_crop(row, width, height):
    x, y = row["point"]
    x0, y0, x1, y1 = row["box"]
    half = max(64, x-x0, x1-x, y-y0, y1-y)
    return (max(0, x-half), max(0, y-half), min(width, x+half), min(height, y+half))


def shared_targets(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"], row["map_sha256"]].append(row["point"])
    if any(len(points) != 2 or points[0] == points[1] for points in groups.values()):
        raise ValueError("Shared crops require two distinct target points for every scene-map pair")
    return groups


def shared_point_crop(points, width, height):
    xs, ys = zip(*points)
    return (max(0, min(xs)-48), max(0, min(ys)-48), min(width, max(xs)+49), min(height, max(ys)+49))


def oracle_view(row, mode, rgb, spatial, points=None):
    width, height = rgb.size
    x, y = row["point"]
    if spatial.size != rgb.size or not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Invalid aligned point for {row['id']}")
    if mode == "oracle_box":
        box = tuple(round(value) for value in row["box"])
        return rgb, spatial, (f" The target box in the {width} by {height} image is "
                              f"{box} in xyxy pixel coordinates from the upper left.")
    if mode in ("oracle_crop", "binary_crop", "shared_point_control"):
        # Give both inputs the same view, with enough context around a small target.
        crop = shared_point_crop(points, width, height) if points is not None else point_crop(row, width, height)
        rgb, spatial = rgb.crop(crop), spatial.crop(crop)
        x, y = x-crop[0], y-crop[1]
    return rgb, spatial, (f" The target surface point in image 1 is at (x={x}, y={y}) "
                          f"in the {rgb.width} by {rgb.height} pixel image, from the upper left.")


def evaluate(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from grounding_readout import cluster_delta
    from replay_marked_points import counts

    root = args.manifest.parent
    rows = read(args.manifest)["rows"]
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.shared_point_control:
        targets = shared_targets(rows)
        boxes = dict.fromkeys(row["id"] for row in rows)
        reference = {}
    else:
        targets = {}
        boxes, reference = auto_boxes(rows, root, args.candidates, args.reference)
    conditions = ("without_box", "with_auto_box")
    if args.oracle:
        conditions = ("oracle_box", "oracle_point", "oracle_crop")
    if args.binary_crop:
        conditions = ("binary_crop",)
    if args.full_point_control:
        conditions = ("full_point_control",)
    if args.shared_point_control:
        conditions = ("shared_point_control",)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "protocol.json", dict(manifest=str(args.manifest), model=str(args.model),
         model_sha256=args.model_sha256, cases=len(rows), dtype="bfloat16", cap=8, legend=LEGEND,
         answer_mode=args.answer_mode, conditions=conditions,
         manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest()))
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                           local_files_only=True).to("cuda").eval()
    choices = ("Yes", "No")
    choice_ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in choices]
    if args.answer_mode == "choice" and any(len(ids) != 1 for ids in choice_ids):
        raise ValueError("Next-token scoring requires single-token Yes and No")
    print(f"Loaded in {time.perf_counter()-start:.1f}s", flush=True)
    predictions = []
    with torch.inference_mode(), (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            rgb = image(root / row["marked_image"], row["marked_sha256"])
            spatial = image(root / row["sp_map"], row["map_sha256"])
            if args.shared_point_control:
                reference[row["id"]] = "Y" if workspace_mask(spatial).getpixel(tuple(row["point"])) else "N"
            modes = {}
            box = boxes[row["id"]]
            for mode in conditions:
                text = LEGEND
                view_rgb, view_map = rgb, spatial
                question = row["question"]
                if mode == "with_auto_box":
                    if box is None:
                        modes[mode] = dict(raw=None, prediction="U")
                        continue
                    x0, y0, x1, y1 = [round(coord) for coord in box]
                    text += (f" The predicted target box in the {rgb.width} by {rgb.height} image is "
                             f"({x0}, {y0}, {x1}, {y1}) in xyxy pixel coordinates from the upper left.")
                elif targets or args.oracle or args.binary_crop or args.full_point_control:
                    points = targets.get((row["scene"], row["map_sha256"]))
                    view_rgb, view_map, location = oracle_view(row, mode, rgb, spatial, points)
                    text += location
                    if mode in ("oracle_crop", "binary_crop", "full_point_control", "shared_point_control"):
                        question = "Is the specified surface point in the robot's reachable space?"
                    if mode == "binary_crop":
                        view_map = workspace_mask(view_map).convert("RGB")
                        text = text.replace(
                            "A point is reachable if its map color is strongly saturated "
                            "(largest minus smallest RGB component is at least 83 on the 0–255 scale).",
                            "White pixels in the map are reachable and black pixels are unreachable.",
                        )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": view_rgb}, {"type": "image", "image": view_map},
                    {"type": "text", "text": f"{text}\n\n{question}"},
                ]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt").to(model.device)
                if args.answer_mode == "choice":
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    scores = [float(logits[ids[0]]) for ids in choice_ids]
                    label = choices[scores.index(max(scores))]
                    modes[mode] = dict(raw=label, prediction=label[0], yes_minus_no_logit=scores[0]-scores[1])
                    continue
                sequence = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                          temperature=None, top_p=None, top_k=None)
                suffix = sequence[0, inputs["input_ids"].shape[1]:]
                raw = processor.decode(suffix, skip_special_tokens=True).strip()
                answer = re.fullmatch(r"(yes|no)[.!]?", raw, re.IGNORECASE)
                modes[mode] = dict(raw=raw, prediction=answer[1][0].upper() if answer else "U")
            result = dict(id=row["id"], scene=row["scene"], truth=row["truth"], box=box, **modes)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            predictions.append(result)
            if (index + 1) % 16 == 0:
                print(f"{index+1}/{len(rows)}", flush=True)
    summaries = {}
    for mode in conditions:
        answers = {r["id"]: r[mode]["prediction"] for r in predictions}
        summaries[mode] = dict(**counts(rows, answers), versus_geometric=cluster_delta(rows, answers, reference))
    save(args.output / "metrics.json", dict(**summaries, geometric=counts(rows, reference),
         total_seconds=time.perf_counter()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "manifest", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    for field in ("candidates", "reference"):
        parser.add_argument(f"--{field}", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--answer-mode", choices=("generate", "choice"), default="generate")
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--oracle", action="store_true", help="Evaluate ground-truth box, point, and paired crops.")
    views.add_argument("--binary-crop", action="store_true", help="Repeat oracle crop with a binary reachability map.")
    views.add_argument("--full-point-control", action="store_true",
                       help="Repeat the direct point question with full images and full-frame coordinates.")
    views.add_argument("--shared-point-control", action="store_true",
                       help="Give two target queries identical cropped images for each workspace.")
    args = parser.parse_args()
    if not args.shared_point_control and (args.candidates is None or args.reference is None):
        parser.error("--candidates and --reference are required to replay the V1 autobox baseline.")
    evaluate(args)


if __name__ == "__main__":
    main()
