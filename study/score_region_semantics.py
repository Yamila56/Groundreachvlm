"""Score detector candidates from RGB crops and the requested phrase using frozen CLIP."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "candidates", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--confidence", type=float, default=.12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    files = {r["image_sha256"]: r["image"] for r in manifest}
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    model = CLIPModel.from_pretrained(args.checkpoint, local_files_only=True).cuda().eval()
    processor = CLIPProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    start = time.perf_counter()
    results = []
    features = []
    for row in candidates:
        with Image.open(args.manifest.parent/files[row["image_sha256"]]) as source:
            rgb = source.convert("RGB")
        boxes, crops = [], []
        for candidate in sorted(row["boxes"], key=lambda box: -box["score"]):
            if candidate["score"] < args.confidence:
                continue
            x0, y0, x1, y1 = np.clip(candidate["box"], 0, [rgb.width, rgb.height]*2)
            if x1-x0 < 2 or y1-y0 < 2:
                continue
            current = np.array([x0, y0, x1, y1])
            if any(overlap(current, crop_box) > .7 for crop_box, _ in boxes):
                continue
            boxes.append((current, candidate))
            crops.append(rgb.crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))))
            if len(boxes) >= args.top_k:
                break
        scored = []
        if crops:
            text = processor(text=[f"a photo of the {row['phrase']}"], return_tensors="pt").to("cuda")
            pixels = processor(images=crops, return_tensors="pt").to("cuda")
            visual = normalize(model.get_image_features(**pixels).float(), dim=-1)
            language = normalize(model.get_text_features(**text).float(), dim=-1)
            scores = (visual @ language.T).flatten().tolist()
            scored = [dict(candidate, semantic=score) for (_, candidate), score in zip(boxes, scores, strict=True)]
        results.append(dict(row, boxes=scored))
        if args.features:
            features.append(dict(image_sha256=row["image_sha256"], phrase=row["phrase"], boxes=scored,
                                 visual=visual.cpu() if crops else None,
                                 language=language[0].cpu() if crops else None, size=list(rgb.size)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.features:
        args.features.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, args.features)
    print(json.dumps(dict(queries=len(results), seconds=time.perf_counter()-start)), flush=True)


def overlap(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return intersection / max(1, np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)


if __name__ == "__main__":
    main()
