"""Predict target boxes from question text and RGB, with a cache shared across maps."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def target_phrase(question):
    prefix, suffix = "Is the ", " in the robot's reachable space?"
    if not question.startswith(prefix) or not question.endswith(suffix):
        raise ValueError(f"Unsupported reachability question: {question}")
    return question[len(prefix):-len(suffix)]


def detect_candidates(manifest_path, checkpoint, *, batch_size=8, query="full"):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if batch_size < 1:
        raise ValueError("Detector batch size must be positive")
    if query not in ("full", "final-word"):
        raise ValueError("Expected full or final-word detector query")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = {(r["image_sha256"], target_phrase(r["question"])): r["image"] for r in manifest["rows"]}
    tasks = list(examples.items())
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    predictions = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        images = []
        for (digest, _), name in batch:
            path = manifest_path.parent / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Unexpected image contents at {path}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        names = [name.lower() for (_, name), _ in batch]
        texts = [(name if query == "full" else name.split()[-1]) + "." for name in names]
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to("cuda")
        encoded = processor.tokenizer(texts, return_offsets_mapping=True, padding=True, return_tensors="pt")
        if not torch.equal(encoded.input_ids, inputs.input_ids.cpu()):
            raise RuntimeError("Cannot align detector language probabilities with the query text")
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=.12, text_threshold=.20,
            target_sizes=[(image.height, image.width) for image in images],
        )
        probabilities = outputs.logits.sigmoid().cpu()
        for index, (((digest, name), _), result) in enumerate(zip(batch, results, strict=True)):
            candidates = probabilities[index]
            active = candidates.amax(-1) > .12
            if not torch.allclose(candidates[active].amax(-1), result["scores"].cpu()):
                raise RuntimeError("Detector boxes and token confidence rows are not aligned")
            offsets = encoded.offset_mapping[index].tolist()
            words = [match.span() for match in re.finditer(r"[\w]+", texts[index])]
            spans = [[j for j, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
                     for start, end in words]
            spans = [span for span in spans if span]
            word_scores = torch.stack([candidates[active][:, span].mean(-1) for span in spans], dim=-1)
            boxes = [dict(box=box, score=score, all_words=float(word_values.min()), last_word=float(word_values[-1]))
                     for box, score, word_values in zip(result["boxes"].tolist(), result["scores"].tolist(),
                                                        word_scores, strict=True)]
            predictions.append(dict(image_sha256=digest, phrase=name, boxes=sorted(boxes, key=lambda b: -b["score"])))
        print(json.dumps(dict(processed=len(predictions), total=len(tasks))), flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query", choices=["full", "final-word"], default="full")
    args = parser.parse_args()
    predictions = detect_candidates(args.manifest, args.checkpoint, batch_size=args.batch_size, query=args.query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
