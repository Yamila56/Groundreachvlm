"""Counterfactually replace the residual ranker's text embedding with its same-image alternative."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from groundreach.grounding.detector import target_phrase
from experiments.relational_grounding.evaluate_marked_points import checked_image, read, write
from groundreach.grounding.markers import find_markers, locate_marker
from experiments.relational_grounding.train_region_reranker import features, network


@torch.inference_mode()
def audit(manifest, cache, checkpoint):
    folder = manifest.parent
    rows = {(r["image_sha256"], target_phrase(r["question"])): r
            for r in read(manifest)["rows"] if r["side"] == "A"}
    decoded = torch.load(cache, map_location="cpu", weights_only=True)
    images = defaultdict(list)
    for item in decoded:
        if (item["image_sha256"], item["phrase"]) in rows:
            images[item["image_sha256"]].append(item)
    if len(rows) != sum(len(items) for items in images.values()) or any(len(items) != 2 for items in images.values()):
        raise ValueError("The cache needs both unique query embeddings for each image")
    weight = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    ranked = network(weight["dimensions"], weight.get("variant", "full")).eval()
    ranked.load_state_dict(weight["model"])
    per_item = []
    for items in images.values():
        for primary, alternate in (items, items[::-1]):
            row = rows[primary["image_sha256"], primary["phrase"]]
            if not primary["boxes"] or alternate["language"] is None:
                per_item.append(dict(id=row["id"], eligible=False, candidates=len(primary["boxes"])))
                continue
            original_x, fixed = features(primary, weight.get("variant", "full"))
            replacement_x, check = features(dict(primary, language=alternate["language"]), weight.get("variant", "full"))
            torch.testing.assert_close(fixed, check, rtol=0, atol=0)
            original = ranked(original_x).flatten() + fixed
            replacement = ranked(replacement_x).flatten() + fixed
            shifts = (replacement - original).tolist()
            original_order = torch.argsort(original, descending=True, stable=True)
            replacement_order = torch.argsort(replacement, descending=True, stable=True)
            markers = find_markers(checked_image(folder / row["marked_image"], row["marked_sha256"]))
            correct = torch.tensor([locate_marker(markers, box["box"]) == tuple(row["point"]) for box in primary["boxes"]])
            margins = ([float(scores[correct].max() - scores[~correct].max()) for scores in (original, replacement)]
                       if correct.any() and (~correct).any() else None)
            per_item.append(dict(
                id=row["id"], eligible=True, candidates=len(shifts),
                original_order=original_order.tolist(), replacement_order=replacement_order.tolist(),
                top1_changed=original_order[0].item() != replacement_order[0].item(),
                rank_changed=not torch.equal(original_order, replacement_order),
                original_correct=bool(correct[original_order[0]]), replacement_correct=bool(correct[replacement_order[0]]),
                scores_shift=shifts, shift_range=float(max(shifts) - min(shifts)), margins=margins,
            ))
    eligible = [entry for entry in per_item if entry["eligible"]]
    comparable = [entry for entry in eligible if entry["candidates"] > 1]
    margins = [entry["margins"] for entry in comparable if entry["margins"] is not None]
    return dict(
        queries=len(per_item), eligible=len(eligible), several_candidates=len(comparable),
        top1_changed=sum(r["top1_changed"] for r in eligible), rank_changed=sum(r["rank_changed"] for r in eligible),
        original_correct=sum(r["original_correct"] for r in eligible),
        replacement_correct=sum(r["replacement_correct"] for r in eligible),
        shift_range_quantiles=np.quantile([r["shift_range"] for r in comparable], [.1, .5, .9]).tolist()
        if comparable else [],
        margin_n=len(margins),
        original_margin_median=float(np.median([a for a, _ in margins])) if margins else None,
        replacement_margin_median=float(np.median([b for _, b in margins])) if margins else None,
        margin_shift_median=float(np.median([b - a for a, b in margins])) if margins else None,
        predictions=per_item,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "cache", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    write(args.output, audit(args.manifest, args.cache, args.checkpoint))
