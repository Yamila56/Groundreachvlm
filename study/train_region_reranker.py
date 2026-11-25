"""Fit a residual candidate ranker with RGB, language, and target boxes."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_grounding_candidates import reference_boxes
from grounding_readout import iou, save
from prepare_official_box_probe import PREFIX, SUFFIX, normalized


def features(item, variant="full"):
    boxes = item["boxes"]
    visual, text = item["visual"], item["language"].expand(len(boxes), -1)
    similarity = torch.tensor([[box["score"], box["semantic"]] for box in boxes])
    base = ((similarity-similarity.mean(0))/similarity.std(0, unbiased=False).clamp_min(1e-6)).mean(1)
    inputs = (visual, text) if variant == "no-interaction" else (visual, visual*text*visual.shape[-1]**.5, text)
    return torch.cat(inputs, dim=1)*visual.shape[-1]**.5, base


def load_labeled(manifest, cache, variant="full"):
    rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
    source = dict(root=manifest.parent, by_image=defaultdict(list))
    annotation = {}
    for row in rows:
        if "box" in row and row["side"] == "A":
            source["by_image"][row["image_sha256"]].append(row)
        phrase = row["question"][len(PREFIX):-len(SUFFIX)]
        annotation[row["image_sha256"], phrase] = row
    items = torch.load(cache, map_location="cpu", weights_only=True)
    indexed = {(r["image_sha256"], r["phrase"]): r for r in items}
    if indexed.keys() != annotation.keys() or len(indexed) != len(items):
        raise ValueError(f"Candidate cache and annotations disagree for {manifest}")
    pairs = []
    for key, row in annotation.items():
        item = indexed[key]
        width, height = item["size"]
        refs = reference_boxes(source, row, height, width)
        match = [name for name in refs if (name if "box" in row else normalized(name)) == key[1]]
        positives = [iou(np.array(c["box"]), refs[match[0]]) >= .5 for c in item["boxes"]] if len(match) == 1 else []
        if not positives:
            continue
        x, base = features(item, variant)
        pairs.append((x, base, torch.tensor(positives)))
    return pairs, {r["image_sha256"] for r in rows}


def pack(pairs):
    width = max(len(x) for x, _, _ in pairs)
    x = torch.zeros(len(pairs), width, pairs[0][0].shape[1])
    base = torch.full((len(pairs), width), -torch.inf)
    positive = torch.zeros_like(base, dtype=torch.bool)
    for i, (values, prior, mask) in enumerate(pairs):
        x[i, :len(values)] = values
        base[i, :len(values)] = prior
        positive[i, :len(values)] = mask
    return x, base, positive


def network(dimensions, variant="full"):
    if variant == "linear":
        return nn.Sequential(nn.Linear(dimensions, 1))
    return nn.Sequential(nn.Linear(dimensions, 32), nn.GELU(), nn.Dropout(.15), nn.Linear(32, 1))


def validation(model, data):
    x, base, m = data
    model.eval()
    with torch.no_grad():
        scores = base + model(x).squeeze(-1)
    return int(m.gather(1, scores.argmax(1, keepdim=True)).sum())


def train(args):
    train_pairs, dev_pairs, train_images, dev_images = [], [], set(), set()
    for specs, collection, images in ((args.train, train_pairs, train_images), (args.dev, dev_pairs, dev_images)):
        for manifest, cache in specs:
            pairs, identities = load_labeled(Path(manifest), Path(cache), args.variant)
            collection.extend(pairs)
            images.update(identities)
    if train_images & dev_images:
        raise ValueError("Training and development images overlap")
    informative = [p for p in train_pairs if p[2].any() and not p[2].all()]
    x, base, positive = pack(informative)
    dev = pack([p for p in dev_pairs if p[2].any()])
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = network(x.shape[-1], args.variant)
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    history = [dict(epoch=0, correct=validation(model, dev))]
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_step = 0
    for epoch in range(1, 9):
        model.train()
        for ids in torch.randperm(len(x), generator=generator).split(32):
            scores = base[ids] + model(x[ids]).squeeze(-1)
            loss = (scores.logsumexp(1) - scores.masked_fill(~positive[ids], -torch.inf).logsumexp(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        hits = validation(model, dev)
        if hits > history[best_step]["correct"]:
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_step = epoch
        history.append(dict(epoch=epoch, correct=hits))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(model=best, dimensions=x.shape[-1], variant=args.variant), args.output/"weights.pt")
    save(args.output/"training.json", dict(seed=args.seed, variant=args.variant, train_groups=len(train_pairs),
         informative_groups=len(informative), dev_groups_with_positive=len(dev[0]),
         selected_epoch=best_step, dev_history=history,
         train_images=sorted(train_images), dev_images=sorted(dev_images)))


@torch.inference_mode()
def apply(args):
    weights = torch.load(args.checkpoint/"weights.pt", map_location="cpu", weights_only=True)
    provenance = json.loads((args.checkpoint/"training.json").read_text(encoding="utf-8"))
    variant = weights.get("variant", "full")
    model = network(weights["dimensions"], variant)
    model.load_state_dict(weights["model"])
    model.eval()
    output = []
    known = set(provenance["train_images"]) | set(provenance["dev_images"])
    for row in torch.load(args.cache, map_location="cpu", weights_only=True):
        if row["image_sha256"] in known:
            raise ValueError("Held-out RGB overlaps the ranker training or development split")
        boxes = row["boxes"]
        if boxes:
            x, base = features(row, variant)
            best = int((base+model(x).squeeze(-1)).argmax())
            boxes = [boxes[best]]
        output.append(dict(image_sha256=row["image_sha256"], phrase=row["phrase"], boxes=boxes))
    save(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--train", nargs=2, action="append", required=True, metavar=("MANIFEST", "CACHE"))
    fit.add_argument("--dev", nargs=2, action="append", required=True, metavar=("MANIFEST", "CACHE"))
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--variant", choices=("full", "linear", "no-interaction"), default="full")
    predict = commands.add_parser("apply")
    for name in ("checkpoint", "cache", "output"):
        predict.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    (train if args.mode == "train" else apply)(args)


if __name__ == "__main__":
    main()
