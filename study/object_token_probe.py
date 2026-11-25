"""Exercise geometry object tokens alongside the frozen PhysVLM global inputs."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnose_reachability import inputs
from evaluate_reachability import image, load_model
from grounding_readout import THRESHOLD, rectangle, region
from physvlm.constants import DEPTH_TOKEN_INDEX
from physvlm.mm_utils import tokenizer_image_token
from tune_reachability import configure, event


def features(spatial_map, candidate):
    if candidate is None:
        return [0.] * 12
    h, w = spatial_map.shape
    bbox = np.array(candidate["box"], dtype=float)
    large = rectangle(bbox, spatial_map.shape)
    small = rectangle(bbox, spatial_map.shape, .5)
    if large is None or small is None:
        return [0.] * 12
    x0, y0, x1, y1 = large
    values = region(spatial_map, large)
    center = region(spatial_map, small)
    return [
        (x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h,
        float(np.median(values))/255, float(np.median(center))/255,
        float((values >= THRESHOLD).mean()), float((center >= THRESHOLD).mean()),
        float(values.std())/255, float(center.std())/255, candidate["score"], 1.,
    ]


def load_examples(source, boxes):
    rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
    predictions = {(r["image_sha256"], r["phrase"]): r["boxes"]
                   for r in json.loads(boxes.read_text(encoding="utf-8"))}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene"], row["side"]].append(row)
    records = []
    for key, targets in grouped.items():
        if len(targets) != 2 or {r["truth"] for r in targets} != {"Y", "N"}:
            raise ValueError(f"Expected two named targets and one positive in {key}")
        first = targets[0]
        for field, expected in (("image", "image_sha256"), ("sp_map", "map_sha256")):
            if any(r[expected] != first[expected] for r in targets):
                raise ValueError(f"Modalities differed within {key}")
            if hashlib.sha256((source.parent/first[field]).read_bytes()).hexdigest() != first[expected]:
                raise ValueError(f"Hash mismatch for {key}/{field}")
        spatial_map = np.asarray(image(source.parent/first["sp_map"]))
        chroma = np.ptp(spatial_map, axis=-1)
        targets.sort(key=lambda row: row["target"])
        random.Random(key[0]).shuffle(targets)
        objects = []
        for row in targets:
            candidates = predictions[row["image_sha256"], row["target"]]
            objects.append(features(chroma, candidates[0] if candidates else None))
        records.append(dict(
            key=key, layout=first["layout"], rgb=source.parent/first["image"],
            spatial_map=source.parent/first["sp_map"], objects=objects,
            names=[r["target"] for r in targets], truths=[r["truth"] for r in targets],
        ))
    return records, {r["image_sha256"] for r in rows}


class ObjectProjection(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(12, 64), nn.GELU(), nn.Linear(64, hidden))
        self.slot = nn.Embedding(2, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values):
        slots = torch.arange(values.size(1), device=values.device)
        # Match the scale of the pretrained language embeddings.
        return .025*self.norm(self.projection(values) + self.slot(slots))


def answer_tokens(tokenizer, prefix, ids, alternatives):
    original = ids[0].tolist()
    tokens = [tokenizer_image_token(prefix + item, tokenizer) for item in alternatives]
    if any(t[:-1] != original for t in tokens):
        raise RuntimeError(f"Expected atomic slot and label answers: {alternatives}")
    return [t[-1] for t in tokens]


@torch.no_grad()
def encode_records(model, tokenizer, records):
    result = []
    for row in records:
        rgb, spatial_map = (image(row[name]) for name in ("rgb", "spatial_map"))
        names = f"Object 1 is the {row['names'][0]}. Object 2 is the {row['names'][1]}."
        tasks = [(f"Is object {i+1} reachable? Answer Yes or No.", ("Yes", "No"), int(truth == "N"))
                 for i, truth in enumerate(row["truths"])]
        tasks.append(("Which object is reachable? Answer 1 or 2.", ("1", "2"), row["truths"].index("Y")))
        encodings = []
        for query, alternatives, gold in tasks:
            prefix, ids, visual = inputs(model, tokenizer, rgb, spatial_map, f"{names}\n{query}", False)
            _, _, _, _, embedded, _ = model.prepare_inputs_labels_for_multimodal(
                ids, None, None, None, None, visual["images"], visual["depths"], visual["image_sizes"])
            depth = torch.where(ids[0] == DEPTH_TOKEN_INDEX)[0].tolist()
            if len(depth) != 1:
                raise ValueError("Expected one global map marker")
            after_depth = ids.size(1)-depth[0]-1
            split = embedded.size(1)-after_depth
            if not torch.equal(embedded[:, split:], model.get_input_embeddings()(ids[:, -after_depth:])):
                raise ValueError("Global fusion changed the text boundary")
            candidates = answer_tokens(tokenizer, prefix, ids, alternatives)
            encodings.append((embedded.cpu(), split, candidates, gold))
        result.append((row, encodings))
    return result


def logits(model, projection, row, encoding, enabled=True):
    embedded, split, candidates, target = encoding
    embedded = embedded.to(model.device)
    if enabled:
        values = torch.tensor([row["objects"]], dtype=torch.float32, device=model.device)
        encoded = projection(values).to(embedded.dtype)
        embedded = torch.cat((embedded[:, :split], encoded, embedded[:, split:]), dim=1)
    ids = torch.arange(embedded.size(1), device=model.device).unsqueeze(0)
    hidden = model.model(inputs_embeds=embedded, position_ids=ids, attention_mask=torch.ones_like(ids),
                         use_cache=False).last_hidden_state
    return model.lm_head(hidden[:, -1]).float(), candidates, target


@torch.no_grad()
def evaluate(model, projection, data, enabled=True, swap_objects=False):
    hits = [0, 0, 0]
    scenes = defaultdict(list)
    rule_scenes = defaultdict(list)
    program = [0, 0, 0]
    for row, encodings in data:
        selected = [item[5] >= THRESHOLD/255 and item[-1] for item in row["objects"]]
        choices = [int(not value) for value in selected] + [
            selected.index(True) if sum(selected) == 1 else -1]
        inputs_row = dict(row, objects=row["objects"][::-1]) if swap_objects else row
        for i, enc in enumerate(encodings):
            values, candidates, target = logits(model, projection, inputs_row, enc, enabled)
            hit = int(values[0, candidates].argmax()) == target
            hits[i] += hit
            program[i] += choices[i] == target
            if i < 2:
                scenes[row["key"][0]].append(hit)
                rule_scenes[row["key"][0]].append(choices[i] == target)
    if any(len(v) != 4 for v in scenes.values()):
        raise ValueError("Expected both maps and both objects per view")
    return dict(n_views=len(data), correct=hits, rule_correct=program, joint_four=sum(all(v) for v in scenes.values()),
                rule_joint_four=sum(all(v) for v in rule_scenes.values()),
                four_question_groups=len(scenes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "train", "train-boxes", "heldout", "heldout-boxes", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--mode", choices=("projection", "lora", "projection-lora"), default="projection")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--projection-lr", type=float, default=2e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--train-layouts", type=int, default=4)
    parser.add_argument("--heldout-layouts", type=int, default=4)
    args = parser.parse_args()
    training, ti = load_examples(args.train, args.train_boxes)
    heldout, hi = load_examples(args.heldout, args.heldout_boxes)
    if ti & hi:
        raise ValueError("Training and held-out RGB overlap")
    training = [r for r in training if r["layout"] in sorted({r["layout"] for r in training})[:args.train_layouts]]
    heldout = [r for r in heldout if r["layout"] in sorted({r["layout"] for r in heldout})[:args.heldout_layouts]]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, "cuda")
    model.requires_grad_(False)
    projection = ObjectProjection(model.config.hidden_size).to(model.device)
    train = encode_records(model, tokenizer, training)
    test = encode_records(model, tokenizer, heldout)
    baseline = dict(train=evaluate(model, projection, train, False), heldout=evaluate(model, projection, test, False))
    event(args.output, event="frozen", **baseline)
    use_projection = args.mode != "lora"
    train_lora = args.mode != "projection"
    language = configure(model, "lora", lora_rank=8) if train_lora else {}
    object_parameters = {n: p for n, p in projection.named_parameters()} if use_projection else {}
    parameters = {**language, **{"objects."+n: p for n, p in object_parameters.items()}}
    groups = [dict(params=list(items.values()), lr=lr) for items, lr in
              ((language, args.lora_lr), (object_parameters, args.projection_lr)) if items]
    optimizer = torch.optim.AdamW(groups)
    examples = [(row, item) for row, tasks in train for item in tasks]
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        losses = []
        for start in range(0, len(examples), 8):
            batch = examples[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            for row, item in batch:
                values, candidates, target = logits(model, projection, row, item, use_projection)
                gold = torch.tensor([candidates[target]], device=model.device)
                loss = F.cross_entropy(values, gold)
                (loss/len(batch)).backward()
                losses.append(float(loss.detach()))
            if any(not any(p.grad is not None and torch.count_nonzero(p.grad) for p in items.values())
                   for items in (language, object_parameters) if items):
                raise RuntimeError("A trainable module has no gradient")
            grad = torch.nn.utils.clip_grad_norm_(parameters.values(), 1., error_if_nonfinite=True)
            if not grad:
                raise RuntimeError("Zero object-token gradient")
            optimizer.step()
        event(args.output, event="epoch", epoch=epoch+1, loss=mean(losses), grad_norm=float(grad))
    weights = args.output/"adapter.pt"
    torch.save({name: p.detach().cpu() for name, p in parameters.items()}, weights)
    outcome = dict(train=evaluate(model, projection, train, use_projection),
                   heldout=evaluate(model, projection, test, use_projection),
                   n_parameters=sum(p.numel() for p in parameters.values()),
                   peak_mib=torch.cuda.max_memory_allocated()/2**20, mode=args.mode, seed=args.seed)
    if use_projection:
        outcome["heldout_swapped_tokens"] = evaluate(model, projection, test, swap_objects=True)
    with torch.no_grad():
        restored = torch.load(weights, map_location=model.device, weights_only=True)
        for name, p in parameters.items():
            p.zero_()
            p.copy_(restored[name])
    if evaluate(model, projection, test, use_projection) != outcome["heldout"]:
        raise RuntimeError("Reloaded predictions changed")
    event(args.output, event="complete", **outcome)
    (args.output/"complete.json").write_text(json.dumps(dict(baseline=baseline, projection=outcome), indent=2)+"\n")


if __name__ == "__main__":
    main()
