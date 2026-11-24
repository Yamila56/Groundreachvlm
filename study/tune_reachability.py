"""Tune PhysVLM's physical projector or language attention on matched scene pairs."""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from diagnose_reachability import candidates, inputs, predict
from evaluate_reachability import image, load_model, parse_label


def pairs(directory):
    manifest = json.loads((directory / "selection.json").read_text())
    return [(entry, directory / entry["fixture"]) for entry in manifest["selection"]]


def prepare(model, tokenizer, selected):
    for entry, path in selected:
        content = json.loads(path.read_text())
        target, truth = next(iter(content["pairs"].items()))
        rgb = image(path.parent / content["rgb"])
        question = f"Is the {target} in the robot's reachable space?"
        for side, gold in zip(("A", "B"), truth, strict=True):
            spatial_map = image(path.parent / content["configurations"][side]["sp_map"])
            yield entry, side, gold, rgb, spatial_map, question


class AttentionLoRA(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = 2.0

    def forward(self, x):
        residual = F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
        return self.base(x) + residual * self.scale


def configure(model, mode, *, lora_rank=8):
    model.requires_grad_(False)
    if mode == "projector":
        # Master weights remain FP32. CUDA autocast casts the linear ops to BF16.
        model.model.mm_depth_projector.float().requires_grad_(True)
    elif mode == "lora":
        for layer in model.model.layers:
            for key in ("q_proj", "v_proj"):
                setattr(layer.self_attn, key, AttentionLoRA(getattr(layer.self_attn, key), rank=lora_rank))
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def training_loss(model, tokenizer, example):
    _, _, gold, rgb, spatial_map, question = example
    prefix, ids, visual = inputs(model, tokenizer, rgb, spatial_map, question, True)
    answer = candidates(tokenizer, prefix, ids)[gold]
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    suffix = torch.tensor([answer + end], device=model.device)
    tokens = torch.cat((ids, suffix), dim=1)
    labels = torch.cat((torch.full_like(ids, -100), suffix), dim=1)
    _, position, attention, _, embedded, expanded = model.prepare_inputs_labels_for_multimodal(
        tokens, None, None, None, labels, visual["images"], visual["depths"], visual["image_sizes"])
    n = suffix.shape[-1]
    if expanded.shape[1] != embedded.shape[1] or not torch.equal(expanded[:, -n:], suffix):
        raise ValueError("Labels lost alignment during visual-token fusion")
    hidden = model.model(inputs_embeds=embedded, attention_mask=attention,
                         position_ids=position, use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, -n-1:-1]).float()
    return F.cross_entropy(logits.flatten(0, 1), suffix.flatten())


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def validate(model, tokenizer, selected):
    both = {"flip": [], "stable": []}
    outcomes = []
    crossed = defaultdict(list)
    for entry, side, gold, rgb, spatial_map, question in prepare(model, tokenizer, selected):
        _, ids, visual = inputs(model, tokenizer, rgb, spatial_map, question, True)
        generated = model.generate(ids, images=visual["images"], depth_images=visual["depths"],
                                   image_sizes=visual["image_sizes"], do_sample=False, max_new_tokens=16,
                                   temperature=None, top_k=None, top_p=None, use_cache=True)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        outcomes.append(parse_label(decoded) == gold)
        if side == "B":
            group = "stable" if entry["label"] in ("YY", "NN") else "flip"
            both[group].append(int(all(outcomes)))
            if "crossed_scene" in entry:
                crossed[entry["crossed_scene"]].append(all(outcomes))
            outcomes.clear()
    if crossed:
        if any(len(scene) != 2 for scene in crossed.values()):
            raise ValueError("Expected two labeled targets in each validation view")
        both["crossed"] = [int(all(scene)) for scene in crossed.values()]
    return {key: (sum(values), len(values)) for key, values in both.items()}


def state_hash(params):
    value = hashlib.sha256()
    for name, tensor in params:
        value.update(name.encode())
        value.update(tensor.detach().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def event(output, **fields):
    line = json.dumps(fields)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def tune(args, model, tokenizer, params):
    selected = pairs(args.train)
    dev = pairs(args.dev)
    if args.probe:
        selected = [row for label in ("YN", "NY", "YY", "NN")
                    for row in [p for p in selected if p[0]["label"] == label][:2]]
        dev = selected
    frozen = [(n, p.flatten()[:128]) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_before = state_hash(frozen)
    starting = state_hash(params.items())
    event(args.output, event="parameters", count=sum(p.numel() for p in params.values()), names=list(params))
    optimizer = torch.optim.AdamW(params.values(), lr=args.learning_rate, weight_decay=.01)
    total_steps = args.epochs * ((2 * len(selected) + 7) // 8)
    warmup = max(1, round(total_steps * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        (s + 1) / warmup if s < warmup else max(0, (total_steps - s) / max(1, total_steps - warmup)))
    progress, best = 0, -1.
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ordered = random.Random(args.seed + epoch).sample(selected, len(selected))
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for i, example in enumerate(prepare(model, tokenizer, ordered), 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = training_loss(model, tokenizer, example)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PhysVLM loss")
            (loss / min(8, 2 * len(ordered) - ((i - 1) // 8) * 8)).backward()
            losses.append(float(loss.detach()))
            if i % 8 == 0 or i == 2 * len(ordered):
                grad = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
                if progress == 0 and float(grad) == 0:
                    raise RuntimeError("First training update has zero gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress += 1
                event(args.output, event="update", epoch=epoch, step=progress, grad_norm=float(grad),
                      loss=sum(losses[-min(i, 8):])/min(i, 8), seconds=time.perf_counter()-train_start,
                      peak_mib=torch.cuda.max_memory_allocated()/2**20)
        val = validate(model, tokenizer, dev)
        if "crossed" in val:
            correct, total = val["crossed"]
            quality = correct / total
        else:
            rates = [a / b for a, b in val.values() if b]
            quality = sum(rates) / len(rates)
        checkpoint = dict(mode=args.mode, seed=args.seed, epoch=epoch, lora_rank=args.lora_rank,
                          params={n: p.detach().cpu() for n, p in params.items()})
        if quality > best:
            best = quality
            torch.save(checkpoint, args.output / "best.tmp")
            (args.output / "best.tmp").replace(args.output / "best.pt")
        torch.save(dict(checkpoint=checkpoint, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), args.output / "latest.tmp")
        (args.output / "latest.tmp").replace(args.output / "latest.pt")
        event(args.output, event="epoch", epoch=epoch, validation=val, score=quality,
              mean_train_loss=sum(losses)/len(losses), seconds=time.perf_counter()-train_start)
    if starting == state_hash(params.items()) or frozen_before != state_hash(frozen):
        raise RuntimeError("Trainable/frozen parameter update audit failed")
    event(args.output, event="complete", seconds=time.perf_counter()-train_start, updates=progress)


def evaluate(args, model, tokenizer):
    output = args.output / "predictions.jsonl"
    start = time.perf_counter()
    reuse_features = True
    with output.open("x") as stream:
        for entry, side, gold, rgb, spatial_map, question in prepare(model, tokenizer, pairs(args.test)):
            box = None
            if args.roi:
                from audit_matched_pairs import COLOR_INDEX, locate
                import numpy as np
                pixels = np.asarray(rgb)[:, :, ::-1]
                name = entry["color"]
                if args.roi == "wrong":
                    alternatives = set(COLOR_INDEX) | {"yellow block"}
                    alternatives.remove(name)
                    name = max(sorted(alternatives), key=lambda key: int(locate(pixels, key).sum()))
                ys, xs = np.where(locate(pixels, name))
                if len(xs):
                    edge = 3 * max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
                    left = int(xs.min() + xs.max() - edge) // 2
                    top = int(ys.min() + ys.max() - edge) // 2
                    box = (left, top, left + edge, top + edge)
                    rgb, spatial_map = rgb.crop(box), spatial_map.crop(box)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if stream.tell() == 0:
                    reference = predict(model, tokenizer, rgb, spatial_map, question, True)
                    candidate = predict(model, tokenizer, rgb, spatial_map, question, True, reuse_features=True)
                    reuse_features = candidate["generated_ids"] == reference["generated_ids"] and all(
                        abs(candidate[field] - reference[field]) < .05 for field in ("margin", "marker_margin"))
                    event(args.output, event="evaluation_feature_reuse", enabled=reuse_features)
                    outcome = candidate if reuse_features else reference
                else:
                    outcome = predict(model, tokenizer, rgb, spatial_map, question, True,
                                      reuse_features=reuse_features)
            row = dict(scene=entry["scene"], family=entry["family"], block=entry["block"],
                       side=side, gold=gold, pair_label=entry["label"], roi=box, **outcome)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    event(args.output, event="complete", seconds=time.perf_counter()-start,
          peak_mib=torch.cuda.max_memory_allocated()/2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("projector", "lora", "frozen"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--roi", choices=("target", "wrong"))
    args = parser.parse_args()
    if args.lora_rank < 1 or not (0 < args.learning_rate < 1):
        parser.error("Use a positive rank and a learning rate between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, "cuda:0")
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu") if args.checkpoint else None
    rank = payload.get("lora_rank", 8) if payload else args.lora_rank
    params = configure(model, args.mode, lora_rank=rank)
    if payload:
        if payload["mode"] != args.mode or payload["params"].keys() != params.keys():
            raise ValueError("Checkpoint trainable parameters do not match")
        with torch.no_grad():
            for name, value in payload["params"].items():
                params[name].copy_(value)
    if args.test:
        evaluate(args, model, tokenizer)
    else:
        if not args.train or not args.dev or not params:
            parser.error("Training requires train and dev paths and a trainable mode")
        tune(args, model, tokenizer, params)


if __name__ == "__main__":
    main()
