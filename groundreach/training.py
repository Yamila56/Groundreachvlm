"""Train or evaluate matched point and mean residual heads."""

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn

from groundreach.adapters.residual import ResidualProjection
from groundreach.evaluation.inference import evaluate, load_examples
from groundreach.evaluation.metrics import counts
from groundreach.io import save, sha256_file, verify_model_weights
from groundreach.models.frozen_vlm import FrozenVLM


def fit(engine, training, feature, seed, epochs):
    usable = [(row,x) for row,x in training if x is not None and row["truth"] in ("Y","N")]
    torch.manual_seed(seed)
    head = ResidualProjection(usable[0][1]["features"][feature].numel()).cuda()
    optimizer = torch.optim.AdamW(head.parameters(),lr=.001)
    order = random.Random(seed)
    for epoch in range(epochs):
        order.shuffle(usable)
        losses = []
        for row, example in usable:
            optimizer.zero_grad(set_to_none=True)
            token = head(example["features"][feature].to("cuda"))
            logits = engine.logits(example,token)
            label = torch.tensor([int(row["truth"]=="Y")], device="cuda")
            loss = nn.functional.cross_entropy(logits.unsqueeze(0),label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(feature,seed,epoch+1,len(usable),sum(losses)/len(losses),flush=True)
    if any(parameter.grad is not None for parameter in engine.model.parameters()):
        raise ValueError("Frozen model received gradients")
    return head


def run(args):
    verify_model_weights(args.model, args.model_sha256)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"protocol.json",dict(
        model_sha256=args.model_sha256, seed=args.seed, epochs=args.epochs, limit=args.limit,
        manifest_hashes={split:sha256_file(getattr(args,split)) for split in ("train","test")},
        heads_from=str(args.load_heads) if args.load_heads else None,
        point_source="automatic" if args.train_predictions else "oracle",
    ))
    start = time.perf_counter()
    engine = FrozenVLM(args.model)
    train = load_examples(args.train,engine,args.train_predictions,4 if args.load_heads else args.limit)
    test = load_examples(args.test,engine,args.test_predictions,args.limit)
    if {row["image_sha256"] for row,_ in train} & {row["image_sha256"] for row,_ in test}:
        raise ValueError("Train and test share an image")
    first = next(example for _,example in train if example is not None)
    with torch.no_grad():
        native = engine.logits(first,replay=False)
        cached = engine.logits(first)
        torch.testing.assert_close(native,cached,atol=0,rtol=0)
    predictions = {
        "coordinates_crop":evaluate(engine,test),
        "direct_point":evaluate(engine,test,feature="point"),
        "direct_mean":evaluate(engine,test,feature="mean"),
        "pixel_readout":{row["id"]:{"answer":x["pixel_answer"] if x else "U"} for row,x in test},
    }
    for feature in ("mean","point"):
        if args.load_heads:
            path = args.load_heads/f"{feature}.pt"
            head = ResidualProjection(first["features"][feature].numel()).cuda()
            head.load_state_dict(torch.load(path,weights_only=True))
        else:
            head = fit(engine,train,feature,args.seed,args.epochs)
            path = args.output/f"{feature}.pt"
            torch.save(head.state_dict(),path)
        predictions[f"adapted_{feature}"] = evaluate(engine,test,head,feature)
        if feature == "point" and not args.limit:
            predictions["adapted_point_swapped_map_token"] = evaluate(engine,test,head,feature,swap_map=True)
            predictions["adapted_point_swapped_target_token"] = evaluate(engine,test,head,feature,swap_target=True)
        restored = ResidualProjection(head.layers[0].in_features).cuda()
        restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():
            token = first["features"][feature].cuda()
            torch.testing.assert_close(head(token),restored(token),atol=0,rtol=0)
            torch.testing.assert_close(engine.logits(first,head(token)),engine.logits(first,restored(token)),atol=0,rtol=0)
    rows = [row for row,_ in test]
    metrics = {key:counts(rows,{i:r["answer"] for i,r in values.items()}) for key,values in predictions.items()}
    patterns = sorted({row["layout_pattern"] for row in rows if "layout_pattern" in row})
    if patterns:
        metrics["by_layout_pattern"] = {
            pattern:{key:counts([row for row in rows if row["layout_pattern"]==pattern],
                                {i:r["answer"] for i,r in values.items()})
                     for key,values in predictions.items()} for pattern in patterns}
    save(args.output/"predictions.json",predictions)
    save(args.output/"metrics.json",dict(**metrics, wall_seconds=time.perf_counter()-start,
                                         cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train","test","model","output"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    for name in ("train-predictions","test-predictions"):
        parser.add_argument(f"--{name}",type=Path)
    parser.add_argument("--model-sha256",required=True)
    parser.add_argument("--load-heads",type=Path)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args(argv)
    if bool(args.train_predictions)!=bool(args.test_predictions):
        parser.error("Provide both automatic-point prediction sets")
    run(args)
