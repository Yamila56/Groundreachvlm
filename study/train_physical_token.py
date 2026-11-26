"""Compare point and mean map tokens at the frozen Qwen3-VL language interface."""

import argparse
import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_frozen_vlm import LEGEND, image, read, save, shared_point_crop, workspace_mask
from replay_marked_points import counts

QUESTION = "Is the specified surface point in the robot's reachable space?"


class ResidualProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return value + self.layers(value)


class FrozenVLM:
    def __init__(self, path):
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True,
        ).to("cuda").eval().requires_grad_(False)
        tokenizer = self.processor.tokenizer
        choices = [tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes", ".")]
        if any(len(choice) != 1 for choice in choices):
            raise ValueError("Expected single tokens for No, Yes, and the injected slot")
        self.no, self.yes, self.slot = (item[0] for item in choices)

    @torch.no_grad()
    def prepare(self, rgb, spatial, x, y, keep_pixels=False):
        w, h = spatial.size
        text = (f"{LEGEND} The target surface point in image 1 is at (x={x}, y={y}) "
                f"in the {w} by {h} pixel image, from the upper left.\n\n{QUESTION}")
        messages = [{"role":"user","content":[{"type":"image","image":rgb},
                                                {"type":"image","image":spatial},{"type":"text","text":text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to("cuda")
        vision = self.model.visual(inputs.pixel_values.to(self.model.visual.dtype), grid_thw=inputs.image_grid_thw)
        merge = self.model.visual.spatial_merge_size
        _, rows, cols = inputs.image_grid_thw[-1].tolist()
        rows, cols = rows//merge, cols//merge
        grid = vision[0][-rows*cols:].reshape(rows,cols,-1).float()
        row, col = int((y+.5)*rows/h), int((x+.5)*cols/w)
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("Point fell outside its merged visual grid")
        features = {"point":grid[row,col].cpu(),"mean":grid.mean((0,1)).cpu()}
        pixels = inputs.pop("pixel_values")
        example = {
            "inputs":inputs.to("cpu"),
            "vision":(vision[0].cpu(), [value.cpu() for value in vision[1]]),
            "features":features,
            "index":[row,col],
        }
        if keep_pixels:
            example["pixels"] = pixels.cpu()
        return example

    def logits(self, example, token=None, *, replay=True):
        inputs = {key:value.to("cuda") for key,value in example["inputs"].items()}
        if token is not None:
            original_ids = inputs["input_ids"]
            inputs["input_ids"] = torch.cat((original_ids, original_ids.new_full((1,1), self.slot)),dim=1)
            mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat((mask, mask.new_ones((1,1))),dim=1)
        inputs["pixel_values"] = torch.empty(0, device="cuda")
        if not replay:
            inputs["pixel_values"] = example["pixels"].to("cuda")
        callbacks = []
        if replay:
            visual, deep = example["vision"]
        original_forward = self.model.visual.forward
        if replay:
            self.model.visual.forward = lambda *args, **kwargs: (visual.to("cuda"), [x.to("cuda") for x in deep])
        if token is not None:
            def inject(_, ids, values):
                if ids[0][0,-1].item() != self.slot:
                    raise ValueError("The physical token is outside the designated slot")
                return torch.cat((values[:,:-1], token.to(values.dtype).view(1,1,-1)),dim=1)
            callbacks.append(self.model.get_input_embeddings().register_forward_hook(inject))
        try:
            logits = self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        finally:
            if replay:
                self.model.visual.forward = original_forward
            for callback in callbacks:
                callback.remove()
        return logits[[self.no,self.yes]]


def load_examples(manifest, engine, prediction_file=None, limit=None):
    rows = read(manifest)["rows"]
    if limit:
        rows = rows[:limit]
    if prediction_file:
        positions = {r["id"]:r["methods"]["unmarked_appearance"]["marker"] for r in read(prediction_file)}
    else:
        positions = {r["id"]:r["point"] for r in rows}
    grouped = defaultdict(dict)
    for row in rows:
        point = positions[row["id"]]
        if point is not None:
            grouped[row["scene"]][row["target"]] = point
    examples = []
    for index,row in enumerate(rows):
        point = positions[row["id"]]
        if point is None:
            examples.append((row,None))
            continue
        rgb = image(manifest.parent/row["marked_image"], row["marked_sha256"])
        spatial = image(manifest.parent/row["sp_map"], row["map_sha256"])
        if rgb.size != spatial.size:
            raise ValueError("Map and RGB dimensions differ")
        region = shared_point_crop(list(grouped[row["scene"]].values()), *rgb.size)
        x,y = point[0]-region[0], point[1]-region[1]
        prepared = engine.prepare(rgb.crop(region), spatial.crop(region), x, y,
                                  keep_pixels=not any(x is not None for _,x in examples))
        prepared["pixel_answer"] = "Y" if workspace_mask(spatial).getpixel(tuple(point)) else "N"
        examples.append((row,prepared))
        if (index+1)%64==0:
            print(manifest.parent.name,index+1,flush=True)
    return examples


def evaluate(engine, examples, head=None, feature=None, *, swap_map=False, swap_target=False):
    result = {}
    opposite = {}
    if swap_map or swap_target:
        for row, example in examples:
            key = row["scene"], row["target"] if swap_map else row["side"]
            opposite.setdefault(key, []).append((row["id"],example))
    with torch.no_grad():
        for row, example in examples:
            if example is None:
                result[row["id"]] = {"answer":"U"}
                continue
            source = example
            if opposite:
                key = row["scene"], row["target"] if swap_map else row["side"]
                source = next((x for ident,x in opposite[key] if ident!=row["id"]), None)
                if source is None:
                    result[row["id"]] = {"answer":"U"}
                    continue
            token = source["features"][feature].to("cuda") if feature else None
            if head is not None:
                token = head(token)
            logits = engine.logits(example,token)
            result[row["id"]] = {"answer": "Y" if logits[1]>logits[0] else "N",
                                 "margin":float(logits[1]-logits[0]), "grid_index":example["index"]}
    return result


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
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"protocol.json",dict(
        model_sha256=args.model_sha256, seed=args.seed, epochs=args.epochs, limit=args.limit,
        manifest_hashes={split:hashlib.sha256(getattr(args,split).read_bytes()).hexdigest() for split in ("train","test")},
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


if __name__ == "__main__":
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
    args=parser.parse_args()
    if bool(args.train_predictions)!=bool(args.test_predictions):
        parser.error("Provide both automatic-point prediction sets")
    run(args)
