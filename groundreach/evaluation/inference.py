"""Prepare aligned inputs and apply token interventions."""

from collections import defaultdict

import torch

from groundreach.io import image, read
from groundreach.vision.spatial_map import shared_point_crop, workspace_mask


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
