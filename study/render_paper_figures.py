"""Build publication figures from simulation renders and logged model answers."""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image

from summarize_matched import read

PAPER = "#ffffff"
INK = "#172431"
MUTED = "#4d5b69"
RULE = "#dae1e8"
SOFT = "#f1f5f9"
MINT = "#e0f5e9"
PINK = "#fce6ef"
SKY = "#e5f1fd"
NAVY = "#3c5488"
GREEN_TEXT = "#006c5a"
RED_TEXT = "#aa3124"
BODY = ["Calibri", "DejaVu Sans"]
DISPLAY = ["Segoe UI", "DejaVu Sans"]

plt.rcParams.update({"font.family": BODY, "figure.facecolor": PAPER, "axes.facecolor": PAPER})


def image(path):
    with Image.open(path) as scene:
        return np.asarray(scene.convert("RGB"))


def text(fig, x, y, label, size=10, weight="normal", color=INK, **kwargs):
    if weight == "bold":
        weight = "semibold"
    fig.text(x, y, label, fontsize=size, fontweight=weight, color=color, va="center", **kwargs)


def rule(fig, x0, x1, y):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, lw=.8, color=RULE))


def title(fig, heading, detail):
    text(fig, .035, .948, heading, size=18, weight="medium", fontfamily=DISPLAY)
    text(fig, .035, .948 - .31 / fig.get_figheight(), detail, size=9.8, color=MUTED)


def picture(fig, rect, rgb, pixel=None, bbox=None, zoom=False):
    ax = fig.add_axes(rect)
    ax.imshow(rgb)
    if zoom:
        ax.set_xlim(52, 590)
        ax.set_ylim(360, 122)
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(.7)
    if pixel is not None:
        for color, width in (("#14191d", 2.8), ("white", 1.45)):
            ax.add_patch(Circle(pixel, radius=12, fill=False, edgecolor=color, lw=width))
    if bbox is not None:
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = bbox
        for color, width in (("#12191f", 2.55), ("white", 1.15)):
            ax.add_patch(Rectangle((x0*w, y0*h), (x1-x0)*w, (y1-y0)*h,
                                   fill=False, edgecolor=color, lw=width))


def answer(label):
    return {"Y": "Yes", "N": "No", "U": "Unclear"}[label]


def group_label(fig, x, y, panel, label):
    text(fig, x, y, panel, size=9, weight="bold", color="white", ha="center",
         bbox=dict(boxstyle="round,pad=.32", facecolor=NAVY, edgecolor="none"))
    text(fig, x + .022, y, label, size=10.5, weight="bold")


def save(fig, destination):
    fig.savefig(destination, dpi=224, facecolor=PAPER)
    plt.close(fig)


def panel(fig, x, y, w, h, fill, accent=RULE):
    fig.add_artist(Rectangle((x,y), w, h, transform=fig.transFigure,
                             facecolor=fill, edgecolor=accent, lw=.75, zorder=0))


def arrow(fig, x0, y0, x1, y1):
    from matplotlib.patches import FancyArrowPatch
    fig.add_artist(FancyArrowPatch((x0,y0), (x1,y1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1, zorder=3))


def workflow(root, evaluation, selected_path, destination):
    selection = json.loads((root / "selection.json").read_text())["selection"]
    key = json.loads(selected_path.read_text())["correct-flip"]
    path = root / next(row["fixture"] for row in selection if row["scene"] == key)
    fixture = json.loads(path.read_text())
    target = next(iter(fixture["pairs"]))
    pixel = fixture["targets"][target]["pixel_xy"]
    tuned = read(evaluation / "projector17/predictions.jsonl")[key]
    frozen = read(evaluation / "frozen/predictions.jsonl")[key]
    fig = plt.figure(figsize=(10.0, 5.1))
    title(fig, "Do reachability answers follow the workspace?",
          "Counterfactual evaluation and lightweight adaptation with PhysVLM-3B")
    for x,w,fill,label in ((.033,.300,SKY,"A   Construct paired inputs"),
                           (.350,.302,PINK,"B   Adapt one model component"),
                           (.669,.298,MINT,"C   Test the map intervention")):
        panel(fig, x,.148,w,.675, PAPER)
        panel(fig,x,.751,w,.072,fill)
        text(fig,x+.013,.787,label,size=10.4,weight="medium",color=NAVY)
    # Put the fixed image alongside the two changing conditions.
    picture(fig, [.054,.571,.255,.156], image(path.parent/fixture["rgb"]), pixel=pixel, zoom=True)
    text(fig,.181,.732,"Shared RGB and target",size=8.8,weight="medium",ha="center")
    names = [fixture["configurations"][side]["sp_map"] for side in ("A","B")]
    for x, label, name in zip((.053,.184), ("Map A","Map B"), names, strict=True):
        picture(fig,[x,.396,.126,.130],image(path.parent/name),pixel=pixel,zoom=True)
        text(fig,x+.063,.545,label,size=8.7,weight="bold",ha="center")
    text(fig,.050,.332,f"Question target   {target}",size=8.6,weight="bold")
    text(fig,.050,.245,"Move the simulated Panda base.\nKeep camera, target and question fixed.",
         color=MUTED,size=8.5)
    arrow(fig,.333,.494,.350,.494)
    for x,label in ((.369,"RGB → SigLIP"),(.505,"Map → SigLIP")):
        panel(fig,x,.596,.128,.112,SKY)
        text(fig,x+.064,.665,label,size=8.45,ha="center")
        text(fig,x+.064,.627,"frozen",size=7.4,ha="center",color=MUTED)
        arrow(fig,x+.064,.596,.501,.486)
    panel(fig,.369,.397,.264,.084,PINK)
    text(fig,.501,.440,"Fuse RGB + S-P Map features",size=8.65,ha="center")
    arrow(fig,.501,.396,.501,.354)
    panel(fig,.369,.267,.264,.081,MINT)
    text(fig,.501,.308,"Tune the physical projector",size=8.65,ha="center",weight="bold",color=GREEN_TEXT)
    arrow(fig,.501,.266,.501,.239)
    panel(fig,.369,.168,.264,.068,"white")
    text(fig,.501,.203,"Question + visual tokens → frozen Qwen2.5",size=7.6,color=MUTED,ha="center")
    arrow(fig,.652,.494,.669,.494)
    for j,(label, pair, field, fill) in enumerate((
        ("IK/FK or URDF certificate", tuned, "gold", "white"),
        ("Frozen checkpoint", frozen, "free", SKY),
        ("Tuned projector", tuned, "free", "white"),
    )):
        y=.651-j*.109
        panel(fig,.687,y,.261,.075,fill)
        text(fig,.696,y+.037,label,size=8.15)
        text(fig,.938,y+.037," / ".join(answer(r[field]) for r in pair),
             size=8.7,weight="bold",ha="right",color=GREEN_TEXT if j==2 else INK)
    text(fig,.689,.329,"Repeat with A/B maps exchanged.",size=8.5,weight="bold")
    text(fig,.689,.237,"160 held-out pairs · 8 layout families\nCompare frozen, projector and LLM LoRA.",
         color=MUTED,size=8.05)
    text(fig,.034,.062,"Example " + key + " · The certificates are computed separately from the model.",
         size=7.9,color=MUTED)
    save(fig,destination)


def gallery(root, evaluation, selected_path, destination):
    pairs = json.loads((root / "selection.json").read_text())["selection"]
    by_id = {entry["scene"]: entry for entry in pairs}
    chosen = json.loads(selected_path.read_text())
    frozen = read(evaluation / "frozen/predictions.jsonl")
    tuned = read(evaluation / "projector17/predictions.jsonl")
    fig = plt.figure(figsize=(9.2, 6.65))
    title(fig, "Same RGB. Different workspace.",
          "Franka Panda simulation · Three held-out scenes · Rings mark the target surface point.")
    for x, heading in zip((.035, .260, .485, .719), ("Shared RGB", "S-P Map A", "S-P Map B", "Free answers"), strict=True):
        text(fig, x, .837, heading, size=9, weight="bold", color=NAVY)
    text(fig, .886, .837, "A", size=9, weight="bold", ha="center", color=NAVY)
    text(fig, .949, .837, "B", size=9, weight="bold", ha="center", color=NAVY)
    cases = (("correct-flip", "Correct flip"), ("correct-stable", "Correct stable answer"),
             ("failed-flip", "Failed flip"))
    for index, (kind, label) in enumerate(cases):
        key = chosen[kind]
        fixture_path = root / by_id[key]["fixture"]
        fixture = json.loads(fixture_path.read_text())
        target = next(iter(fixture["pairs"]))
        center = fixture["targets"][target]["pixel_xy"]
        filenames = [fixture["rgb"], *(f["sp_map"] for f in fixture["configurations"].values())]
        head = .790 - index * .249
        rule(fig, .035, .97, head + .024)
        group_label(fig, .047, head-.005, "ABC"[index], label)
        text(fig, .315, head-.005, f"Is the {target} in the robot’s reachable space?", size=9, color=MUTED)
        text(fig, .968, head-.005, key, size=7.5, color=MUTED, ha="right")
        for x, name in zip((.035, .260, .485), filenames, strict=True):
            picture(fig, [x, head-.208, .216, .175], image(fixture_path.parent/name), pixel=center, zoom=True)
        for i, (label, pair) in enumerate((("Certified", tuned[key]), ("Frozen", frozen[key]),
                                          ("Projector", tuned[key]))):
            y = head - .081 - i*.044
            tint = (SKY, SOFT, MINT if kind != "failed-flip" else PINK)[i]
            panel(fig, .715, y-.018, .255, .036, tint, tint)
            text(fig, .725, y, label, size=8.8, color=MUTED)
            for x, row in zip((.886, .949), pair, strict=True):
                response = row["gold"] if i == 0 else row["free"]
                color = INK if i == 0 else GREEN_TEXT if response == row["gold"] else RED_TEXT
                text(fig, x, y, answer(response), size=9, weight="bold", color=color, ha="center")
    text(fig, .035, .032, "Frozen checkpoint and projector seed 17. Image crops and rings are for display.", size=7.8, color=MUTED)
    save(fig, destination)


def control_gallery(root, predictions, destination):
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    keys = [(r["image"], r["question"]) for r in rows]
    picks = []
    for truth, correct in (("Y", True), ("N", True), ("N", False)):
        picks.append(next(row for row in rows
                          if row["truth"] == truth
                          and (row["original"]["free"] == truth) == correct
                          and keys.count((row["image"], row["question"])) == 1
                          and row["image"] not in {item["image"] for item in picks}))
    fig = plt.figure(figsize=(9.2, 5.22))
    title(fig, "PhysVLM on the authors’ simulation recipe",
          "Regenerated Panda scenes · Original question wording · Released checkpoint")
    rule(fig, .035, .97, .856)
    text(fig, .035, .806, "Scene and question", size=9, weight="bold", color=NAVY)
    text(fig, .427, .806, "RGB", size=9, weight="bold", color=NAVY, ha="center")
    text(fig, .671, .806, "S-P Map", size=9, weight="bold", color=NAVY, ha="center")
    text(fig, .821, .806, "Answer", size=9, weight="bold", color=NAVY)
    for idx, row in enumerate(picks):
        y = .748 - idx*.237
        label = ["Correct Yes", "Correct No", "False positive"][idx]
        group_label(fig, .048, y-.022, "ABC"[idx], label)
        name = re.fullmatch(r"Is the (.+) in the robot's reachable space\?", row["question"])[1]
        metadata_name = row["image"].replace("_rgb_", "_data_").replace(".jpg", ".json")
        raw = json.loads((root / metadata_name).read_text())["bounding_boxes"]
        bbox = raw.get(name.replace(" ", "_"))
        text(fig, .035, y-.092, f"Target   {name}", size=9.5, weight="bold")
        text(fig, .035, y-.141, textwrap.fill(row["question"], 34), size=8.45, color=MUTED)
        text(fig, .035, y-.199, row["id"], size=7.5, color=MUTED)
        for x, file in zip((.317, .561), (row["image"], row["sp_map"]), strict=True):
            picture(fig, [x, y-.216, .22, .211], image(root / file), bbox=bbox)
        predicted = row["original"]["free"]
        for offset, tag, value, color in ((.082, "Recipe label", row["truth"], INK),
                (.144, "PhysVLM", predicted, GREEN_TEXT if predicted == row["truth"] else RED_TEXT)):
            tint = SKY if tag == "Recipe label" else MINT if predicted == row["truth"] else PINK
            panel(fig,.812,y-offset-.019,.158,.038,tint,tint)
            text(fig, .819, y-offset, tag, size=8.8, color=MUTED)
            text(fig, .965, y-offset, answer(value), size=9, color=color, weight="bold", ha="right")
        if idx < 2:
            rule(fig, .035, .97, y-.224)
    text(fig, .035, .023, "Boxes locate the named object. The recipe labels and predictions are recorded in the control manifest.",
         size=7.55, color=MUTED)
    save(fig, destination)
    return [row["id"] for row in picks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workflow(args.pairs, args.evaluation, args.examples, args.output / "workflow.png")
    gallery(args.pairs, args.evaluation, args.examples, args.output / "counterfactual-gallery.png")
    ids = control_gallery(args.control, args.control_predictions, args.output / "official-recipe-gallery.png")
    combine_image_results(args.output)
    (args.output / "official-gallery-scenes.json").write_text(json.dumps(ids, indent=2) + "\n")


def combine_image_results(output):
    names = ("counterfactual-gallery.png", "official-recipe-gallery.png")
    components = []
    for name in names:
        with Image.open(output / name) as original:
            components.append(original.convert("RGB"))
    width = max(pic.width for pic in components)
    merged = Image.new("RGB", (width, sum(pic.height for pic in components) + 30), PAPER)
    y = 0
    for pic in components:
        merged.paste(pic, ((width - pic.width) // 2, y))
        y += pic.height + 30
    merged.save(output / "image-results.png")


if __name__ == "__main__":
    main()
