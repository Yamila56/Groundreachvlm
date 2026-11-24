"""Render held-out examples selected deterministically from the paired predictions."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from summarize_matched import read


def grid(path, fixture, baseline, tuned, title):
    rgb_names = [fixture["rgb"], *(f["sp_map"] for f in fixture["configurations"].values())]
    canvas = Image.new("RGB", (1140, 385), "#f4f6f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    draw.text((18, 12), title, font=font, fill="#1f2933")
    for index, name in enumerate(rgb_names):
        with Image.open(path.parent / name) as image:
            resized = image.convert("RGB")
            resized.thumbnail((360, 268))
        x = 15 + 380 * index
        canvas.paste(resized, (x, 45))
        if index == 0:
            text = "Same RGB in both conditions"
            details = f"Target: {next(iter(fixture['pairs']))}"
        else:
            a, b = baseline[index-1], tuned[index-1]
            text = f"Map {b['side']}    Certified answer: {b['gold']}"
            details = f"Frozen: {a['free']}     Projector: {b['free']}"
        draw.text((x, 322), text, font=font, fill="#1f2933")
        draw.text((x, 349), details, font=small, fill="#465463")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("projector", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frozen, tuned = read(args.frozen), read(args.projector)
    selection = json.loads(args.selection.read_text())["selection"]
    args.output.mkdir(parents=True, exist_ok=False)
    used = {}
    for entry in sorted(selection, key=lambda item: item["scene"]):
        key = entry["scene"]
        both_correct = all(r["gold"] == r["free"] for r in tuned[key])
        label = entry["label"]
        if both_correct:
            kind = "correct-flip" if label in ("YN", "NY") else "correct-stable"
        elif label in ("YN", "NY"):
            kind = "failed-flip"
        else:
            continue
        if kind in used:
            continue
        path = args.selection.parent / entry["fixture"]
        figure = grid(path, json.loads(path.read_text()), frozen[key], tuned[key], kind.replace("-", " ").title())
        figure.save(args.output / f"{kind}.png")
        used[kind] = key
    (args.output / "examples.json").write_text(json.dumps(used, indent=2) + "\n")


if __name__ == "__main__":
    main()
