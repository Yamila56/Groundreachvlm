"""Draw a crossed two-target example from a generated alignment manifest."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    scene = data[0]["scene"]
    rows = [row for row in data if row["scene"] == scene]
    source = args.manifest.parent
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    axes[0].imshow(plt.imread(source / rows[0]["image"]))
    axes[0].set_title("Shared RGB", fontsize=13)
    for index, side in enumerate(("A", "B"), 1):
        side_rows = [row for row in rows if row["side"] == side]
        axes[index].imshow(plt.imread(source / side_rows[0]["sp_map"]))
        axes[index].set_title(f"Workspace {side}", fontsize=13)
        labels = "  |  ".join(f"{r['target'].title()}: {'Yes' if r['truth']=='Y' else 'No'}"
                             for r in side_rows)
        axes[index].text(.5, -.055, labels, ha="center", transform=axes[index].transAxes, fontsize=8.5)
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(args.output, dpi=170, facecolor="white")


if __name__ == "__main__":
    main()
