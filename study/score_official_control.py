"""Audit official-recipe prompts for conflicting labels and recompute the control scores."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows):
    labels = defaultdict(set)
    for row in rows:
        labels[row["image_sha256"], row["map_sha256"], row["question"]].add(row["truth"])
    clean, excluded = [], []
    for row in rows:
        key = row["image_sha256"], row["map_sha256"], row["question"]
        (clean if len(labels[key]) == 1 else excluded).append(row)
    metrics = {}
    for name, values in (("all", rows), ("unambiguous", clean)):
        metrics[name] = dict(count=len(values), prompts={})
        for prompt in ("original", "explicit"):
            outcomes = Counter(f"{row['truth']}-{row[prompt]['free']}" for row in values)
            scores = {}
            for field in ("margin", "marker_margin"):
                if values and all(field in row[prompt] for row in values):
                    scores[field] = sum(("Y" if row[prompt][field] > 0 else "N" if row[prompt][field] < 0
                                         else "U") == row["truth"] for row in values)
            metrics[name]["prompts"][prompt] = dict(confusion=outcomes, scores=scores)
    metrics["excluded"] = [dict(id=r["id"], truth=r["truth"], question=r["question"]) for r in excluded]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    with args.output.open("x") as stream:
        json.dump(summarize(rows), stream, indent=2)


if __name__ == "__main__":
    main()
