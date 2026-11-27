"""Score original and boxed evaluations on the same author-recipe rows."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.physvlm_reproduction.evaluate_target_alignment import load_rows, write_json


def summarize(rows, prompt):
    correct = sum(row[prompt]["free"] == row["truth"] for row in rows)
    negative = [row for row in rows if row["truth"] == "N"]
    positive = [row for row in rows if row["truth"] == "Y"]
    count = dict(n=len(rows), correct=correct, yes_correct=sum(row[prompt]["free"] == "Y" for row in positive),
                 no_correct=sum(row[prompt]["free"] == "N" for row in negative),
                 invalid=sum(row[prompt]["free"] == "U" for row in rows), yes_n=len(positive), no_n=len(negative))
    return count


def paired(reference, treated, identity, prompt):
    clusters = defaultdict(list)
    for r, t in zip(reference, treated, strict=True):
        raw = int(t[prompt]["free"] == t["truth"]) - int(r[prompt]["free"] == r["truth"])
        clusters[identity[r["id"]]].append(raw)
    values = [(sum(rows), len(rows)) for rows in clusters.values()]
    rng = random.Random(20261125)
    draws = []
    for _ in range(10000):
        sample = rng.choices(values, k=len(values))
        draws.append(sum(n for n, _ in sample) / sum(d for _, d in sample))
    draws.sort()
    return dict(n_scenes=len(values), delta=sum(n for n, _ in values) / sum(d for _, d in values),
                ci95=[draws[250], draws[9749]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched_rows", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", help="name=prediction-file for each run, including frozen_full.")
    args = parser.parse_args()
    rows = json.loads(args.matched_rows.read_text())
    expected = {r["id"]: r for r in rows}
    identity = {r["id"]: r["image_sha256"] for r in rows}
    methods = {}
    for entry in args.runs:
        name, path = entry.split("=", 1)
        raw = load_rows(Path(path))
        mapping = {r["id"]: r for r in raw}
        if len(mapping) != len(raw) or not expected.keys() <= mapping.keys():
            raise ValueError(f"Missing or duplicate predictions for {name}")
        items = [mapping[key] for key in sorted(expected)]
        if any(row["truth"] != expected[row["id"]]["truth"] or row["question"] != expected[row["id"]]["question"]
               for row in items):
            raise ValueError(f"Reference question or label changed for {name}")
        methods[name] = items
    base = methods["frozen_full"]
    write_json(args.output, dict(
        geometry=dict(n=len(rows), correct=sum(r["geometry"] == r["truth"] for r in rows)),
        methods={name: {prompt: summarize(items, prompt) for prompt in ("original", "explicit")}
                 for name, items in methods.items()},
        versus_frozen_full={name: {prompt: paired(base, items, identity, prompt) for prompt in ("original", "explicit")}
                            for name, items in methods.items() if name != "frozen_full"}))


if __name__ == "__main__":
    main()
