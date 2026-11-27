"""Summarize paired free answers and candidate margins under map interventions."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[row["scene"]].append(row)
    result = {}
    for key, rows in groups.items():
        a, b = sorted(rows, key=lambda r: r["side"])
        if a["side"] != "A" or b["side"] != "B" or a["gold"] + b["gold"] != a["pair_label"]:
            raise ValueError(f"Mismatched inference rows: {key}")
        result[key] = a, b
    return result


def report(groups, method, condition):
    pairs, outcomes, margins = defaultdict(list), [], []
    key = "free" if method == "free" else method

    def extract(row):
        return row[key] if method == "free" else "Y" if row[key] > 0 else "N" if row[key] < 0 else "U"

    for a, b in groups.values():
        predict = {"correct": (extract(a), extract(b)), "swapped": (extract(b), extract(a)),
                   "fixed_a": (extract(a), extract(a)), "fixed_b": (extract(b), extract(b))}[condition]
        truth = (a["gold"], b["gold"])
        hit = int(predict == truth)
        group = "flip" if truth[0] != truth[1] else "stable"
        for name in (group, a["pair_label"], f"family_{a['family']}_{group}"):
            pairs[name].append(hit)
        outcomes.extend(zip(truth, predict, strict=True))
        if group == "flip" and method != "free":
            margins.append((a[key] - b[key]) * (1 if a["gold"] == "Y" else -1))
    n_yes = sum(y == "Y" for y, _ in outcomes)
    n_no = sum(y == "N" for y, _ in outcomes)
    return dict(pairs={name: dict(correct=sum(v), n=len(v), accuracy=mean(v)) for name, v in pairs.items()},
        accuracy=mean(y == p for y, p in outcomes), invalid=sum(p == "U" for _, p in outcomes),
        recall_yes=sum(y == p == "Y" for y, p in outcomes)/n_yes,
        false_positive=sum(y == "N" and p == "Y" for y, p in outcomes)/n_no,
        flip_margin=({"mean": mean(margins), "correct_direction": mean(x > 0 for x in margins)}
                     if condition == "correct" and margins else None))


def interval(reference, tuned):
    if reference.keys() != tuned.keys():
        raise ValueError("The comparison requires matching scenes")
    clusters = defaultdict(list)
    families = defaultdict(list)
    for key, baseline in reference.items():
        group = tuned[key]
        if any(a["gold"] != b["gold"] for a, b in zip(baseline, group, strict=True)):
            raise ValueError("Gold conditions changed")
        if baseline[0]["gold"] == baseline[1]["gold"]:
            continue
        value = int(all(r["gold"] == r["free"] for r in group)) - int(all(
                    r["gold"] == r["free"] for r in baseline))
        clusters[baseline[0]["block"]].append(value)
        families[baseline[0]["family"]].append(value)
    rng = random.Random(59)
    result = {}
    for name, pool in (("anchor", clusters), ("layout_family", families)):
        units = list(pool.values())
        draws = sorted(mean(x for unit in rng.choices(units, k=len(units)) for x in unit)
                       for _ in range(10000))
        result[name] = dict(n=len(units), bounds=[draws[250], draws[9749]])
    return dict(delta=mean(d for cluster in clusters.values() for d in cluster), ci95=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("groups", nargs="+")
    args = parser.parse_args()
    methods = {name: read(Path(path)) for name, path in (g.split("=", 1) for g in args.groups)}
    result = {name: {method: {condition: report(pairs, method, condition)
                for condition in ("correct", "swapped", "fixed_a", "fixed_b")}
                for method in ("free", "margin", "marker_margin")} for name, pairs in methods.items()}
    if "frozen" in methods:
        result["flip_difference_vs_frozen"] = {n: interval(methods["frozen"], g)
                                               for n, g in methods.items() if n != "frozen"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
