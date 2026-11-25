"""Recompute certified point and paired metrics from the released per-question predictions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def counts(rows, values):
    known = [row for row in rows if row["truth"] != "U"]
    views, targets, maps = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        views[row["scene"]].append(row)
        targets[row["scene"], row["side"]].append(row)
        maps[row["scene"], row["target"]].append(row)

    def correct(row):
        return values[row["id"]] == row["truth"]

    def joint(groups, size, opposite=False):
        selected = [group for group in groups.values()
                    if len(group) == size and all(row["truth"] != "U" for row in group)
                    and (not opposite or {row["truth"] for row in group} == {"Y", "N"})]
        return dict(total=len(groups), certified=len(selected),
                    correct=sum(all(map(correct, group)) for group in selected))

    return dict(certified=len(known), correct=sum(map(correct, known)),
                unanswered=sum(values[row["id"]] == "U" for row in known),
                confusion=dict(Counter(f"{row['truth']}-{values[row['id']]}" for row in rows)),
                target_flip=joint(targets, 2, opposite=True), workspace_flip=joint(maps, 2, opposite=True),
                four=joint(views, 4))


def replay(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique per-question predictions")
    methods = rows[0]["predictions"].keys()
    labels = {"Y", "N", "U"}
    if any(row["predictions"].keys() != methods or row["truth"] not in labels or
           not set(row["predictions"].values()) <= labels for row in rows):
        raise ValueError("Mismatched prediction methods or label values")
    return dict(n=len(rows), physical_layouts=len({row["layout"] for row in rows}),
                labels=dict(Counter(row["truth"] for row in rows)),
                metrics={method: counts(rows, {row["id"]: row["predictions"][method] for row in rows})
                         for method in methods})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay(json.loads(args.predictions.read_text(encoding="utf-8")))
    if args.expected:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        for field in ("n", "physical_layouts", "labels"):
            if expected[field] != report[field]:
                raise ValueError(f"Published {field} disagrees with the replay")
        for name, value in report["metrics"].items():
            if any(expected["metrics"][name]["all"][key] != result for key, result in value.items()):
                raise ValueError(f"Published metrics for {name} disagree with the replay")
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
