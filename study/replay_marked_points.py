"""Recompute certified point and paired metrics from the released per-question predictions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundreach.evaluation.metrics import counts


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
