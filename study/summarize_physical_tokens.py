"""Recompute layout-paired accuracy and token intervention effects."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_frozen_vlm import read, save
from replay_marked_points import counts


def paired_bootstrap(results):
    differences = np.array([sum(v) for v in results.values()])
    totals = np.array([len(v) for v in results.values()])
    if not len(totals):
        return dict(n=0, layouts=0, difference=None, interval95=None)
    samples = np.random.default_rng(914).integers(len(totals), size=(20000, len(totals)))
    delta = differences[samples].sum(1)/totals[samples].sum(1)
    return dict(n=int(totals.sum()), layouts=len(totals), difference=float(differences.sum()/totals.sum()),
                interval95=np.quantile(delta,[.025,.975]).tolist())


def summarize(manifest, predictions):
    rows = read(manifest)["rows"]
    by_id = {row["id"]:row for row in rows}
    answers = {mode:{key:p["answer"] for key,p in records.items()} for mode,records in predictions.items()}
    if any(x.keys()!=by_id.keys() for x in answers.values()):
        raise ValueError("Predictions must cover the entire manifest")
    workspaces, targets = defaultdict(list), defaultdict(list)
    for row in rows:
        workspaces[row["scene"],row["target"]].append(row)
        targets[row["scene"],row["side"]].append(row)
    comparisons = {}
    method = "adapted_point"
    truth = {r["id"]:r["truth"] for r in rows}
    for baseline, control in answers.items():
        if baseline==method:
            continue
        outcomes = answers[method]
        contrasts = {}
        for name, groups in (
            ("workspace_flip",workspaces),("target_flip",targets),("workspace_stable",workspaces),
        ):
            difference = defaultdict(list)
            for (scene,_), pair in groups.items():
                if len(pair)!=2 or any(p["truth"]=="U" for p in pair):
                    continue
                ids = [r["id"] for r in pair]
                flip = truth[ids[0]]!=truth[ids[1]]
                if flip!=(name!="workspace_stable"):
                    continue
                difference[scene].append(int(all(outcomes[i]==truth[i] for i in ids)) -
                                         int(all(control[i]==truth[i] for i in ids)))
            contrasts[name] = paired_bootstrap(difference)
        comparisons[f"{method}_vs_{baseline}"] = contrasts
    return dict(
        labels={value:sum(row["truth"]==value for row in rows) for value in ("Y","N","U")},
        outcomes={mode:counts(rows,values) for mode,values in answers.items()},
        comparisons=comparisons,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "predictions", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    save(args.output,summarize(args.manifest,read(args.predictions)))
