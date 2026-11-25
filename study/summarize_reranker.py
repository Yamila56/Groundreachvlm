"""Collect the held-out ranking and physical readout results."""
import argparse
import json
from pathlib import Path

METHODS = ("detector", "clip", "fusion", "chroma", "learned17", "learned29")
READOUTS = ("frozen", "oracle-full", "oracle-center", "predicted-full",
            "predicted-center", "wrong-full", "predicted-projector")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_split(root):
    candidates = read(root/"candidate-upper-bound.json")
    frozen = read(root/"detector-score.json")["predictions"]["frozen"]
    rows = candidates["rows"]
    if len(rows) != len(frozen) or {r["id"] for r in rows} != frozen.keys():
        raise ValueError(f"Candidate audit and frozen model disagree on evaluated IDs for {root.name}")
    covered = [r for r in rows if r["oracle_top5"] is not None]
    fallback = [r for r in rows if r["oracle_top5"] is None]
    oracle_candidate_correct = sum(r["oracle_top5"] == r["truth"] for r in covered)
    result = dict(
        manifest_sha256=candidates["manifest_sha256"],
        detector_threshold=candidates["confidence"],
        annotated_groups=candidates["n_unique_annotated"],
        candidate_recall=candidates["recalls"],
        oracle_candidate_readout=dict(
            n=len(rows), covered=len(covered), correct_on_covered=oracle_candidate_correct,
            correct_with_frozen_fallback=oracle_candidate_correct + sum(frozen[r["id"]] == r["truth"] for r in fallback),
        ),
        methods={},
    )
    for method in METHODS:
        localization = read(root/f"{method}-target.json")
        if localization["manifest_sha256"] != result["manifest_sha256"]:
            raise ValueError(f"Manifest mismatch for {root.name}/{method}")
        suffix = "-projector-score.json" if method.startswith("learned") else "-score.json"
        physical = read(root/f"{method}{suffix}")["metrics"]
        item = dict(
            top1=localization["recalls"]["1"],
            readouts={k: physical[k]["all"] for k in READOUTS if k in physical},
            scores_by_localization=localization["query_readout"],
        )
        if method != "detector":
            transitions = read(root/f"{method}-transitions.json")
            item["versus_detector"] = {k: transitions[k] for k in ("n", "transitions", "versus_original")}
        result["methods"][method] = item
    return result


def source_semantics(run, audit):
    rows = audit["rows"]
    usable = [row for row in rows if row["point"] is not None]
    by_definition = {
        "radial": {r["id"]: r["radial_label"] for r in usable},
        "sampled_workspace_majority": {r["id"]: "Y" if r["visible_in_workspace_fraction"] >= .5 else "N"
                                       for r in usable},
        "sampled_workspace_point": {r["id"]: r["point"]["sampled_workspace_label"] for r in usable},
    }
    original = read(run/"source-oracle-projector"/"metrics.json")["predictions"]
    predictions = {k: original[k] for k in ("frozen", "oracle-full", "oracle-center",
                                           "oracle-projector17", "oracle-projector29")}
    for seed in (17, 29):
        values = read(run/"source-holdout"/f"learned{seed}-projector-score.json")["predictions"]
        for key in ("predicted-full", "predicted-center", "predicted-projector"):
            predictions[f"{key}-{seed}"] = values[key]
    if any(pred.keys() != {r["id"] for r in rows} for pred in predictions.values()):
        raise ValueError("Map audit and model responses must describe the same questions")
    different = {key for key, label in by_definition["radial"].items()
                 if label != by_definition["sampled_workspace_point"][key]}
    return dict(n=len(usable), changed_radial_to_point=len(different),
                correct={method: {name: sum(pred[i] == value for i, value in truth.items())
                                  for name, truth in by_definition.items()}
                         for method, pred in predictions.items()},
                on_changed={method: {name: sum(pred[i] == truth[i] for i in different)
                                     for name, truth in by_definition.items()}
                            for method, pred in predictions.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-geometry", type=Path)
    args = parser.parse_args()
    training = {}
    for seed in (17, 29):
        rows = read(args.run/f"train-{seed}"/"training.json")
        training[str(seed)] = {k: v for k, v in rows.items() if k not in ("train_images", "dev_images")}
    result = dict(training=training,
                  controlled_holdout=summarize_split(args.run/"controlled-holdout"),
                  source_holdout=summarize_split(args.run/"source-holdout"))
    oracle_projector = args.run/"source-oracle-projector"/"metrics.json"
    if (oracle_projector.parent/"DONE").exists():
        metrics = read(oracle_projector)["metrics"]
        result["source_holdout"]["oracle_projector"] = {
            name: metrics[name]["all"] for name in ("oracle-projector17", "oracle-projector29")
        }
    if args.source_geometry:
        geometry = read(args.source_geometry)
        if geometry["manifest_sha256"] != result["source_holdout"]["manifest_sha256"]:
            raise ValueError("Scene replay and ranking used different manifests")
        result["source_holdout"]["retrospective_semantics"] = source_semantics(args.run, geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
