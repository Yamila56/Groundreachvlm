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
    result = dict(
        manifest_sha256=candidates["manifest_sha256"],
        detector_threshold=candidates["confidence"],
        annotated_groups=candidates["n_unique_annotated"],
        candidate_recall=candidates["recalls"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    training = {}
    for seed in (17, 29):
        rows = read(args.run/f"train-{seed}"/"training.json")
        training[str(seed)] = {k: v for k, v in rows.items() if k not in ("train_images", "dev_images")}
    result = dict(training=training,
                  controlled_holdout=summarize_split(args.run/"controlled-holdout"),
                  source_holdout=summarize_split(args.run/"source-holdout"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
