"""Score paired reachability with geometric baselines and map-response controls."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def wilson(correct, n):
    if not n:
        return None
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return [max(0, mid - half_width), min(1, mid + half_width)]


def summarize(rows, method):
    n = len(rows)
    correct = sum(row[method] == row["truth"] for row in rows)
    valid = [row for row in rows if "U" not in row[method]]
    changes = sum(row[method][0] != row[method][1] for row in valid)
    predicted_and_gold = [(prediction, label) for row in rows for prediction, label in zip(row[method], row["truth"])]
    negatives = [(prediction, label) for prediction, label in predicted_and_gold if label == "N"]
    return {
        "pairs": n, "both_correct": correct, "pair_accuracy": correct / n if n else None,
        "pair_accuracy_ci95": wilson(correct, n),
        "valid_pairs": len(valid), "invalid_answers": sum(key == "U" for key, _ in predicted_and_gold),
        "changed_pairs": changes, "change_rate_on_valid": changes / len(valid) if valid else None,
        "questions_correct": sum(a == b for a, b in predicted_and_gold), "questions": 2 * n,
        "false_positives": sum(a == "Y" for a, _ in negatives), "negative_questions": len(negatives),
    }


def bootstrap_delta(rows, comparator):
    deltas = [int(row["model"] == row["truth"]) - int(row[comparator] == row["truth"]) for row in rows]
    rng = random.Random(17)
    sampled = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
    return {"delta": sum(deltas) / len(deltas), "ci95": [sampled[250], sampled[9749]]}


def load_comparisons(selection_path, prediction_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    predictions = {row["scene"]: row for row in rows}
    scenes = {row["scene"] for row in selection["selection"]}
    if len(rows) != len(predictions) or predictions.keys() != scenes:
        raise ValueError("The predictions must have exactly one entry per selected scene.")
    comparisons = []
    for entry in selection["selection"]:
        row = predictions[entry["scene"]]
        fixture = json.loads((selection_path.parent / entry["fixture"]).read_text(encoding="utf-8"))
        target = row["target"]
        if fixture["pairs"] != {target: row["truth"]} or entry["label"] != row["truth"]:
            raise ValueError(f"Target or label mismatch in {entry['scene']}")
        prediction = "".join(row["predictions"][key]["answer"] for key in ("A", "B"))
        rules = [fixture["configurations"][key]["rules"][target] for key in ("A", "B")]
        comparisons.append({
            "scene": row["scene"], "target": target, "truth": row["truth"],
            "model": prediction, "swapped_map": prediction[::-1],
            "fixed_map_a": prediction[0] * 2, "fixed_map_b": prediction[1] * 2,
            "workspace_nn": "".join(rule["workspace_label"] for rule in rules),
            "radial": "".join(rule["radial_label"] for rule in rules),
        })
    return comparisons


def score(selection, predictions):
    rows = load_comparisons(selection, predictions)
    strata = Counter(row["truth"] for row in rows)
    splits = {
        "flip": [row for row in rows if row["truth"] in ("YN", "NY")],
        "stable": [row for row in rows if row["truth"] in ("YY", "NN")],
    }
    methods = ("model", "swapped_map", "fixed_map_a", "fixed_map_b", "workspace_nn", "radial")
    return {
        "strata": dict(strata),
        "metrics": {split: {method: summarize(group, method) for method in methods} for split, group in splits.items()},
        "paired_comparisons": {
            split: {method: bootstrap_delta(group, method) for method in ("workspace_nn", "radial")}
            for split, group in splits.items() if group
        },
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.selection, args.predictions), indent=2))
