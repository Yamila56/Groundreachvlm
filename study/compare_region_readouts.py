"""Compare target choice and map-answer transitions on the same questions."""
import argparse
from collections import Counter
from pathlib import Path

from grounding_readout import cluster_delta, save
from rerank_grounding_targets import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("regions", "original", "proposed", "original_audit", "proposed_audit", "output"):
        parser.add_argument(option, type=Path)
    args = parser.parse_args()
    truth = read(args.regions)
    a, b = (read(path)["predictions"]["predicted-full"] for path in (args.original, args.proposed))
    audit_a, audit_b = ({row["id"]: row for row in read(path)["rows"]}
                        for path in (args.original_audit, args.proposed_audit))
    ids = {r["id"] for r in truth}
    if any(index.keys() != ids for index in (a, b, audit_a, audit_b)):
        raise ValueError("Methods use different query IDs")
    types = ("target_iou_at_least_half", "target_low_iou")
    counts, records = Counter(), []
    for row in truth:
        key, gold = row["id"], row["truth"]
        a_correct, b_correct = a[key] == gold, b[key] == gold
        a_target, b_target = (audit[key]["category"] in types for audit in (audit_a, audit_b))
        transition = f"{'right' if a_correct else 'wrong'}_to_{'right' if b_correct else 'wrong'}"
        target_change = f"identity_{int(a_target)}_to_{int(b_target)}"
        counts[transition] += 1
        counts[target_change] += 1
        counts[f"{transition}__{target_change}"] += 1
        records.append(dict(id=key, transition=transition, target_change=target_change,
                            before_category=audit_a[key]["category"], after_category=audit_b[key]["category"]))
    save(args.output, dict(n=len(truth), transitions=counts, versus_original=cluster_delta(truth, b, a),
                           rows=records))


if __name__ == "__main__":
    main()
