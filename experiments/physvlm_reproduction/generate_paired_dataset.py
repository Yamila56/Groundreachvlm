"""Generate independent tabletop scenes balanced by certified paired label."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from groundreach.simulation.panda import generate_pair
from groundreach.simulation.upstream import SIMULATOR

COLORS = {
    "red block": [0.86, 0.19, 0.15, 1.0],
    "blue block": [0.15, 0.32, 0.83, 1.0],
    "green block": [0.15, 0.69, 0.28, 1.0],
    "yellow block": [0.94, 0.75, 0.14, 1.0],
}
X_POSITIONS = {
    "YN": (-0.67, -0.63),
    "NY": (0.96, 1.05),
    "YY": (0.40, 0.55),
    "NN": (1.31, 1.46),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=80)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SIMULATOR))
    accepted = {key: 0 for key in X_POSITIONS}
    selection = []
    rejected = []
    for seed in range(args.seed_start, args.seed_start + args.max_candidates):
        incomplete = [key for key in accepted if accepted[key] < args.per_stratum]
        if not incomplete:
            break
        desired = min(incomplete, key=accepted.get)
        rng = np.random.default_rng(seed)
        target, distractor = rng.choice(list(COLORS), size=2, replace=False)
        target_xyz = [rng.uniform(*X_POSITIONS[desired]), rng.uniform(-0.09, 0.09), 0.245]
        distractor_xyz = [rng.uniform(0.25, 0.62), rng.uniform(0.24, 0.38), 0.245]
        layout = {target: (target_xyz, COLORS[target]), distractor: (distractor_xyz, COLORS[distractor])}
        scene = f"scene-{seed:05d}"
        folder = args.output / scene
        folder.mkdir()
        pairs = generate_pair(folder.resolve(), layout, target, scene)
        actual = pairs[target]
        if actual == desired:
            accepted[actual] += 1
            selection.append({"scene": scene, "fixture": f"{scene}/paired_pilot.json", "label": actual})
        else:
            rejected.append({"scene": scene, "target_stratum": desired, "observed": actual})
        print(json.dumps({"scene": scene, "accepted": accepted, "observed": actual}), flush=True)
    payload = {"accepted": accepted, "selection": selection, "rejected": rejected, "start_seed": args.seed_start}
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if any(count != args.per_stratum for count in accepted.values()):
        raise RuntimeError(f"Unable to fill requested strata: {accepted}")


if __name__ == "__main__":
    main()
