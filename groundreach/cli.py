"""Run the point-token study from a recorded protocol or replay its evidence."""

import argparse
from pathlib import Path

from groundreach.io import read, sha256_file


def point_token(*, inference_only=False):
    parser = argparse.ArgumentParser(description="Run the matched point-token study.")
    for name in ("config", "model", "train", "test", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("train-predictions", "test-predictions"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--load-heads", type=Path, required=inference_only)
    args = parser.parse_args()
    cfg = read(args.config)
    if cfg["point_source"] == "automatic" and not (args.train_predictions and args.test_predictions):
        parser.error("This reference protocol requires both automatic localization predictions")
    for split in ("train", "test"):
        actual = sha256_file(getattr(args, split))
        if actual != cfg["manifest_hashes"][split]:
            parser.error(f"The {split} manifest differs from the reference protocol")
    command = ["--model-sha256", cfg["model_sha256"], "--seed", str(cfg["seed"]),
               "--epochs", str(cfg["epochs"])]
    for name in ("model", "train", "test", "output", "train_predictions", "test_predictions", "load_heads"):
        value = getattr(args, name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", str(value)))

    from groundreach.training import main

    main(command)


def reproduce():
    parser = argparse.ArgumentParser(description="Replay all point-token confirmation and perturbation predictions.")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()

    from groundreach.evaluation.counterfactual import summarize

    pilot = args.repository / "results/grounding-study/point-token-pilot"
    cases = (("automatic", "confirm"), ("cross-check-29", "confirm"),
             ("cross-check-17", "permuted"), ("automatic-permuted", "permuted"), ("jitter", "permuted"))
    for case, split in cases:
        folder = pilot / f"point-token-{case}"
        result = summarize(pilot / f"point-token-{split}/manifest.json", read(folder / "predictions.json"))
        if result != read(folder / "paired.json"):
            raise ValueError(f"Replayed statistics disagree for {case}")
        point = result["outcomes"]["adapted_point"]
        mean = result["outcomes"]["adapted_mean"]
        print(f"{case}: point {point['correct']}, mean {mean['correct']}, "
              f"target switches {point['target_flip']['correct']}, map switches {point['workspace_flip']['correct']}")


def infer():
    """Run the recorded protocol with already trained heads."""
    point_token(inference_only=True)
