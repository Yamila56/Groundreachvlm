"""Run PhysVLM validation diagnostics, fine-tuning and held-out evaluation serially."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def execute(module, *args):
    subprocess.run([sys.executable, "-u", "-m", f"{__package__}.{module}", *map(str, args)], check=True)


def run(directory, mode, model, *args):
    events = directory / "events.jsonl"
    if events.is_file() and json.loads(events.read_text().splitlines()[-1])["event"] == "complete":
        return
    if directory.exists():
        directory.rename(directory.with_name(f"{directory.name}-interrupted-{time.time_ns()}"))
    directory.parent.mkdir(parents=True, exist_ok=True)
    execute("tune_reachability", mode, "--model", model, "--output", directory, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phases", nargs="+", choices=("diagnostic", "probes", "train", "test"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trained = [(m, s) for s in (17, 29) for m in ("projector", "lora")]
    args.output.mkdir(parents=True, exist_ok=True)
    for phase in args.phases:
        here = args.output / phase
        if phase == "diagnostic":
            runs = [("whole", ()), ("target", ("--roi", "target")), ("wrong", ("--roi", "wrong"))]
            for name, flags in runs:
                run(here / name, "frozen", args.model, "--test", args.data / "dev", *flags)
            if not (here / "metrics.json").is_file():
                execute("summarize_matched", here / "metrics.json",
                        *(f"{name}={here/name/'predictions.jsonl'}" for name, _ in runs))
        elif phase == "probes":
            for mode in ("projector", "lora"):
                target = here / mode
                run(target, mode, args.model, "--train", args.data / "dev", "--dev", args.data / "dev",
                    "--probe", "--epochs", 5)
                run(here / f"{mode}-reload", mode, args.model, "--test", args.data / "dev",
                    "--checkpoint", target / "best.pt")
        elif phase == "train":
            for mode, seed in trained:
                run(here / f"{mode}{seed}", mode, args.model, "--seed", seed,
                    "--train", args.data / "train", "--dev", args.data / "dev")
        else:
            run(here / "frozen", "frozen", args.model, "--test", args.data / "test")
            predictions = [f"frozen={here}/frozen/predictions.jsonl"]
            for mode, seed in trained:
                name = f"{mode}{seed}"
                checkpoint = args.output / "train" / name / "best.pt"
                run(here / name, mode, args.model, "--seed", seed, "--test", args.data / "test",
                    "--checkpoint", checkpoint)
                predictions.append(f"{name}={here/name/'predictions.jsonl'}")
            if not (here / "metrics.json").is_file():
                execute("summarize_matched", here / "metrics.json", *predictions)
        (args.output / f"{phase}.complete").touch()


if __name__ == "__main__":
    main()
