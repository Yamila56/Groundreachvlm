"""Render and certify one development pair."""

import argparse
import json
from pathlib import Path

from groundreach.simulation.panda import generate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pairs = generate_pair(output)
    if pairs != {"red block": "NY", "blue block": "YY"}:
        raise RuntimeError(f"Expected one flip and one invariant, obtained {pairs}")
    print(json.dumps(pairs))


if __name__ == "__main__":
    main()
