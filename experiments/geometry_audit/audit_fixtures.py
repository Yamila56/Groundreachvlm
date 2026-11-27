"""Replay the reachability certificates in a frozen fixture selection."""

import argparse
import json
from pathlib import Path

from groundreach.simulation.upstream import TOOL_LINK, TOLERANCE, URDF
from groundreach.geometry.certificates import audit as audit_certificates


def audit(selection_path):
    return audit_certificates(selection_path, URDF, TOOL_LINK, TOLERANCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.selection), indent=2))
