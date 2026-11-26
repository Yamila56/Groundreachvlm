"""Compatibility import for the marked-point studies."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundreach.grounding.markers import COLOR, find_markers, locate_marker, marked_image
