"""Compatibility entry point for the packaged implementation."""

import sys

from groundreach.evaluation import counterfactual as implementation


if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
