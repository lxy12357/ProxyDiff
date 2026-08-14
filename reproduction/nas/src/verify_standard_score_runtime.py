#!/usr/bin/env python3
"""Verify the pinned runtime used by the standard operation scores."""

from __future__ import annotations

import sys

import numpy
import pandas
import scipy
import sklearn
import torch
import torchvision


EXPECTED = {
    "python": (3, 9),
    "torch": "2.0.0+cu117",
    "torchvision": "0.15.1+cu117",
    "cudnn": 8500,
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "sklearn": "1.5.0",
    "pandas": "1.4.2",
}


def main() -> None:
    actual = {
        "python": tuple(sys.version_info[:2]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
    if actual != EXPECTED:
        raise SystemExit(f"standard score runtime mismatch: expected={EXPECTED}, actual={actual}")
    print(f"standard score runtime verified: {actual}")


if __name__ == "__main__":
    main()
