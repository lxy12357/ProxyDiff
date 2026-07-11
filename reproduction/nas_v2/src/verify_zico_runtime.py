#!/usr/bin/env python3
"""Verify the pinned CUDA runtime used for deterministic ZiCo scores."""

from __future__ import annotations

import numpy
import pandas
import scipy
import sklearn
import torch
import torchvision


EXPECTED = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "cudnn": 90100,
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "sklearn": "1.5.0",
    "pandas": "1.5.3",
}


def main() -> None:
    actual = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
    if actual != EXPECTED:
        raise SystemExit(f"ZiCo runtime mismatch: expected={EXPECTED}, actual={actual}")
    print(f"ZiCo runtime verified: {actual}")


if __name__ == "__main__":
    main()
