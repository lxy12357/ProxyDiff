#!/usr/bin/env python3
"""Fail early when the NB301 refinement runtime is not the verified one."""

from __future__ import print_function

import sys

import ConfigSpace
import lightgbm
import numpy
import scipy
import sklearn
import torch
import torchvision
import xgboost


EXPECTED = {
    "python": (3, 7),
    "torch": "1.8.0+cu111",
    "torchvision": "0.9.0+cu111",
    "numpy": "1.21.5",
    "scipy": "1.7.3",
    "sklearn": "1.0.2",
    "ConfigSpace": "0.4.19",
    "xgboost": "1.4.2",
    "lightgbm": "4.4.0",
}


def main():
    observed = {
        "python": tuple(sys.version_info[:2]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "ConfigSpace": ConfigSpace.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
    }
    errors = []
    for name, expected in EXPECTED.items():
        if observed[name] != expected:
            errors.append("%s=%r, expected %r" % (name, observed[name], expected))
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
    if errors:
        raise SystemExit("NB301 refinement runtime mismatch: " + "; ".join(errors))
    print(
        "NB301 refinement runtime verified: "
        "python=%d.%d torch=%s torchvision=%s numpy=%s scipy=%s "
        "sklearn=%s ConfigSpace=%s xgboost=%s lightgbm=%s cuda=%s"
        % (
            observed["python"][0],
            observed["python"][1],
            observed["torch"],
            observed["torchvision"],
            observed["numpy"],
            observed["scipy"],
            observed["sklearn"],
            observed["ConfigSpace"],
            observed["xgboost"],
            observed["lightgbm"],
            torch.version.cuda,
        )
    )


if __name__ == "__main__":
    main()
