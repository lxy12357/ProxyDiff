#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "proxydiff_nas.py"
SPEC = importlib.util.spec_from_file_location("proxydiff_nas", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
METHOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METHOD)


def main() -> None:
    rng = np.random.RandomState(9000)
    names = ["fisher", "jacob", "meco", "nwot", "synflow", "zen"]
    raw = rng.normal(size=(196, len(names)))
    vectors, _ = METHOD.factorize_proxy_matrix(raw, names, apply_gate=False)

    permutation = [4, 1, 5, 0, 3, 2]
    permuted_names = [names[index] for index in permutation]
    permuted_raw = raw[:, permutation]
    permuted_vectors, _ = METHOD.factorize_proxy_matrix(
        permuted_raw, permuted_names, apply_gate=False
    )

    if len(vectors) != len(permuted_vectors):
        raise AssertionError("proxy permutation changed the number of factors")
    for index, (left, right) in enumerate(zip(vectors, permuted_vectors)):
        if not np.array_equal(left, right):
            max_abs = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            raise AssertionError(
                f"factor {index} is not exactly proxy-order invariant: {max_abs}"
            )
    print("NAS proxy-order invariance: PASS")


if __name__ == "__main__":
    main()
