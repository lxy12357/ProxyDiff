#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nas_reported"))

import generate_nb301_subset_manifest as subsets


EXPECTED_RETAINED = [
    ["synflow", "zen", "zico"],
    ["l2_norm", "nwot", "synflow"],
    ["meco", "nwot", "zico"],
    ["jacob", "meco", "near", "nwot", "zico"],
    ["nwot", "swap", "synflow", "zen", "zico"],
    ["jacob", "l2_norm", "meco", "zen", "zico"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-selection-details", required=True, type=Path)
    args = parser.parse_args()
    first = subsets.build_manifest(args.full_selection_details, subsets.DEFAULT_SEED)
    second = subsets.build_manifest(args.full_selection_details, subsets.DEFAULT_SEED)
    subsets.validate_manifest(first)
    if first != second:
        raise AssertionError("subset manifest generation is not deterministic")
    retained = [
        row["proxy_names"]
        for row in first["subsets"]
        if row["pool"] == "retained_pool"
    ]
    if retained != EXPECTED_RETAINED:
        raise AssertionError(f"current retained-pool subsets changed: {retained}")
    print("NB301 subset manifest regression: PASS")


if __name__ == "__main__":
    main()
