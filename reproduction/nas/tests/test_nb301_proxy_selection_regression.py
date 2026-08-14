#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import proxydiff_nas as method


EXPECTED_FULL = [
    "zcpt_jacob",
    "zcpt_l2_norm",
    "zcpt_meco",
    "zcpt_near",
    "zcpt_nwot",
    "zcpt_swap",
    "zcpt_synflow",
    "zcpt_zen",
    "zcpt_zico",
]
EXPECTED_BUDGETED = ["zcpt_jacob", "zcpt_snip", "zcpt_zico"]


def selected_names(score_root: Path, proxies: list[str], budget: int | None) -> list[str]:
    raw, _ = method.load_operation_scores(score_root, proxies)
    names = [f"zcpt_{method._clean_name(name)}" for name in proxies]
    _, metadata = method.factorize_proxy_matrix(
        raw,
        names,
        apply_gate=True,
        proxy_budget=budget,
    )
    return sorted(metadata["gate_kept_names"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True, type=Path)
    args = parser.parse_args()

    full = selected_names(args.score_root, method.FULL_PROXY_POOL, None)
    budgeted = selected_names(
        args.score_root,
        method.SHORT_SCORE_PROXY_POOL,
        method.BUDGETED_PROXY_BUDGET,
    )
    if full != EXPECTED_FULL:
        raise AssertionError(f"full selection changed: {full}")
    if budgeted != EXPECTED_BUDGETED:
        raise AssertionError(f"budgeted selection changed: {budgeted}")
    print("NB301 proxy selection regression: PASS")


if __name__ == "__main__":
    main()
