#!/usr/bin/env python3
"""Verify a ProxyDiff cache sidecar before NB301 refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    parser.add_argument("--proxy-set", required=True)
    args = parser.parse_args()

    details = json.loads(args.details.read_text(encoding="utf-8"))
    failures = []
    if details.get("proxy_set") != args.proxy_set:
        failures.append(
            f"proxy_set={details.get('proxy_set')!r}, expected {args.proxy_set!r}"
        )
    expected_hash = details.get("cache_sha256")
    actual_hash = sha256(args.cache)
    if not expected_hash:
        failures.append("cache_sha256 is missing")
    elif expected_hash != actual_hash:
        failures.append(f"cache SHA-256 mismatch: {actual_hash}")
    n_metrics = int(details.get("n_metrics", 0))
    correction_columns = list(details.get("correction_columns", []))
    if n_metrics < 1 or len(correction_columns) != n_metrics - 1:
        failures.append("n_metrics and correction_columns are inconsistent")

    factorization = details.get("factorization") or {}
    selected = list(factorization.get("gate_kept_names", []))
    if args.proxy_set == "budgeted_proxy_subset" and len(selected) != 3:
        failures.append(f"budgeted gate retained {len(selected)} proxies, expected 3")
    if args.proxy_set == "full_proxy_pool":
        configured_budget = (factorization.get("backbone_selection") or {}).get(
            "configured_proxy_budget"
        )
        if configured_budget is not None:
            failures.append("full proxy pool must use automatic gate cardinality")
    if failures:
        raise SystemExit("invalid refinement cache: " + "; ".join(failures))
    print(
        f"validated {args.proxy_set} cache: sha256={actual_hash} "
        f"metrics={n_metrics} retained_proxies={len(selected)}"
    )


if __name__ == "__main__":
    main()
