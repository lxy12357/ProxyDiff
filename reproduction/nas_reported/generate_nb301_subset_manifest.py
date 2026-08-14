#!/usr/bin/env python3
"""Generate the fixed NB301 proxy-subset stability manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


DEFAULT_SEED = 20260615
DEFAULT_SIZES = (3, 5)
DEFAULT_REPEATS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-selection-details", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--print-tsv", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def clean_name(value: str) -> str:
    return value[5:] if value.startswith("zcpt_") else value


def load_pools(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    factorization = payload["factorization"]
    full_pool = [clean_name(name) for name in factorization["proxy_names"]]
    retained_set = {
        clean_name(name) for name in factorization["gate_kept_names"]
    }
    retained_pool = [name for name in full_pool if name in retained_set]
    if len(full_pool) != len(set(full_pool)):
        raise ValueError("full proxy pool contains duplicate names")
    if len(retained_pool) != len(retained_set):
        raise ValueError("retained proxy names are not a subset of the full pool")
    if len(retained_pool) < max(DEFAULT_SIZES):
        raise ValueError("retained pool is too small for the requested subsets")
    return retained_pool, full_pool


def sample_pool(
    pool_name: str,
    proxy_names: list[str],
    full_order: list[str],
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = []
    prefix = "retained" if pool_name == "retained_pool" else "all_proxy"
    for size in DEFAULT_SIZES:
        for repeat in range(1, DEFAULT_REPEATS + 1):
            selected = rng.sample(proxy_names, size)
            selected.sort(key=full_order.index)
            rows.append(
                {
                    "subset_id": f"{prefix}_k{size}_rep{repeat:02d}",
                    "pool": pool_name,
                    "subset_size": size,
                    "repeat": repeat,
                    "proxy_names": selected,
                }
            )
    return rows


def build_manifest(details_path: Path, seed: int) -> dict[str, object]:
    retained_pool, full_pool = load_pools(details_path)
    rows = sample_pool("retained_pool", retained_pool, full_pool, seed)
    rows.extend(sample_pool("full_proxy_pool", full_pool, full_pool, seed))
    return {
        "definition": "predeclared random proxy subsets for NB301 stability comparisons",
        "selection_rule": "Python random.sample without replacement; independent seed-reset stream per source pool",
        "seed": seed,
        "sizes": list(DEFAULT_SIZES),
        "repeats_per_size": DEFAULT_REPEATS,
        "source_pools": {
            "retained_pool": retained_pool,
            "full_proxy_pool": full_pool,
        },
        "full_selection_details": details_path.as_posix(),
        "subsets": rows,
    }


def validate_manifest(payload: dict[str, object]) -> None:
    rows = payload.get("subsets", [])
    expected = 2 * len(DEFAULT_SIZES) * DEFAULT_REPEATS
    if len(rows) != expected:
        raise ValueError(f"expected {expected} subset rows, found {len(rows)}")
    ids = [str(row["subset_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("subset manifest contains duplicate subset IDs")
    for row in rows:
        names = list(row["proxy_names"])
        if len(names) != int(row["subset_size"]) or len(names) != len(set(names)):
            raise ValueError(f"invalid subset row: {row}")


def print_tsv(payload: dict[str, object]) -> None:
    for row in payload["subsets"]:
        print(
            f"{row['subset_id']}\t{row['pool']}\t{row['subset_size']}\t"
            + " ".join(row["proxy_names"])
        )


def main() -> None:
    args = parse_args()
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        if not args.full_selection_details or not args.output:
            raise SystemExit("generation requires --full-selection-details and --output")
        payload = build_manifest(args.full_selection_details, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(args.output)
        finally:
            if temporary.exists():
                temporary.unlink()
    validate_manifest(payload)
    if args.print_tsv:
        print_tsv(payload)
    elif args.output:
        print(args.output)


if __name__ == "__main__":
    main()
