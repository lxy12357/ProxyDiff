#!/usr/bin/env python3
"""Summarize NB301 fixed-subset v2 refinement outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


SUBSETS = {
    "retained_k3_rep01": ("retained_pool", 3),
    "retained_k3_rep02": ("retained_pool", 3),
    "retained_k3_rep03": ("retained_pool", 3),
    "retained_k5_rep01": ("retained_pool", 5),
    "retained_k5_rep02": ("retained_pool", 5),
    "retained_k5_rep03": ("retained_pool", 5),
    "all_proxy_k3_rep01": ("full_proxy_pool", 3),
    "all_proxy_k3_rep02": ("full_proxy_pool", 3),
    "all_proxy_k3_rep03": ("full_proxy_pool", 3),
    "all_proxy_k5_rep01": ("full_proxy_pool", 5),
    "all_proxy_k5_rep02": ("full_proxy_pool", 5),
    "all_proxy_k5_rep03": ("full_proxy_pool", 5),
}


def load_final(path: Path) -> tuple[float, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("rows", []):
        if row.get("method") == "proxydiff_epoch_000_score_params":
            top = row["free_top_score_arch"]
            return float(top["acc"]), int(top["equiv_fixed1000_rank"])
    raise KeyError(f"final refinement row missing: {path}")


def summarize(out_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_id, (pool, size) in SUBSETS.items():
        path = out_root / subset_id / "free_decode.json"
        if not path.exists():
            rows.append(
                {
                    "pool": pool,
                    "subset_size": size,
                    "subset_id": subset_id,
                    "final_acc": "",
                    "final_rank": "",
                    "status": "missing",
                }
            )
            continue
        acc, rank = load_final(path)
        rows.append(
            {
                "pool": pool,
                "subset_size": size,
                "subset_id": subset_id,
                "final_acc": acc,
                "final_rank": rank,
                "status": "ok",
            }
        )
    ok_rows = [row for row in rows if row["status"] == "ok"]
    for pool in sorted({str(row["pool"]) for row in ok_rows}):
        for size in sorted({int(row["subset_size"]) for row in ok_rows if row["pool"] == pool}):
            accs = [float(row["final_acc"]) for row in ok_rows if row["pool"] == pool and int(row["subset_size"]) == size]
            ranks = [int(row["final_rank"]) for row in ok_rows if row["pool"] == pool and int(row["subset_size"]) == size]
            if accs:
                rows.append(
                    {
                        "pool": pool,
                        "subset_size": size,
                        "subset_id": "summary_median",
                        "final_acc": median(accs),
                        "final_rank": median(ranks),
                        "status": "ok",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(args.out_root.resolve())
    fields = ["pool", "subset_size", "subset_id", "final_acc", "final_rank", "status"]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
