#!/usr/bin/env python3
"""Summarize NB301 ProxyDiff main-table evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


ROWS = {
    "nb301_full_proxy_pool": {
        "model": "NB301",
        "retention": "full proxy pool with utility gate",
        "expected_acc": 94.6956,
        "expected_rank": 1,
        "source": "github/nas/evidence/nb301_full_proxy_pool_summary.jsonl",
        "final_stage": "task_conditioned_refinement",
    },
    "nb301_three_proxy_subset": {
        "model": "NB301",
        "retention": "3-proxy fixed subset",
        "expected_acc": 94.5154,
        "expected_rank": 1,
        "source": "github/nas/evidence/nb301_three_proxy_subset_summary.json",
        "final_stage": "score_prior",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_full_proxy_pool(root: Path) -> dict:
    meta = ROWS["nb301_full_proxy_pool"]
    source = root / meta["source"]
    rows = read_jsonl(source)
    final = next(row for row in rows if row["stage"] == meta["final_stage"])
    return {
        "row": "nb301_full_proxy_pool",
        "stage": final["stage"],
        "acc": float(final["acc"]),
        "rank": int(final["rank"]),
        "genotype": final.get("genotype", ""),
        "source": meta["source"],
    }


def load_three_proxy_subset(root: Path) -> dict:
    meta = ROWS["nb301_three_proxy_subset"]
    source = root / meta["source"]
    row = json.loads(source.read_text(encoding="utf-8"))
    if row.get("row") != "nb301_three_proxy_subset":
        raise RuntimeError(f"unexpected three-proxy evidence row in {source}")
    return {
        "row": "nb301_three_proxy_subset",
        "stage": row["stage"],
        "acc": float(row["acc"]),
        "rank": int(row["rank"]),
        "genotype": "",
        "source": meta["source"],
    }


def summarize(root: Path) -> Iterable[dict[str, object]]:
    for row in (load_full_proxy_pool(root), load_three_proxy_subset(root)):
        meta = ROWS[row["row"]]
        row.update(
            {
                "model": meta["model"],
                "retention": meta["retention"],
                "expected_acc": meta["expected_acc"],
                "expected_rank": meta["expected_rank"],
                "matches_expected": (
                    round(float(row["acc"]), 4) == meta["expected_acc"]
                    and int(row["rank"]) == meta["expected_rank"]
                ),
            }
        )
        yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    rows = list(summarize(root))
    fields = [
        "row",
        "model",
        "retention",
        "stage",
        "acc",
        "rank",
        "expected_acc",
        "expected_rank",
        "matches_expected",
        "source",
    ]
    output_rows = [{key: row[key] for key in fields} for row in rows]
    if args.out_csv is not None:
        out = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output_rows)


if __name__ == "__main__":
    main()
