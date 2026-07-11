#!/usr/bin/env python3
"""Combine ProxyDiff and paired aggregation results for fixed NB301 subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


SUBSETS = {
    "retained_k3_rep01": ("retained_pool", 3, ["near", "zen", "zico"]),
    "retained_k3_rep02": ("retained_pool", 3, ["jacob", "meco", "near"]),
    "retained_k3_rep03": ("retained_pool", 3, ["jacob", "l2_norm", "zico"]),
    "retained_k5_rep01": ("retained_pool", 5, ["jacob", "meco", "nwot", "swap", "zico"]),
    "retained_k5_rep02": ("retained_pool", 5, ["l2_norm", "near", "nwot", "swap", "zico"]),
    "retained_k5_rep03": ("retained_pool", 5, ["l2_norm", "meco", "nwot", "swap", "zico"]),
    "all_proxy_k3_rep01": ("full_proxy_pool", 3, ["meco", "snip", "swap"]),
    "all_proxy_k3_rep02": ("full_proxy_pool", 3, ["jacob", "meco", "plain"]),
    "all_proxy_k3_rep03": ("full_proxy_pool", 3, ["grad_norm", "jacob", "zen"]),
    "all_proxy_k5_rep01": ("full_proxy_pool", 5, ["eznas_darts", "grad_norm", "meco", "near", "plain"]),
    "all_proxy_k5_rep02": ("full_proxy_pool", 5, ["epsinas", "eznas_darts", "grasp", "jacob", "meco"]),
    "all_proxy_k5_rep03": ("full_proxy_pool", 5, ["eznas_darts", "fisher", "meco", "nwot", "zen"]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxydiff-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def proxydiff_accuracy(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in payload["rows"] if "epoch_000_score_params" in row.get("method", "")]
    if len(matches) != 1:
        raise ValueError(f"expected one final ProxyDiff row in {path}, found {len(matches)}")
    return float(matches[0]["free_top_score_arch"]["acc"])


def control_accuracy(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_accuracy"])


def main():
    args = parse_args()
    rows = []
    for subset_id, (pool, size, proxy_names) in SUBSETS.items():
        sources = {
            "proxydiff": args.proxydiff_root / subset_id / "free_decode.json",
            "log_rank": args.control_root / f"{subset_id}_log_rank" / "search_result.json",
            "rank_mean": args.control_root / f"{subset_id}_mean_rank" / "search_result.json",
        }
        for protocol, source in sources.items():
            if not source.exists():
                raise FileNotFoundError(source)
            accuracy = proxydiff_accuracy(source) if protocol == "proxydiff" else control_accuracy(source)
            rows.append(
                {
                    "pool": pool,
                    "subset_size": size,
                    "subset_id": subset_id,
                    "proxy_names": ",".join(proxy_names),
                    "protocol": protocol,
                    "accuracy": accuracy,
                    "source_result": str(source),
                }
            )
    summaries = []
    for pool in ["retained_pool", "full_proxy_pool"]:
        for size in [3, 5]:
            for protocol in ["rank_mean", "log_rank", "proxydiff"]:
                values = [
                    float(row["accuracy"])
                    for row in rows
                    if row["pool"] == pool and row["subset_size"] == size and row["protocol"] == protocol
                ]
                summaries.append(
                    {
                        "pool": pool,
                        "subset_size": size,
                        "protocol": protocol,
                        "minimum_accuracy": min(values),
                        "median_accuracy": median(values),
                        "maximum_accuracy": max(values),
                    }
                )
    fields = ["pool", "subset_size", "subset_id", "proxy_names", "protocol", "accuracy", "source_result"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "definition": "fixed subset comparison with identical proxy sets across rank-mean, AZ-style log-rank, and ProxyDiff",
        "rows": rows,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
