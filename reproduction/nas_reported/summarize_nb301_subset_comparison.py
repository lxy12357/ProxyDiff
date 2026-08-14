#!/usr/bin/env python3
"""Combine ProxyDiff and paired aggregation results for fixed NB301 subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxydiff-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, default=None)
    return parser.parse_args()


def proxydiff_result(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in payload["rows"] if row.get("method") == "proxydiff_epoch_000_score_params"]
    if len(matches) != 1:
        raise ValueError(f"expected one component-correction endpoint in {path}, found {len(matches)}")
    final_row = matches[0]
    selected = final_row["free_top_score_arch"]
    return float(selected["acc"]), int(selected["equiv_fixed1000_rank"])


def control_result(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["selected_accuracy"]), int(payload["selected_fixed_pool_rank"])


def main():
    args = parse_args()
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    rows = []
    for subset in manifest["subsets"]:
        subset_id = str(subset["subset_id"])
        pool = str(subset["pool"])
        size = int(subset["subset_size"])
        proxy_names = list(subset["proxy_names"])
        sources = {
            "proxydiff": args.proxydiff_root / subset_id / "free_decode.json",
            "log_rank": args.control_root / f"{subset_id}_log_rank" / "search_result.json",
            "rank_mean": args.control_root / f"{subset_id}_mean_rank" / "search_result.json",
        }
        for protocol, source in sources.items():
            if not source.exists():
                raise FileNotFoundError(source)
            if protocol == "proxydiff":
                accuracy, rank = proxydiff_result(source)
            else:
                accuracy, rank = control_result(source)
            rows.append(
                {
                    "pool": pool,
                    "subset_size": size,
                    "subset_id": subset_id,
                    "proxy_names": ",".join(proxy_names),
                    "protocol": protocol,
                    "accuracy": accuracy,
                    "rank": rank,
                    "source_result": source.as_posix(),
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
                ranks = [
                    int(row["rank"])
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
                        "median_rank": median(ranks),
                    }
                )
    fields = ["pool", "subset_size", "subset_id", "proxy_names", "protocol", "accuracy", "rank", "source_result"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "definition": "fixed subset comparison with identical proxy sets across rank-mean, AZ-style log-rank, and ProxyDiff",
        "subset_manifest": str(args.subset_manifest),
        "subset_seed": manifest["seed"],
        "rows": rows,
        "summaries": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv is not None:
        by_key = {(row["pool"], row["subset_size"], row["protocol"]): row for row in summaries}
        summary_fields = [
            "source_pool",
            "subset_size",
            "rank_mean_median",
            "log_rank_median",
            "proxydiff_median",
            "proxydiff_min",
            "proxydiff_max",
            "proxydiff_median_rank",
        ]
        summary_rows = []
        for pool in ["retained_pool", "full_proxy_pool"]:
            for size in [3, 5]:
                rank_mean = by_key[(pool, size, "rank_mean")]
                log_rank = by_key[(pool, size, "log_rank")]
                proxydiff = by_key[(pool, size, "proxydiff")]
                summary_rows.append(
                    {
                        "source_pool": pool,
                        "subset_size": size,
                        "rank_mean_median": rank_mean["median_accuracy"],
                        "log_rank_median": log_rank["median_accuracy"],
                        "proxydiff_median": proxydiff["median_accuracy"],
                        "proxydiff_min": proxydiff["minimum_accuracy"],
                        "proxydiff_max": proxydiff["maximum_accuracy"],
                        "proxydiff_median_rank": proxydiff["median_rank"],
                    }
                )
        args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_summary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
