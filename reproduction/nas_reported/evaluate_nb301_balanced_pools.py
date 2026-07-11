#!/usr/bin/env python3
"""Map searched NB301 architectures to the paper's fixed balanced pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_POOL_SHA256 = "7818b72ffbbc50a53a9ad9b34bed7007649a438fab1205a45003238208750358"
EXPECTED_SPLITS_SHA256 = "2af3b354dd9ae481d7f6703a82a3062a9c6ba95b59eaa8911ed328fe18dd0624"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pool", type=Path, required=True)
    parser.add_argument("--balanced-splits", type=Path, required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Search result to evaluate; may be repeated",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--allow-unverified-reference", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result_spec(spec: str):
    if "=" not in spec:
        raise ValueError(f"expected LABEL=JSON, got {spec!r}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path)


def selected_accuracy(payload) -> float:
    if isinstance(payload, dict) and "selected_accuracy" in payload:
        return float(payload["selected_accuracy"])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if "selected_acc" in payload[0]:
            return float(payload[0]["selected_acc"])
    if isinstance(payload, dict) and "rows" in payload:
        final_rows = [
            row
            for row in payload["rows"]
            if "epoch_000_score_params" in str(row.get("method", ""))
        ]
        if len(final_rows) != 1:
            raise ValueError(f"expected one final metric_epoch=0 row, found {len(final_rows)}")
        return float(final_rows[0]["free_top_score_arch"]["acc"])
    raise ValueError("unsupported result JSON schema")


def main() -> None:
    args = parse_args()
    pool_hash = sha256(args.reference_pool)
    splits_hash = sha256(args.balanced_splits)
    if not args.allow_unverified_reference:
        if pool_hash != EXPECTED_POOL_SHA256:
            raise ValueError(f"reference pool SHA-256 mismatch: {pool_hash}")
        if splits_hash != EXPECTED_SPLITS_SHA256:
            raise ValueError(f"balanced splits SHA-256 mismatch: {splits_hash}")

    pool = json.loads(args.reference_pool.read_text(encoding="utf-8"))
    split_payload = json.loads(args.balanced_splits.read_text(encoding="utf-8"))
    records = pool["records"]
    split_indices = split_payload["spaces"]["nb301"]["splits"]
    expected_names = ["balanced1000_a", "balanced1000_b", "balanced1000_c"]
    if sorted(split_indices) != sorted(expected_names):
        raise ValueError(f"unexpected split names: {sorted(split_indices)}")
    split_sets = [set(int(index) for index in split_indices[name]) for name in expected_names]
    if any(len(indices) != 1000 for indices in split_sets):
        raise ValueError("each balanced split must contain 1,000 unique indices")
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("balanced splits must be pairwise disjoint")
    if len(set.union(*split_sets)) != 3000 or len(records) != 3000:
        raise ValueError("balanced splits must cover the fixed 3,000-architecture pool")

    rows = []
    for spec in args.result:
        label, result_path = parse_result_spec(spec)
        accuracy = selected_accuracy(json.loads(result_path.read_text(encoding="utf-8")))
        row = {"label": label, "selected_accuracy": accuracy, "source_result": str(result_path)}
        ranks = []
        for name in expected_names:
            rank = 1 + sum(float(records[index]["acc"]) > accuracy for index in split_indices[name])
            row[f"{name}_rank"] = rank
            ranks.append(rank)
        row["average_rank"] = sum(ranks) / len(ranks)
        rows.append(row)

    output = {
        "definition": "rank each searched architecture accuracy against the three fixed balanced1000 pools",
        "reference_pool": str(args.reference_pool),
        "reference_pool_sha256": pool_hash,
        "balanced_splits": str(args.balanced_splits),
        "balanced_splits_sha256": splits_hash,
        "split_protocol_revision": split_payload.get("protocol_revision"),
        "split_seed": split_payload.get("seed"),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "selected_accuracy",
        "balanced1000_a_rank",
        "balanced1000_b_rank",
        "balanced1000_c_rank",
        "average_rank",
        "source_result",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    print(args.output_csv.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
