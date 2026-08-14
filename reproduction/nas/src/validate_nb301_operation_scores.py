#!/usr/bin/env python3
"""Validate NB301 operation-score artifacts before cache construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_ARCH_SHA256 = "852c3187ef2645469b52cfb94e9f6fa9e7b1859814f17727f181efe390c16f37"
EXPECTED_PROTOCOL = "zcpt_single_op_ablation_prior_8cell_c16"
SCALAR_REVISIONS = {
    "cell_local_v2_20260605",
    "cell_local_operation_ablation_20260605",
}
TE_NAS_REVISIONS = {
    "cell_local_tenas_delta_v1_20260612",
    "cell_local_tenas_delta_20260612",
}
EXPECTED_SCALE = "DARTS projected network, 8 cells, C=16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_path(root: Path, method: str) -> Path:
    for prefix in ("nb301_zcpt_", "nb301_official_zcpt_"):
        path = root / f"{prefix}{method}" / "operation_scores.json"
        if path.is_file():
            return path
    raise FileNotFoundError(f"operation score not found for {method}")


def validate_score(path: Path, method: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_batch_size = 8 if method == "near" else 64
    expected = {
        "search_space": "nasbench301",
        "dataset": "cifar10",
        "protocol": EXPECTED_PROTOCOL,
        "network_scale": EXPECTED_SCALE,
        "seed": 9000,
        "batch_size": expected_batch_size,
        "candidate_count": 196,
    }
    failures = [
        f"{key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    allowed_revisions = TE_NAS_REVISIONS if method == "te_nas" else SCALAR_REVISIONS
    if payload.get("protocol_revision") not in allowed_revisions:
        failures.append(
            f"protocol_revision={payload.get('protocol_revision')!r}, "
            f"expected one of {sorted(allowed_revisions)!r}"
        )
    predictor = str(payload.get("predictor", "")).lower()
    if not predictor.endswith(method):
        failures.append(f"predictor={predictor!r}, expected method {method!r}")
    records = payload.get("records", [])
    coordinates = {
        (str(row.get("cell_type")), int(row.get("edge_id", -1)), int(row.get("op_id", -1)))
        for row in records
    }
    expected_coordinates = {
        (cell, edge, operation)
        for cell in ("normal", "reduce")
        for edge in range(14)
        for operation in range(7)
    }
    if len(records) != 196 or coordinates != expected_coordinates:
        failures.append("records do not contain exactly the 196 NB301 operation coordinates")
    operation_scores = [row.get("operation_score") for row in records]
    if any(value is None for value in operation_scores):
        failures.append("one or more operation scores are null")
    elif any(not math.isfinite(float(value)) for value in operation_scores):
        failures.append("one or more operation scores are not finite")
    if failures:
        raise ValueError(f"invalid {path}: " + "; ".join(failures))
    return {
        "method": method,
        "path": str(path),
        "sha256": sha256(path),
        "predictor": payload["predictor"],
        "seed": payload["seed"],
        "batch_size": payload["batch_size"],
        "protocol_revision": payload["protocol_revision"],
        "candidate_count": len(records),
        "operation_score_sha256": sha256(path),
        "method_internal_seed": payload.get("method_internal_seed"),
        "backend": payload.get("backend"),
        "scorer_source_sha256": payload.get("scorer_source_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True, type=Path)
    parser.add_argument("--fixed-arch", required=True, type=Path)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()

    arch_hash = sha256(args.fixed_arch)
    if arch_hash != EXPECTED_ARCH_SHA256:
        raise ValueError(f"fixed architecture SHA-256 mismatch: {arch_hash}")
    rows = [
        validate_score(score_path(args.score_root, method), method)
        for method in args.methods
    ]
    manifest = {
        "fixed_arch_sha256": arch_hash,
        "protocol": EXPECTED_PROTOCOL,
        "accepted_protocol_revisions": {
            "scalar": sorted(SCALAR_REVISIONS),
            "te_nas": sorted(TE_NAS_REVISIONS),
        },
        "seed": 9000,
        "scores": rows,
    }
    if args.output_manifest:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"validated {len(rows)} NB301 operation-score artifacts")


if __name__ == "__main__":
    main()
