#!/usr/bin/env python3
"""Verify the bundled NB301 runtime source and benchmark payload."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "ZeroCostNAS"
SOURCE_MANIFEST = ROOT / "runtime_source_sha256.txt"
BENCHMARK_PATH = SOURCE_ROOT / "data" / "zc_nasbench301.json"
BENCHMARK_SHA256 = "8951464c59220053b5468a694604e209b9fe9692ef024b00cc0db12ef2f11536"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve()
    failures = []
    for line in SOURCE_MANIFEST.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = runtime_root / relative
        if not path.is_file():
            failures.append(f"missing source: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"source hash mismatch: {relative}")
    benchmark_path = runtime_root / "data" / "zc_nasbench301.json"
    if not benchmark_path.is_file():
        failures.append("missing NB301 zero-cost benchmark")
        benchmark_hash = ""
    else:
        benchmark_hash = sha256(benchmark_path)
    if benchmark_hash != BENCHMARK_SHA256:
        failures.append("NB301 zero-cost benchmark hash mismatch")
    if failures:
        raise SystemExit("bundled runtime verification failed: " + "; ".join(failures))
    print(f"NB301 runtime source verified: {runtime_root}")


if __name__ == "__main__":
    main()
