#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import proxydiff_nas as method


def assert_close(left, right, tolerance=1e-8):
    if not np.allclose(left, right, atol=tolerance, rtol=0.0):
        raise AssertionError(f"max error={np.max(np.abs(left - right))}")


def main() -> None:
    rng = np.random.default_rng(9000)
    core = rng.normal(size=(196, 3))
    aligned = method._normalize_columns(core)

    assert method._effective_rank(np.zeros((196, 3))) == 0.0
    assert abs(method._effective_rank(np.ones((196, 3))) - 1.0) < 1e-8
    orthogonal = np.eye(4)
    assert abs(method._effective_rank(orthogonal) - 4.0) < 1e-8
    assert abs(method._effective_rank(core) - method._effective_rank(core[:, [2, 0, 1]])) < 1e-8
    grouped_utilities, grouped_ranks, grouped_rank = (
        method._group_indexed_stage1_statistics(core)
    )
    expected_group_utilities = np.row_stack([
        method._proxy_utilities(method._normalize_columns(group))
        for group in np.split(core, 2, axis=0)
    ]).mean(axis=0)
    expected_group_ranks = [
        method._effective_rank(method._rank_align(group))
        for group in np.split(core, 2, axis=0)
    ]
    assert_close(grouped_utilities, expected_group_utilities)
    assert_close(np.asarray(grouped_ranks), np.asarray(expected_group_ranks))
    if abs(grouped_rank - float(np.mean(expected_group_ranks))) > 1e-8:
        raise AssertionError("group-indexed effective rank must use an equal group mean")

    selected_indices, subset_metadata = method._consensus_rank_subset_gate(
        core, ["a", "b", "c"], subset_size=2
    )
    selected_aligned = method._rank_align(core)[:, selected_indices]
    selected_row = subset_metadata["selected"]
    expected_subset_rank = method._effective_rank(selected_aligned)
    if abs(selected_row["subset_effective_rank"] - expected_subset_rank) > 1e-8:
        raise AssertionError("budget compression must match global factorization geometry")
    assert_close(
        method._rankdata_average(np.asarray([1.0, 1.0, 3.0])),
        np.asarray([1.5, 1.5, 3.0]),
    )
    if method._largest_gap_positive_indices(np.asarray([0.020, 0.005, 0.004])) != [0]:
        raise AssertionError("largest-gap complement cardinality must retain the separated group")
    if method._largest_gap_positive_indices(np.asarray([0.5, 0.5, 0.0])) != [0, 1]:
        raise AssertionError("tied positive complement evidence must remain tied")
    if method._largest_gap_positive_indices(np.asarray([0.064, 0.055])) != [0]:
        raise AssertionError("complement cardinality must use an observed internal gap, not a zero sentinel")
    if abs(method._combine_cross_context_reliability(0.8, 0.5) - 0.4) > 1e-12:
        raise AssertionError("cross-context reliability must multiply support and agreement")
    if method._combine_cross_context_reliability(0.8) != 0.8:
        raise AssertionError("unpaired contexts must use neutral agreement")

    in_span = 0.3 * aligned[:, 0] - 0.7 * aligned[:, 2]
    candidate = rng.normal(size=196)
    matrix = np.column_stack([aligned, in_span, candidate])
    raw_residual, residual, _ = method._project_residual_complement(matrix, [0, 1, 2], 4)
    basis = method._orthonormal_basis(matrix[:, :3])
    assert_close(basis.T @ raw_residual, np.zeros(basis.shape[1]))
    projection = basis @ (basis.T @ matrix[:, 4])
    assert_close(projection + raw_residual, matrix[:, 4])
    assert method._safe_corr(residual, matrix[:, 4]) >= 0.0

    span_residual, _, _ = method._project_residual_complement(matrix, [0, 1, 2], 3)
    if np.linalg.norm(span_residual) > 1e-8:
        raise AssertionError("a core-span candidate must have zero residual")

    vectors, _ = method.factorize_proxy_matrix(core, ["a", "b", "c"], apply_gate=False)
    term_count = len(vectors)
    probe = vectors[0] + residual / float(term_count)
    old_scaled_probe = (term_count * vectors[0] + residual) / float(term_count + 1)
    assert_close(
        method._rankdata_average(probe),
        method._rankdata_average(old_scaled_probe),
    )

    duplicate_core = np.column_stack([core[:, 0], core[:, 0], core[:, 1]])
    duplicate_vectors, duplicate_meta = method.factorize_proxy_matrix(
        duplicate_core,
        ["a", "a_duplicate", "b"],
        apply_gate=False,
    )
    if duplicate_meta["n_axes"] != 2 or len(duplicate_vectors) != 3:
        raise AssertionError(
            "duplicate proxy columns must not create a degenerate refinement axis"
        )
    if min(duplicate_meta["eigvals_desc"]) <= duplicate_meta["axis_rank_tolerance"]:
        raise AssertionError("cached axes must all exceed the numerical-rank tolerance")
    print("proxy factorization invariants: PASS")


if __name__ == "__main__":
    main()
