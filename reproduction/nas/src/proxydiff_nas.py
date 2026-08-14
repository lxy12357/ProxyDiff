#!/usr/bin/env python3
"""Build ProxyDiff NAS score caches from ZCPT operation-ablation scores.

This script intentionally keeps only the paper-facing NB301 logic:

1. Load operation-level proxy scores from `operation_scores.json`.
2. Build a consensus backbone, project it to a hard proxy budget when needed,
   then admit reliable residual complements within the remaining budget.
3. Rank-align the retained proxies and build whitened PCA axes.
4. Orient axes by their strongest proxy profile and write the refinement cache.

The output cache format matches the external NAS runtime evaluator expected by
`run_nb301_proxy_refinement.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch


FULL_PROXY_POOL = [
    "epe_nas",
    "epsinas",
    "eznas_darts",
    "fisher",
    "grad_norm",
    "grasp",
    "jacob",
    "jacob_cov",
    "l2_norm",
    "meco",
    "near",
    "nwot",
    "plain",
    "snip",
    "swap",
    "synflow",
    "te_nas",
    "zen",
    "zico",
]

SHORT_SCORE_PROXY_POOL = [
    "epe_nas",
    "epsinas",
    "eznas_darts",
    "fisher",
    "grad_norm",
    "jacob",
    "jacob_cov",
    "l2_norm",
    "near",
    "nwot",
    "plain",
    "snip",
    "synflow",
    "zico",
]

COMPLEMENT_OPERATION_TOPK_COUNT = 5
REPEATED_STRUCTURE_CONFIDENCE_Z = 1.645
BUDGETED_PROXY_BUDGET = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_name(name: str) -> str:
    return name[5:] if name.startswith("zcpt_") else name


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    z = matrix.astype(float).copy()
    z = z - np.mean(z, axis=0, keepdims=True)
    scale = np.std(z, axis=0, ddof=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    return z / scale


def _sanitize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    eps = max(abs(lo), abs(hi), 1.0) * 1e-6
    out = arr.copy()
    out[np.isnan(out)] = lo
    out[np.isneginf(out)] = lo - eps
    out[np.isposinf(out)] = hi + eps
    return out


def _rank_align(raw: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(raw.shape[1]):
        ranks = _rankdata_average(_sanitize(raw[:, j]))
        cols.append(ranks)
    return _normalize_columns(np.column_stack(cols))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = _rankdata_average(np.asarray(a, dtype=float))
    b_rank = _rankdata_average(np.asarray(b, dtype=float))
    if float(np.std(a_rank)) == 0.0 or float(np.std(b_rank)) == 0.0:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def _proxy_utilities(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 1:
        return np.ones(matrix.shape[1], dtype=float)
    utilities = np.zeros(matrix.shape[1], dtype=float)
    for i in range(matrix.shape[1]):
        corrs = [abs(_safe_corr(matrix[:, i], matrix[:, j])) for j in range(matrix.shape[1]) if j != i]
        utilities[i] = float(np.mean(corrs)) if corrs else 1.0
    return utilities


def _effective_context_support(values: np.ndarray) -> tuple[float, float]:
    evidence = np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, None)
    context_count = int(evidence.size)
    total = float(evidence.sum())
    if context_count <= 1:
        return (1.0 if total > 1e-12 else 0.0), float(context_count)
    if total <= 1e-12:
        return 0.0, 0.0
    probabilities = evidence / total
    positive = probabilities[probabilities > 1e-12]
    effective_support = float(math.exp(-float(np.sum(positive * np.log(positive)))))
    reliability = effective_support / float(context_count)
    return float(np.clip(reliability, 0.0, 1.0)), effective_support


def _combine_cross_context_reliability(
    selection_change_support: float,
    paired_context_agreement: float = 1.0,
) -> float:
    support = float(np.clip(selection_change_support, 0.0, 1.0))
    agreement = float(np.clip(paired_context_agreement, 0.0, 1.0))
    return support * agreement


def _conservative_correlation_reliability(
    correlation: float,
    sample_count: int,
) -> tuple[float, float]:
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    standard_error = 1.0 / np.sqrt(max(sample_count - 3, 1))
    lower_bound = float(np.tanh(
        np.arctanh(clipped) - REPEATED_STRUCTURE_CONFIDENCE_Z * standard_error
    ))
    positive_lower_bound = max(lower_bound, 0.0)
    return positive_lower_bound ** 2, lower_bound


def _largest_gap_positive_indices(scores: np.ndarray) -> list[int]:
    values = np.nan_to_num(
        np.asarray(scores, dtype=float).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    positive = np.flatnonzero(values > 0.0)
    if positive.size == 0:
        return []
    order = positive[np.argsort(-values[positive], kind="mergesort")]
    if order.size == 1:
        return [int(order[0])]
    sorted_values = values[order]
    gaps = sorted_values[:-1] - sorted_values[1:]
    if float(np.max(gaps)) <= 1e-12:
        return [int(index) for index in order]
    cut = int(np.argmax(gaps)) + 1
    return [int(index) for index in order[:cut]]


def _group_indexed_stage1_statistics(
    raw: np.ndarray,
    group_count: int = 2,
) -> tuple[np.ndarray, list[float], float]:
    """Aggregate consensus utility and intrinsic rank over structural groups."""
    if group_count < 1 or raw.shape[0] % int(group_count) != 0:
        raise ValueError(
            f"cannot divide {raw.shape[0]} observations into {group_count} groups"
        )
    groups = np.split(np.asarray(raw, dtype=float), int(group_count), axis=0)
    utilities = np.row_stack([
        _proxy_utilities(_normalize_columns(group))
        for group in groups
    ]).mean(axis=0)
    effective_ranks = [
        _effective_rank(_rank_align(group))
        for group in groups
    ]
    return utilities, effective_ranks, float(np.mean(effective_ranks))


def _effective_rank(matrix: np.ndarray) -> float:
    covariance = (matrix.T @ matrix) / max(matrix.shape[0] - 1, 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    positive = probabilities[probabilities > 1e-12]
    return float(math.exp(-float(np.sum(positive * np.log(positive)))))


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    threshold = max(float(singular_values[0]), 1.0) * 1e-10
    return left_vectors[:, singular_values > threshold]


def _progressive_operation_mask(score: np.ndarray, top_k: int) -> np.ndarray:
    cells = np.asarray(score, dtype=float).reshape(2, 14, 7)
    mask = np.zeros_like(cells, dtype=bool)
    for cell in range(cells.shape[0]):
        for edge in range(cells.shape[1]):
            order = np.argsort(-cells[cell, edge], kind="mergesort")
            mask[cell, edge, order[:top_k]] = True
    return mask


def _project_residual_complement(
    aligned: np.ndarray,
    core_indices: list[int],
    candidate_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project one candidate against a fixed core and orient its residual."""
    core_basis = _orthonormal_basis(aligned[:, core_indices])
    candidate = aligned[:, candidate_index]
    raw_residual = candidate - core_basis @ (core_basis.T @ candidate)
    residual = _normalize_columns(raw_residual.reshape(-1, 1))[:, 0]
    direction = 1.0 if _safe_corr(residual, candidate) >= 0.0 else -1.0
    return raw_residual, residual * direction, direction


def _consensus_rank_subset_gate(
    raw: np.ndarray,
    proxy_names: list[str],
    subset_size: int,
) -> tuple[list[int], dict[str, object]]:
    aligned = _rank_align(raw)
    utilities = _proxy_utilities(_normalize_columns(raw))
    rows = []
    for indices_tuple in itertools.combinations(range(raw.shape[1]), subset_size):
        indices = list(indices_tuple)
        names = [proxy_names[index] for index in indices]
        consensus_strength = float(np.mean(utilities[indices]))
        subset_effective_rank = _effective_rank(aligned[:, indices])
        rank_efficiency = subset_effective_rank / subset_size
        rows.append({
            "indices": indices,
            "names": names,
            "consensus_strength": consensus_strength,
            "rank_efficiency": rank_efficiency,
            "subset_effective_rank": subset_effective_rank,
            "backbone_score": consensus_strength * rank_efficiency,
        })
    rows.sort(key=lambda row: (
        -float(row["backbone_score"]),
        -float(row["consensus_strength"]),
        -float(row["rank_efficiency"]),
        row["names"],
    ))
    selected = list(rows[0]["indices"])
    return selected, {
        "subset_size": subset_size,
        "selection_rule": (
            "within the fixed consensus core, maximize global consensus strength "
            "times global factorization-rank efficiency"
        ),
        "selected": rows[0],
        "top_candidates": rows[:10],
    }


def _select_budget_constrained_backbone(
    raw: np.ndarray,
    proxy_names: list[str],
    proxy_budget: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Retain the consensus core, projecting it only when the hard budget requires it."""
    if proxy_budget < 1:
        raise ValueError("proxy_budget must retain at least one proxy")

    utilities, group_effective_ranks, intrinsic_rank = (
        _group_indexed_stage1_statistics(raw)
    )
    intrinsic_count = max(
        1,
        min(raw.shape[1], int(math.ceil(intrinsic_rank - 1e-12))),
    )
    backbone_capacity = min(int(proxy_budget), raw.shape[1])
    order = np.asarray(sorted(
        range(raw.shape[1]),
        key=lambda index: (-float(utilities[index]), proxy_names[index]),
    ), dtype=int)
    consensus_kept = [int(index) for index in order[:intrinsic_count]]
    consensus_dropped = [int(index) for index in order[intrinsic_count:]]
    if backbone_capacity < intrinsic_count:
        compressed_indices, compression = _consensus_rank_subset_gate(
            raw[:, consensus_kept],
            [proxy_names[index] for index in consensus_kept],
            subset_size=backbone_capacity,
        )
        selected_indices = [consensus_kept[index] for index in compressed_indices]
        compression["candidate_scope"] = "consensus_core"
    else:
        selected_indices = consensus_kept
        compression = None
    selected_indices = sorted(selected_indices, key=lambda index: proxy_names[index])
    selected_count = len(selected_indices)
    backbone_mask = np.zeros(raw.shape[1], dtype=bool)
    backbone_mask[selected_indices] = True
    centered = utilities - float(np.mean(utilities))
    scale = max(float(np.std(centered)), 1e-12)
    reweighted = np.exp(np.clip(centered / scale, -20.0, 20.0))

    selected_set = set(selected_indices)
    return backbone_mask, reweighted, {
        "proxy_budget": int(proxy_budget),
        "backbone_capacity": int(backbone_capacity),
        "consensus_core_names": [proxy_names[index] for index in consensus_kept],
        "consensus_dropped_names": [proxy_names[index] for index in consensus_dropped],
        "backbone_names": [proxy_names[index] for index in selected_indices],
        "backbone_dropped_names": [
            name for index, name in enumerate(proxy_names) if index not in selected_set
        ],
        "intrinsic_effective_rank": intrinsic_rank,
        "stage1_group_effective_ranks": group_effective_ranks,
        "stage1_mean_effective_rank": intrinsic_rank,
        "intrinsic_core_count": intrinsic_count,
        "selected_core_count": selected_count,
        "stage1_utility_rule": (
            "mean absolute rank correlation within each structural group, "
            "then equal mean across groups"
        ),
        "cardinality_rule": (
            "ceiling of mean group spectral effective rank, followed by "
            "budget-constrained consensus-rank coreset when required"
        ),
        "compressed_to_budget": compression is not None,
        "compression": compression,
    }


def _repeated_structure_complement_gate(
    raw: np.ndarray,
    proxy_names: list[str],
    core_mask: np.ndarray,
    *,
    max_complements: int,
    operation_topk_count: int = COMPLEMENT_OPERATION_TOPK_COUNT,
) -> tuple[np.ndarray, dict[str, object], dict[int, np.ndarray]]:
    """Admit novel residuals that are coherent across NB301 cell types."""
    core_indices = [index for index in range(raw.shape[1]) if bool(core_mask[index])]
    candidate_indices = [index for index in range(raw.shape[1]) if not bool(core_mask[index])]
    if not candidate_indices:
        return core_mask.copy(), {
            "operation_topk_count": int(operation_topk_count),
            "candidate_evidence": [],
            "added_names": [],
        }, {}

    aligned = _rank_align(raw)
    core_aligned = aligned[:, core_indices]
    core_rank = _effective_rank(core_aligned)
    core_names = [proxy_names[index] for index in core_indices]
    core_vectors, _ = factorize_proxy_matrix(
        raw[:, core_indices],
        core_names,
        apply_gate=False,
    )
    core_readout = np.asarray(core_vectors[0], dtype=float)
    core_axes = [np.asarray(vector, dtype=float) for vector in core_vectors[1:]]
    core_term_count = 1 + len(core_axes)
    core_operation_mask = _progressive_operation_mask(core_readout, operation_topk_count)
    evidence = []
    candidate_residuals = {}
    for candidate_index in candidate_indices:
        raw_residual, residual, direction = _project_residual_complement(
            aligned,
            core_indices,
            candidate_index,
        )
        # Use one canonical factorized-term perturbation to test whether the
        # residual can change a structural decision. This label-free gate probe
        # is separate from the zero-weight refinement initialization.
        joined_readout = core_readout + residual / float(core_term_count)
        joined_operation_mask = _progressive_operation_mask(
            joined_readout,
            operation_topk_count,
        )
        fixed_context_changes = []
        for cell in range(core_operation_mask.shape[0]):
            for edge in range(core_operation_mask.shape[1]):
                fixed_core = set(np.flatnonzero(core_operation_mask[cell, edge]))
                fixed_joined = set(np.flatnonzero(joined_operation_mask[cell, edge]))
                fixed_context_changes.append(
                    1.0 - len(fixed_core & fixed_joined) / max(len(fixed_core | fixed_joined), 1)
                )
        fixed_context_changes = np.asarray(fixed_context_changes, dtype=float)
        decision_change = float(fixed_context_changes.mean())
        support_reliability, effective_context_support = (
            _effective_context_support(fixed_context_changes)
        )
        residual_cells = residual.reshape(2, 14, 7)
        global_cross_cell_correlation = _safe_corr(
            residual_cells[0].reshape(-1),
            residual_cells[1].reshape(-1),
        )
        repeated_structure_reliability, correlation_lower_bound = (
            _conservative_correlation_reliability(
                global_cross_cell_correlation,
                residual_cells[0].size,
            )
        )
        rank_delta = max(
            0.0,
            _effective_rank(np.column_stack([core_aligned, residual])) - core_rank,
        )
        cross_context_reliability = _combine_cross_context_reliability(
            support_reliability,
            repeated_structure_reliability,
        )
        if (
            rank_delta > 0.0
            and decision_change > 0.0
            and cross_context_reliability > 0.0
        ):
            complement_score = (
                rank_delta
                * decision_change
                * cross_context_reliability
            )
        else:
            complement_score = 0.0
        evidence.append({
            "index": candidate_index,
            "name": proxy_names[candidate_index],
            "residual_direction": direction,
            "residual_norm_before_standardization": float(np.linalg.norm(raw_residual)),
            "effective_rank_delta": rank_delta,
            "fixed_view_operation_jaccard_drop": float(fixed_context_changes.mean()),
            "admission_operation_jaccard_drop": decision_change,
            "fixed_context_operation_jaccard_drop": fixed_context_changes.tolist(),
            "effective_context_support": effective_context_support,
            "selection_change_support_reliability": support_reliability,
            "paired_context_agreement": repeated_structure_reliability,
            "global_cross_cell_spearman": global_cross_cell_correlation,
            "cross_cell_correlation_lower_bound": correlation_lower_bound,
            "repeated_structure_reliability": repeated_structure_reliability,
            "cross_context_reliability": cross_context_reliability,
            "complement_score": complement_score,
        })
        candidate_residuals[candidate_index] = residual

    evidence.sort(key=lambda row: (-float(row["complement_score"]), row["name"]))
    admitted_candidates = _largest_gap_positive_indices(np.asarray([
        float(row["complement_score"]) for row in evidence
    ]))
    admitted_candidate_set = set(admitted_candidates[:max(0, int(max_complements))])
    mask = core_mask.copy()
    added_names = []
    for evidence_index, row in enumerate(evidence):
        admitted = evidence_index in admitted_candidate_set
        row["admitted_by_largest_gap"] = bool(admitted)
        row["admitted_within_budget"] = bool(admitted)
        if admitted:
            mask[int(row["index"])] = True
            added_names.append(str(row["name"]))
    admitted_residuals = {
        int(row["index"]): candidate_residuals[int(row["index"])]
        for row in evidence
        if bool(row["admitted_within_budget"])
    }
    return mask, {
        "selection_rule": (
            "effective-rank gain x fixed-factorization operation Jaccard change "
            "x effective context support x paired-cell agreement; retain the "
            "positive-evidence group above its largest gap"
        ),
        "cross_context_reliability": (
            "selection-change effective support multiplied by optional paired-context "
            "agreement; NB301 supplies conservative normal/reduction residual agreement"
        ),
        "admission_capacity": max(0, int(max_complements)),
        "operation_topk_count": int(operation_topk_count),
        "gate_probe_coefficient": "one_over_core_factorized_term_count",
        "core_effective_rank": core_rank,
        "candidate_evidence": evidence,
        "added_names": added_names,
        "abstained": not added_names,
    }, admitted_residuals


def load_operation_scores(op_root: Path, proxies: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    vectors = []
    metadata = {}
    for proxy in proxies:
        name = _clean_name(proxy)
        candidates = [
            op_root / f"nb301_zcpt_{name}" / "operation_scores.json",
            op_root / f"nb301_official_zcpt_{name}" / "operation_scores.json",
        ]
        path = next((item for item in candidates if item.exists()), candidates[0])
        payload = json.loads(path.read_text(encoding="utf-8"))
        tensor = np.full((2, 14, 7), np.nan, dtype=float)
        cell_map = {"normal": 0, "reduce": 1}
        for rec in payload["records"]:
            cell = cell_map[str(rec["cell_type"])]
            edge = int(rec["edge_id"])
            op = int(rec["op_id"])
            value = rec.get("operation_score")
            tensor[cell, edge, op] = float("nan") if value is None else float(value)
        if np.isnan(tensor).any():
            raise RuntimeError(f"Incomplete operation score tensor: {path}")
        vectors.append(_sanitize(tensor.reshape(-1)))
        metadata[f"zcpt_{name}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "batch_size": payload.get("batch_size"),
            "protocol_revision": payload.get("protocol_revision"),
            "score_definition": payload.get("score_definition"),
        }
    return np.column_stack(vectors), metadata


def factorize_proxy_matrix(
    raw: np.ndarray,
    proxy_names: list[str],
    *,
    apply_gate: bool,
    proxy_budget: Optional[int] = None,
    complement_admission_rule: str = "repeated_structure_reliability",
) -> tuple[list[np.ndarray], dict[str, object]]:
    gate_utilities = _proxy_utilities(_normalize_columns(raw))
    if apply_gate:
        selection_capacity = raw.shape[1] if proxy_budget is None else int(proxy_budget)
        core_mask, reweighted, backbone_meta = _select_budget_constrained_backbone(
            raw,
            proxy_names,
            selection_capacity,
        )
        backbone_meta["budget_mode"] = "automatic" if proxy_budget is None else "hard"
        backbone_meta["configured_proxy_budget"] = proxy_budget
        core_kept = sorted(
            (index for index in range(raw.shape[1]) if bool(core_mask[index])),
            key=lambda index: proxy_names[index],
        )
        core_dropped = [index for index in range(raw.shape[1]) if not bool(core_mask[index])]
        if complement_admission_rule == "repeated_structure_reliability":
            mask, complement_meta, admitted_residuals = _repeated_structure_complement_gate(
                raw,
                proxy_names,
                core_mask,
                max_complements=selection_capacity - len(core_kept),
            )
        else:
            raise ValueError(
                f"unknown complement_admission_rule={complement_admission_rule}"
            )
        gate_kept = [index for index in range(raw.shape[1]) if bool(mask[index])]
        gate_dropped = [index for index in range(raw.shape[1]) if not bool(mask[index])]
        complement_indices = [index for index in gate_kept if not bool(core_mask[index])]
        retained = raw[:, core_kept]
        retained_names = [proxy_names[index] for index in core_kept]
    else:
        reweighted = gate_utilities
        backbone_meta = None
        core_kept = sorted(
            range(raw.shape[1]),
            key=lambda index: proxy_names[index],
        )
        core_dropped = []
        complement_meta = None
        admitted_residuals = {}
        complement_indices = []
        gate_kept = list(core_kept)
        gate_dropped = []
        retained = raw[:, core_kept]
        retained_names = [proxy_names[index] for index in core_kept]

    z = _rank_align(retained)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = np.clip(eigvals[::-1], 0.0, None)
    largest_eigenvalue = float(eigvals_desc[0]) if eigvals_desc.size else 0.0
    if largest_eigenvalue <= 0.0:
        raise RuntimeError("retained proxy matrix has no non-degenerate spectral axis")
    axis_rank_tolerance = (
        largest_eigenvalue
        * max(z.shape)
        * np.finfo(np.float64).eps
    )
    nondegenerate = eigvals_desc > axis_rank_tolerance
    eigvals_desc = eigvals_desc[nondegenerate]
    pc_all = eigvecs[:, ::-1][:, nondegenerate]
    axes = (z @ pc_all) / np.sqrt(np.clip(eigvals_desc.reshape(1, -1), 1e-12, None))

    consensus = z.mean(axis=1)
    for j in range(axes.shape[1]):
        if _safe_corr(axes[:, j], consensus) < 0:
            axes[:, j] *= -1.0

    prior = axes.mean(axis=1, keepdims=True)
    comp = np.column_stack([prior, axes])
    proxy_lookup = {
        _clean_name(name): retained[:, i]
        for i, name in enumerate(retained_names)
    }
    orient_names = list(proxy_lookup)

    pre_oriented_axes = []
    oriented_axes = []
    axis_meta = []
    for axis_id in range(1, comp.shape[1]):
        profile = np.asarray([_safe_corr(comp[:, axis_id], proxy_lookup[name]) for name in orient_names], dtype=float)
        strongest_idx = int(np.argmax(np.abs(profile))) if profile.size else 0
        direction = 1.0 if profile.size == 0 or profile[strongest_idx] >= 0 else -1.0
        oriented_profile = direction * profile
        pre_oriented_axes.append(comp[:, axis_id])
        oriented_axes.append(comp[:, axis_id] * direction)
        axis_meta.append(
            {
                "axis": int(axis_id),
                "direction": float(direction),
                "strongest_proxy": orient_names[strongest_idx] if orient_names else "",
                "strongest_abs_corr": float(abs(profile[strongest_idx])) if profile.size else 0.0,
                "corr": {name: float(profile[i]) for i, name in enumerate(orient_names)},
                "oriented_corr": {name: float(oriented_profile[i]) for i, name in enumerate(orient_names)},
            }
        )

    axis_matrix = np.column_stack(oriented_axes) if oriented_axes else np.zeros((raw.shape[0], 0), dtype=float)
    refinement_axis_meta = [dict(row, axis_type="principal_axis") for row in axis_meta]
    # `prior` is already the mean of the consensus-oriented whitened axes.
    # Writing it directly avoids the algebraically equivalent but misleading
    # second averaging of `prior` with the same axes.
    readout = prior[:, 0].astype(np.float32).reshape(-1)
    axes_only = [axis_matrix[:, j] for j in range(axis_matrix.shape[1])]
    residual_complements = []
    if complement_indices:
        all_aligned = _rank_align(raw)
        admitted_directions = {
            int(row["index"]): float(row["residual_direction"])
            for row in (complement_meta or {}).get("candidate_evidence", [])
            if bool(row.get("admitted_within_budget", False))
        }
        for candidate_index in complement_indices:
            candidate = all_aligned[:, candidate_index]
            residual = admitted_residuals.get(candidate_index)
            if residual is None:
                _, residual, direction = _project_residual_complement(
                    all_aligned,
                    core_kept,
                    candidate_index,
                )
            else:
                direction = admitted_directions.get(candidate_index, 1.0)
            axes_only.append(residual)
            profile_names = [proxy_names[index] for index in gate_kept]
            profile = {
                proxy_names[index]: float(_safe_corr(residual, all_aligned[:, index]))
                for index in gate_kept
            }
            strongest_proxy = max(profile, key=lambda name: abs(profile[name]))
            refinement_axis_meta.append({
                "axis": len(axes_only),
                "axis_type": "residual_complement",
                "direction": direction,
                "strongest_proxy": strongest_proxy,
                "strongest_abs_corr": abs(profile[strongest_proxy]),
                "corr": profile,
                "oriented_corr": profile,
                "profile_proxy_order": profile_names,
            })
            residual_complements.append({
                "name": proxy_names[candidate_index],
                "axis": len(axes_only),
                "direction": direction,
                "correlation_to_proxy": float(_safe_corr(residual, candidate)),
            })
    cache_vectors = [readout] + axes_only

    meta = {
        "proxy_names": proxy_names,
        "apply_gate": bool(apply_gate),
        "cache_layout": "readout_axes",
        "factorization_device": "cpu",
        "gate_utility_raw": gate_utilities.tolist(),
        "gate_utility_reweighted": reweighted.tolist(),
        "gate_stage1_kept_names": [proxy_names[i] for i in core_kept],
        "gate_stage1_dropped_names": [proxy_names[i] for i in core_dropped],
        "backbone_selection": backbone_meta,
        "complement_admission_rule": complement_admission_rule,
        "complement_admission": complement_meta,
        "gate_kept_names": [proxy_names[i] for i in gate_kept],
        "gate_dropped_names": [proxy_names[i] for i in gate_dropped],
        "align": "rank",
        "transform": "rank_aligned_whitened_pc_axes",
        "shared_basis": "projected_columns",
        "axis_retention_rule": "retain_all_axes",
        "eigvals_desc": eigvals_desc.tolist(),
        "axis_rank_tolerance": float(axis_rank_tolerance),
        "n_axes": len(axes_only),
        "n_cache_metrics": len(cache_vectors),
        "orientation_proxy_order": orient_names,
        "oriented_axes": axis_meta,
        "refinement_axes": refinement_axis_meta,
        "residual_complement_axes": residual_complements,
        "readout": "mean of consensus-oriented whitened principal axes",
    }
    return cache_vectors, meta


def write_refinement_cache(path: Path, vectors: list[np.ndarray], cache_device: str) -> None:
    import torch

    cache = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=np.float32).reshape(2, 14, 7)
        cache.append([[torch.tensor(arr[0], device=cache_device), torch.tensor(arr[1], device=cache_device)]])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def build_proxy_set(
    op_root: Path,
    out_dir: Path,
    proxy_set_name: str,
    proxies: list[str],
    *,
    apply_gate: bool,
    proxy_budget: Optional[int] = None,
    cache_device: str,
) -> dict[str, object]:
    raw, input_meta = load_operation_scores(op_root, proxies)
    proxy_names = [f"zcpt_{_clean_name(p)}" for p in proxies]
    vectors, factor_meta = factorize_proxy_matrix(
        raw,
        proxy_names,
        apply_gate=apply_gate,
        proxy_budget=proxy_budget,
    )
    cache_path = out_dir / f"{proxy_set_name}_proxydiff_cache.pt"
    write_refinement_cache(cache_path, vectors, cache_device)
    details = {
        "proxy_set": proxy_set_name,
        "proxies": proxy_names,
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "output_cache_device": cache_device,
        "n_metrics": len(vectors),
        "metric0": "consensus_aligned_axis_mean",
        "correction_columns": [f"axis_{i}" for i in range(1, len(vectors))],
        "input_meta": input_meta,
        "factorization": factor_meta,
    }
    (out_dir / f"{proxy_set_name}_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def build_budgeted_proxy_set(
    op_root: Path,
    out_dir: Path,
    cache_device: str,
) -> dict[str, object]:
    raw, input_meta = load_operation_scores(op_root, SHORT_SCORE_PROXY_POOL)
    all_names = [f"zcpt_{name}" for name in SHORT_SCORE_PROXY_POOL]
    vectors, factor_meta = factorize_proxy_matrix(
        raw,
        all_names,
        apply_gate=True,
        proxy_budget=BUDGETED_PROXY_BUDGET,
        complement_admission_rule="repeated_structure_reliability",
    )
    cache_path = out_dir / "budgeted_proxy_subset_proxydiff_cache.pt"
    write_refinement_cache(cache_path, vectors, cache_device)
    details = {
        "proxy_set": "budgeted_proxy_subset",
        "proxies": factor_meta["gate_kept_names"],
        "candidate_pool": all_names,
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "output_cache_device": cache_device,
        "n_metrics": len(vectors),
        "metric0": "consensus_aligned_axis_mean",
        "correction_columns": [f"axis_{index}" for index in range(1, len(vectors))],
        "input_meta": input_meta,
        "selection": factor_meta["backbone_selection"],
        "factorization": factor_meta,
    }
    (out_dir / "budgeted_proxy_subset_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--proxy-set",
        choices=["full_proxy_pool", "budgeted_proxy_subset", "both"],
        default="both",
        help="Proxy set to convert into a ProxyDiff refinement cache.",
    )
    parser.add_argument(
        "--custom-proxy-set-name",
        default=None,
        help="Name for a fixed custom proxy subset cache.",
    )
    parser.add_argument(
        "--custom-proxies",
        default=None,
        help="Space-separated proxy names for a fixed custom subset.",
    )
    parser.add_argument(
        "--cache-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Tensor device used while writing score caches; auto uses CUDA when available.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_device = "cuda" if args.cache_device == "auto" and torch.cuda.is_available() else args.cache_device
    if cache_device == "cuda" and not torch.cuda.is_available():
        cache_device = "cpu"
    rows = []
    if args.custom_proxy_set_name or args.custom_proxies:
        if not args.custom_proxy_set_name or not args.custom_proxies:
            raise SystemExit("--custom-proxy-set-name and --custom-proxies must be provided together")
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                args.custom_proxy_set_name,
                args.custom_proxies.split(),
                apply_gate=False,
                cache_device=cache_device,
            )
        )
        manifest = {
            "description": "ProxyDiff NB301 custom fixed-subset cache generated from ZCPT operation-ablation scores",
            "op_root": str(args.op_root),
            "rows": rows,
        }
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"out_dir": str(args.out_dir), "proxy_sets": [r["proxy_set"] for r in rows]}, indent=2))
        return
    if args.proxy_set in {"full_proxy_pool", "both"}:
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                "full_proxy_pool",
                FULL_PROXY_POOL,
                apply_gate=True,
                proxy_budget=None,
                cache_device=cache_device,
            )
        )
    if args.proxy_set in {"budgeted_proxy_subset", "both"}:
        rows.append(build_budgeted_proxy_set(args.op_root, args.out_dir, cache_device))

    manifest = {
        "description": "ProxyDiff NB301 caches generated from ZCPT operation-ablation scores",
        "op_root": str(args.op_root),
        "rows": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "proxy_sets": [r["proxy_set"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
