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

COMPLEMENT_KEEP_RATIO = 0.9
COMPLEMENT_OPERATION_TOPK_COUNT = 5
REPEATED_STRUCTURE_CONFIDENCE_Z = 1.645
BUDGETED_PROXY_BUDGET = 3


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


def _utility_gap_gate(
    utilities: np.ndarray,
    min_keep: int = 3,
    allow_two_ratio: float = 2.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    utilities = np.nan_to_num(np.asarray(utilities, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(utilities.size)
    keep_min = max(1, min(int(min_keep), n))
    if n <= keep_min:
        kept = list(range(n))
        return np.ones(n, dtype=bool), utilities, kept, []

    centered = utilities - float(np.mean(utilities))
    scale = float(np.std(centered))
    if scale <= eps:
        kept = list(range(n))
        return np.ones(n, dtype=bool), utilities, kept, []

    reweighted = np.exp(np.clip(centered / scale, -20.0, 20.0))
    order = np.argsort(-reweighted, kind="mergesort")
    sorted_values = reweighted[order]
    gaps = sorted_values[:-1] - sorted_values[1:]
    if gaps.size == 0 or float(np.max(gaps)) <= eps:
        kept = list(range(n))
        return np.ones(n, dtype=bool), reweighted, kept, []

    cut = int(np.argmax(gaps)) + 1
    if keep_min >= 3 and n >= 3 and cut < keep_min:
        if cut == 2 and gaps.size >= 2:
            sorted_gaps = np.sort(gaps)[::-1]
            top_gap = float(sorted_gaps[0])
            second_gap = float(sorted_gaps[1])
            dominant_two = top_gap > eps and top_gap >= float(allow_two_ratio) * max(second_gap, eps)
            cut = 2 if dominant_two else keep_min
        else:
            cut = keep_min
    else:
        cut = max(cut, keep_min)

    kept = [int(i) for i in order[:cut]]
    dropped = [int(i) for i in order[cut:]]
    mask = np.zeros(n, dtype=bool)
    mask[kept] = True
    return mask, reweighted, kept, dropped


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


def _mean_operation_jaccard_drop(left: np.ndarray, right: np.ndarray) -> float:
    drops = []
    for cell in range(left.shape[0]):
        for edge in range(left.shape[1]):
            left_ops = set(np.flatnonzero(left[cell, edge]))
            right_ops = set(np.flatnonzero(right[cell, edge]))
            union = left_ops | right_ops
            drops.append(1.0 - len(left_ops & right_ops) / max(len(union), 1))
    return float(np.mean(drops))


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


def _factorized_readout(raw: np.ndarray, proxy_names: list[str]) -> np.ndarray:
    vectors, _ = factorize_proxy_matrix(
        raw,
        proxy_names,
        apply_gate=False,
    )
    return np.asarray(vectors[0], dtype=float)


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
        rank_efficiency = _effective_rank(aligned[:, indices]) / subset_size
        rows.append({
            "indices": indices,
            "names": names,
            "consensus_strength": consensus_strength,
            "rank_efficiency": rank_efficiency,
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
        "selection_rule": "maximize consensus strength times effective-rank efficiency",
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

    utilities = _proxy_utilities(_normalize_columns(raw))
    consensus_mask, reweighted, consensus_kept, consensus_dropped = _utility_gap_gate(
        utilities
    )
    backbone_capacity = min(int(proxy_budget), raw.shape[1])
    if len(consensus_kept) <= backbone_capacity:
        backbone_mask = consensus_mask
        selected_indices = consensus_kept
        compression = None
    else:
        selected_indices, compression = _consensus_rank_subset_gate(
            raw,
            proxy_names,
            subset_size=backbone_capacity,
        )
        backbone_mask = np.zeros(raw.shape[1], dtype=bool)
        backbone_mask[selected_indices] = True

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
        "compressed_to_budget": compression is not None,
        "compression": compression,
    }


def _repeated_structure_complement_gate(
    raw: np.ndarray,
    proxy_names: list[str],
    core_mask: np.ndarray,
    *,
    max_complements: int,
    complement_keep_ratio: float = COMPLEMENT_KEEP_RATIO,
    operation_topk_count: int = COMPLEMENT_OPERATION_TOPK_COUNT,
) -> tuple[np.ndarray, dict[str, object]]:
    """Admit novel residuals that are coherent across NB301 cell types."""
    core_indices = [index for index in range(raw.shape[1]) if bool(core_mask[index])]
    candidate_indices = [index for index in range(raw.shape[1]) if not bool(core_mask[index])]
    if not candidate_indices:
        return core_mask.copy(), {
            "complement_keep_ratio": float(complement_keep_ratio),
            "operation_topk_count": int(operation_topk_count),
            "candidate_evidence": [],
            "added_names": [],
        }

    aligned = _rank_align(raw)
    core_aligned = aligned[:, core_indices]
    core_rank = _effective_rank(core_aligned)
    core_basis = _orthonormal_basis(core_aligned)
    core_names = [proxy_names[index] for index in core_indices]
    core_readout = _factorized_readout(raw[:, core_indices], core_names)
    core_operation_mask = _progressive_operation_mask(core_readout, operation_topk_count)

    evidence = []
    for candidate_index in candidate_indices:
        joined_indices = sorted(core_indices + [candidate_index])
        joined_names = [proxy_names[index] for index in joined_indices]
        joined_readout = _factorized_readout(raw[:, joined_indices], joined_names)
        joined_operation_mask = _progressive_operation_mask(
            joined_readout,
            operation_topk_count,
        )
        decision_change = _mean_operation_jaccard_drop(
            core_operation_mask,
            joined_operation_mask,
        )
        changed_by_cell = np.asarray(
            [
                [
                    not np.array_equal(
                        core_operation_mask[cell, edge],
                        joined_operation_mask[cell, edge],
                    )
                    for edge in range(core_operation_mask.shape[1])
                ]
                for cell in range(core_operation_mask.shape[0])
            ],
            dtype=bool,
        )
        changes_both_cell_types = bool(
            changed_by_cell[0].any() and changed_by_cell[1].any()
        )
        rank_delta = max(
            0.0,
            _effective_rank(aligned[:, joined_indices]) - core_rank,
        )
        candidate = aligned[:, candidate_index]
        residual = candidate - core_basis @ (core_basis.T @ candidate)
        residual = _normalize_columns(residual.reshape(-1, 1))[:, 0]
        residual_cells = residual.reshape(2, 14, 7)
        global_cross_cell_correlation = _safe_corr(
            residual_cells[0].reshape(-1),
            residual_cells[1].reshape(-1),
        )
        paired_edge_correlations = [
            _safe_corr(residual_cells[0, edge], residual_cells[1, edge])
            for edge in range(residual_cells.shape[1])
        ]
        median_edge_correlation = float(np.median(paired_edge_correlations))
        repeated_structure_reliability, correlation_lower_bound = (
            _conservative_correlation_reliability(
                global_cross_cell_correlation,
                residual_cells[0].size,
            )
        )
        if (
            rank_delta > 0.0
            and decision_change > 0.0
            and repeated_structure_reliability > 0.0
            and changes_both_cell_types
        ):
            complement_score = (
                rank_delta
                * decision_change
                * repeated_structure_reliability
            )
        else:
            complement_score = 0.0
        evidence.append({
            "index": candidate_index,
            "name": proxy_names[candidate_index],
            "effective_rank_delta": rank_delta,
            "mean_operation_jaccard_drop": decision_change,
            "changes_both_cell_types": changes_both_cell_types,
            "global_cross_cell_spearman": global_cross_cell_correlation,
            "paired_edge_spearman": paired_edge_correlations,
            "median_paired_edge_spearman": median_edge_correlation,
            "cross_cell_correlation_lower_bound": correlation_lower_bound,
            "repeated_structure_reliability": repeated_structure_reliability,
            "complement_score": complement_score,
        })

    evidence.sort(key=lambda row: (-float(row["complement_score"]), row["name"]))
    strongest = max(
        (float(row["complement_score"]) for row in evidence),
        default=0.0,
    )
    mask = core_mask.copy()
    added_names = []
    for row in evidence:
        normalized = (
            float(row["complement_score"]) / strongest
            if strongest > 1e-12
            else 0.0
        )
        passes = (
            float(row["complement_score"]) > 0.0
            and normalized >= float(complement_keep_ratio)
        )
        admitted = passes and len(added_names) < max(0, int(max_complements))
        row["normalized_complement_score"] = normalized
        row["passes_relative_threshold"] = bool(passes)
        row["admitted_within_budget"] = bool(admitted)
        row["passes_threshold"] = bool(admitted)
        if admitted:
            mask[int(row["index"])] = True
            added_names.append(str(row["name"]))
    return mask, {
        "selection_rule": (
            "effective-rank gain x mean operation Jaccard change "
            "x conservative repeated-structure reliability"
        ),
        "repeated_structure_reliability": (
            "squared positive one-sided 95% Fisher lower bound of the "
            "normal/reduction cell Spearman correlation"
        ),
        "complement_keep_ratio": float(complement_keep_ratio),
        "admission_capacity": max(0, int(max_complements)),
        "operation_topk_count": int(operation_topk_count),
        "core_effective_rank": core_rank,
        "candidate_evidence": evidence,
        "added_names": added_names,
        "abstained": not added_names,
    }


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
        core_kept = [index for index in range(raw.shape[1]) if bool(core_mask[index])]
        core_dropped = [index for index in range(raw.shape[1]) if not bool(core_mask[index])]
        if complement_admission_rule == "repeated_structure_reliability":
            mask, complement_meta = _repeated_structure_complement_gate(
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
        retained = raw[:, core_mask]
        retained_names = [
            proxy_names[index] for index in range(len(proxy_names)) if bool(core_mask[index])
        ]
    else:
        reweighted = gate_utilities
        backbone_meta = None
        core_kept = list(range(raw.shape[1]))
        core_dropped = []
        complement_meta = None
        complement_indices = []
        gate_kept = list(range(raw.shape[1]))
        gate_dropped = []
        retained = raw
        retained_names = list(proxy_names)

    z = _rank_align(retained)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = eigvals[::-1]
    pc_all = eigvecs[:, ::-1]
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
    denom = 1.0 + float(axis_matrix.shape[1])
    prior_tensor = torch.tensor(prior[:, 0].astype(np.float32).reshape(2, 14, 7), device="cpu")
    readout_cells = [prior_tensor[cell].clone() for cell in range(prior_tensor.shape[0])]
    axis_cells = []
    for j, axis in enumerate(pre_oriented_axes):
        axis_tensor = torch.tensor(axis.astype(np.float32).reshape(2, 14, 7), device="cpu")
        direction = float(axis_meta[j]["direction"])
        cells = [axis_tensor[cell].clone() for cell in range(axis_tensor.shape[0])]
        for cell in range(len(cells)):
            cells[cell] = cells[cell] * direction
        axis_cells.append(cells)
    for cell in range(len(readout_cells)):
        readout_cells[cell] = readout_cells[cell] / denom
    for cells in axis_cells:
        for cell in range(len(readout_cells)):
            readout_cells[cell] = readout_cells[cell] + cells[cell] / denom
    readout = torch.stack(readout_cells, dim=0).detach().cpu().numpy().reshape(-1)
    axes_only = [axis_matrix[:, j] for j in range(axis_matrix.shape[1])]
    residual_complements = []
    if complement_indices:
        all_aligned = _rank_align(raw)
        core_aligned = all_aligned[:, core_kept]
        core_basis = _orthonormal_basis(core_aligned)
        for candidate_index in complement_indices:
            candidate = all_aligned[:, candidate_index]
            residual = candidate - core_basis @ (core_basis.T @ candidate)
            residual = _normalize_columns(residual.reshape(-1, 1))[:, 0]
            direction = 1.0 if _safe_corr(residual, candidate) >= 0.0 else -1.0
            residual = residual * direction
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
        "n_axes": len(axes_only),
        "n_cache_metrics": len(cache_vectors),
        "orientation_proxy_order": orient_names,
        "oriented_axes": axis_meta,
        "refinement_axes": refinement_axis_meta,
        "residual_complement_axes": residual_complements,
        "readout": "uniform average over factorized prior and profile-oriented axes",
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
        "output_cache_device": cache_device,
        "n_metrics": len(vectors),
        "metric0": "uniform_axis_readout",
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
        "output_cache_device": cache_device,
        "n_metrics": len(vectors),
        "metric0": "uniform_axis_readout",
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
