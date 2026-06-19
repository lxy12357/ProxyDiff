#!/usr/bin/env python3
"""Re-evaluate ProxyDiff refinement artifacts by their free discrete NB301 architecture.

This is different from fixed-pool candidate selection.  It uses
the saved cell/edge/op score matrix, discretizes an NB301 architecture from
that matrix, queries the selected architecture accuracy, and then maps that
accuracy to an equivalent rank within the fixed1000 pool.

Discretization maximizes the same additive objective used by the external NAS
runtime evaluator, namely the sum of
`score[cell][edge][op]` over the selected NB301 operations.  Under the legal
NB301/DARTS cell constraint, each intermediate node independently chooses two
distinct incoming sources and one operation on each source.  Because the score
is additive, enumerating legal pairs per node and taking the best pair per node
gives the exact global best free architecture; a small beam is used only to
report the top-k alternatives.
"""

import argparse
import glob
import itertools
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

NAS_RUNTIME_PACKAGE_ROOT = os.environ.get(
    "NAS_RUNTIME_PACKAGE_ROOT",
    "/hdd/xiaoyun/ProxyDARTS/Reproduction/nas_runtime",
)
REPRO_ROOT = os.environ.get("REPRO", "/hdd/xiaoyun/ProxyDARTS/Reproduction")
sys.path.insert(0, NAS_RUNTIME_PACKAGE_ROOT)

from ZeroCostNAS.utils.get_dataset_api import get_nasbench301_api  # noqa: E402


DartsGenotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
OP_NAMES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


EDGE_TO_INOUT = [
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),
    (0, 5),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
]


def node_of_edge(edge_idx: int) -> int:
    end = EDGE_TO_INOUT[edge_idx][1]
    return end - 2


def arch_key_from_triples(arch3):
    cells = []
    for cell in arch3:
        cells.append(tuple((int(start), int(op)) for start, _end, op in cell))
    return tuple(cells)


def genotype_from_arch_key(arch_key):
    normal = [(OP_NAMES[int(op)], int(start)) for start, op in arch_key[0]]
    reduce = [(OP_NAMES[int(op)], int(start)) for start, op in arch_key[1]]
    return DartsGenotype(normal=normal, normal_concat=[2, 3, 4, 5], reduce=reduce, reduce_concat=[2, 3, 4, 5])


def fixed1000_rank(acc: float, fixed_acc: np.ndarray) -> int:
    return int(1 + np.sum(fixed_acc > float(acc)))


def score_arch(arch3, score_np: np.ndarray) -> float:
    total = 0.0
    for cell_idx, cell in enumerate(arch3):
        for start, end, op in cell:
            edge = EDGE_TO_INOUT.index((int(start), int(end)))
            total += float(score_np[cell_idx, edge, int(op)])
    return total


def legal_node_pairs(score_np: np.ndarray, cell_idx: int, node_idx: int):
    entries = []
    for edge_idx, (start, end) in enumerate(EDGE_TO_INOUT):
        if node_of_edge(edge_idx) != node_idx:
            continue
        for op in range(score_np.shape[2]):
            entries.append(((int(start), int(end), int(op)), float(score_np[cell_idx, edge_idx, op])))
    node_pairs = []
    for first, second in itertools.combinations(entries, 2):
        first_triple, first_score = first
        second_triple, second_score = second
        if first_triple[0] == second_triple[0]:
            continue
        node_pairs.append((first, second, float(first_score + second_score)))
    node_pairs.sort(key=lambda item: item[2], reverse=True)
    if not node_pairs:
        raise RuntimeError(f"no valid distinct-source pairs for cell={cell_idx} node={node_idx}")
    return node_pairs


def top_arch_candidates(score_np: np.ndarray, top_k: int = 10):
    per_node_options = []
    for cell_idx in range(2):
        for node_idx in range(4):
            per_node_options.append(legal_node_pairs(score_np, cell_idx, node_idx))

    beam = [(tuple(), 0.0)]
    keep = max(1, int(top_k))
    for node_pairs in per_node_options:
        next_beam = []
        for partial, partial_score in beam:
            for pair in node_pairs:
                next_beam.append((partial + (pair,), float(partial_score + pair[2])))
        next_beam.sort(key=lambda item: item[1], reverse=True)
        beam = next_beam[:keep]

    candidates = []
    for comb, beam_score in beam:
        arch = [[], []]
        for block_idx, pair_with_score in enumerate(comb):
            cell_idx = 0 if block_idx < 4 else 1
            pair = pair_with_score[:2]
            for triple, value in pair:
                arch[cell_idx].append(triple)
        # Keep original local ordering by target node and then edge/op.
        arch = tuple(tuple(cell) for cell in arch)
        candidates.append({"arch3": arch, "arch_key": arch_key_from_triples(arch), "score": float(beam_score)})
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates


def query_acc(model, arch_key):
    genotype = genotype_from_arch_key(arch_key)
    return float(model.predict(config=genotype, representation="genotype", with_noise=False))


def method_name_from_path(path: Path) -> str:
    name = path.name
    legacy_token = "rank{}".format(1 + 1)
    legacy_epoch = f"{legacy_token}_epoch_"
    if f"__{legacy_epoch}" in name and name.endswith("_score_params.pt"):
        prefix, rest = name.split(f"__{legacy_epoch}", 1)
        epoch = rest.replace("_score_params.pt", "")
        return f"{prefix}__{legacy_epoch}{epoch}"
    if name.startswith(legacy_epoch) and name.endswith("_score_params.pt"):
        return name.replace("_score_params.pt", "")
    return path.stem


def summarize_artifact(path: Path, model, fixed_acc: np.ndarray, top_k: int):
    obj = torch.load(path, map_location="cpu")
    score = obj["score"]
    if isinstance(score, list):
        score = torch.stack([torch.as_tensor(s) for s in score], dim=0)
    score_np = score.detach().cpu().numpy().astype(float)
    candidates = top_arch_candidates(score_np, top_k=top_k)
    top = candidates[0]
    top_acc = query_acc(model, top["arch_key"])
    top_out = {
        "arch3": repr(top["arch3"]),
        "arch_key": repr(top["arch_key"]),
        "genotype": repr(genotype_from_arch_key(top["arch_key"])),
        "score": top["score"],
        "acc": top_acc,
        "equiv_fixed1000_rank": fixed1000_rank(top_acc, fixed_acc),
    }

    top_by_method_score = []
    for cand in candidates[:10]:
        acc = query_acc(model, cand["arch_key"])
        top_by_method_score.append(
            {
                "arch3": repr(cand["arch3"]),
                "arch_key": repr(cand["arch_key"]),
                "genotype": repr(genotype_from_arch_key(cand["arch_key"])),
                "score": cand["score"],
                "acc": acc,
                "equiv_fixed1000_rank": fixed1000_rank(acc, fixed_acc),
            }
        )

    return {
        "method": method_name_from_path(path),
        "metric_epoch": obj.get("metric_epoch", None),
        "artifact": str(path),
        "free_top_score_arch": top_out,
        "top10_by_method_score": top_by_method_score,
        "n_reported_candidates": len(candidates),
        "decode": "exact additive optimum with legal distinct-source pairs per NB301 node",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_glob", default="/tmp/proxydiff_refinement_reeval/*score_params.pt")
    ap.add_argument("--fixed_arch", default=os.path.join(REPRO_ROOT, "fixed_archs", "arch_dataset_20cell_c36.pt"))
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    fixed = torch.load(args.fixed_arch, map_location="cpu")
    fixed_acc = np.asarray(fixed[1], dtype=float)
    model = get_nasbench301_api("cifar10")["nb301_model"][0]

    rows = [
        summarize_artifact(Path(path), model, fixed_acc, top_k=args.top_k)
        for path in sorted(glob.glob(args.artifact_glob))
    ]
    fixed_best_idx = int(np.argmax(fixed_acc))
    fixed_best_arch = fixed[0][fixed_best_idx]
    fixed_best_live_acc = query_acc(model, fixed_best_arch)
    result = {
        "definition": "free selected arch from saved cell-edge-op scores; equivalent rank = 1 + count(fixed1000_acc > selected_arch_acc)",
        "fixed_arch": args.fixed_arch,
        "fixed1000_best_acc": float(np.max(fixed_acc)),
        "fixed1000_best_index": fixed_best_idx,
        "fixed1000_best_live_surrogate_acc_sanity": fixed_best_live_acc,
        "fixed1000_best_live_abs_diff": abs(float(np.max(fixed_acc)) - fixed_best_live_acc),
        "fixed1000_num": int(len(fixed_acc)),
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
