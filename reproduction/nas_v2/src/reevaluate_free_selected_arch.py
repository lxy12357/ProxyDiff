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


def install_configspace_normal_bounds_compat() -> None:
    """Allow NASBench-301 ConfigSpace JSON files to load on newer ConfigSpace."""
    try:
        from ConfigSpace.read_and_write import dictionary as cs_dictionary
    except Exception:
        return

    float_decoder = getattr(cs_dictionary, "_decode_normal_float", None)
    int_decoder = getattr(cs_dictionary, "_decode_normal_int", None)
    if float_decoder is None or getattr(float_decoder, "_proxydiff_normal_bounds_compat", False):
        return

    def _decode_normal_float_compat(item, cs, dec):
        if "lower" not in item or "upper" not in item:
            item = dict(item)
            mu = float(item.get("mu", 0.0))
            sigma = abs(float(item.get("sigma", 1.0)))
            if item.get("log", False):
                item.setdefault("lower", max(mu * 1e-3, 1e-12))
                item.setdefault("upper", max(mu * 1e3, mu + 10.0 * sigma, 1.0))
            else:
                item.setdefault("lower", mu - 10.0 * sigma)
                item.setdefault("upper", mu + 10.0 * sigma)
        return float_decoder(item, cs, dec)

    def _decode_normal_int_compat(item, cs, dec):
        if "lower" not in item or "upper" not in item:
            item = dict(item)
            mu = int(round(float(item.get("mu", 0))))
            sigma = max(1, int(round(abs(float(item.get("sigma", 1))))))
            item.setdefault("lower", mu - 10 * sigma)
            item.setdefault("upper", mu + 10 * sigma)
        return int_decoder(item, cs, dec)

    _decode_normal_float_compat._proxydiff_normal_bounds_compat = True
    _decode_normal_int_compat._proxydiff_normal_bounds_compat = True
    cs_dictionary._decode_normal_float = _decode_normal_float_compat
    if int_decoder is not None:
        cs_dictionary._decode_normal_int = _decode_normal_int_compat
    if hasattr(cs_dictionary, "HYPERPARAMETER_DECODERS"):
        cs_dictionary.HYPERPARAMETER_DECODERS["normal_float"] = _decode_normal_float_compat
        if int_decoder is not None:
            cs_dictionary.HYPERPARAMETER_DECODERS["normal_int"] = _decode_normal_int_compat


install_configspace_normal_bounds_compat()

NAS_RUNTIME_PACKAGE_ROOT = os.environ.get("NAS_RUNTIME_PACKAGE_ROOT")
if not NAS_RUNTIME_PACKAGE_ROOT:
    raise RuntimeError(
        "NAS_RUNTIME_PACKAGE_ROOT must point to the parent directory of the "
        "ZeroCostNAS runtime package."
    )
sys.path.insert(0, NAS_RUNTIME_PACKAGE_ROOT)


def install_nasbench301_configloader_compat() -> None:
    """Patch NASBench-301 ConfigSpace internal-field access."""
    try:
        from ConfigSpace import hyperparameters as CSH
        from nasbench301.surrogate_models import utils as nb301_utils
    except Exception:
        return

    original = getattr(nb301_utils.ConfigLoader, "load_config_space", None)
    if original is None or getattr(original, "_proxydiff_configspace_compat", False):
        return

    def _drop_hyperparameter(config_space, name: str) -> None:
        internal_mapping = getattr(config_space, "_hyperparameters", None)
        if hasattr(internal_mapping, "pop") and internal_mapping is not config_space:
            internal_mapping.pop(name, None)
            return

        dag = getattr(config_space, "_dag", None)
        if dag is None or name not in getattr(dag, "nodes", {}):
            return
        for mapping_name in (
            "nodes",
            "roots",
            "non_roots",
            "index_of",
            "children_of",
            "parents_of",
            "child_conditions_of",
            "parent_conditions_of",
        ):
            mapping = getattr(dag, mapping_name, None)
            if hasattr(mapping, "pop"):
                mapping.pop(name, None)
        dag.at = [hp_name for hp_name in getattr(dag, "at", []) if hp_name != name]
        dag.hyperparameters = [hp for hp in getattr(dag, "hyperparameters", []) if hp.name != name]
        dag.index_of = {hp_name: idx for idx, hp_name in enumerate(dag.at)}
        for idx, hp_name in enumerate(dag.at):
            if hp_name in dag.nodes:
                dag.nodes[hp_name].idx = idx
        config_space._len = len(dag.at)

    @staticmethod
    def _load_config_space_compat(path):
        with open(path, "r") as fh:
            config_space = nb301_utils.config_space_json_r_w.read(fh.read())

        replacements = [
            CSH.UniformIntegerHyperparameter(
                name="NetworkSelectorDatasetInfo:darts:layers", lower=1, upper=10000
            ),
            CSH.UniformIntegerHyperparameter(
                name="SimpleLearningrateSchedulerSelector:cosine_annealing:T_max",
                lower=1,
                upper=10000,
            ),
            CSH.UniformIntegerHyperparameter(
                name="NetworkSelectorDatasetInfo:darts:init_channels", lower=1, upper=10000
            ),
            CSH.UniformFloatHyperparameter(
                name="SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min",
                lower=0,
                upper=10000,
            ),
        ]
        for hp in replacements:
            _drop_hyperparameter(config_space, hp.name)
        config_space.add_hyperparameters(replacements)
        return config_space

    _load_config_space_compat._proxydiff_configspace_compat = True
    nb301_utils.ConfigLoader.load_config_space = _load_config_space_compat


install_nasbench301_configloader_compat()

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
    ap.add_argument(
        "--fixed_arch",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "arch_dataset_20cell_c36.pt"),
    )
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
