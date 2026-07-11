#!/usr/bin/env python3
"""Run the controlled NB301 evolution baseline on operation-level proxies.

The search space, mutation/crossover schedule, and population-relative rank
aggregation match the paper's paired AZ-style and rank-mean controls. Only the
proxy subset and aggregation rule vary between controlled comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch


PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation-score-root", type=Path, required=True)
    parser.add_argument("--proxy-names", required=True, help="Comma-separated proxy names")
    parser.add_argument("--aggregation", choices=["log_rank", "mean_rank"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nas-runtime-root", type=Path, required=True)
    parser.add_argument("--fixed-architecture-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--parent-count", type=int, default=10)
    parser.add_argument("--generation-count", type=int, default=5)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-count", type=int, default=25)
    parser.add_argument("--crossover-count", type=int, default=25)
    args = parser.parse_args()
    if args.seed != 9000:
        parser.error("paper reproduction requires --seed 9000")
    if not 1 <= args.parent_count <= args.population_size:
        parser.error("parent count must be in [1, population size]")
    if args.mutation_count + args.crossover_count > args.population_size:
        parser.error("mutation and crossover counts exceed population size")
    return args


def clean_proxy_name(name: str) -> str:
    name = name.strip()
    return name[5:] if name.startswith("zcpt_") else name


def architecture_key(architecture):
    return tuple(tuple((int(source), int(operation)) for source, operation in cell) for cell in architecture)


def architecture_from_key(key):
    return [[(int(source), int(operation)) for source, operation in cell] for cell in key]


def random_cell(rng: random.Random):
    cell = []
    for node in range(4):
        for source in rng.sample(range(node + 2), 2):
            cell.append((source, rng.randrange(len(PRIMITIVES))))
    return cell


def random_architecture(rng: random.Random):
    return [random_cell(rng), random_cell(rng)]


def mutate_cell(cell, rng: random.Random, mutation_rate: float):
    output = []
    cursor = 0
    for node in range(4):
        edges = list(cell[cursor : cursor + 2])
        cursor += 2
        if rng.random() < mutation_rate:
            edge_index = rng.randrange(2)
            used_sources = {source for index, (source, _) in enumerate(edges) if index != edge_index}
            choices = [source for source in range(node + 2) if source not in used_sources]
            _, operation = edges[edge_index]
            edges[edge_index] = (rng.choice(choices), operation)
        if rng.random() < mutation_rate:
            edge_index = rng.randrange(2)
            source, _ = edges[edge_index]
            edges[edge_index] = (source, rng.randrange(len(PRIMITIVES)))
        output.extend(edges)
    return output


def mutate_architecture(parent, rng: random.Random, mutation_rate: float):
    return [mutate_cell(parent[0], rng, mutation_rate), mutate_cell(parent[1], rng, mutation_rate)]


def crossover_architecture(first, second, rng: random.Random):
    child = []
    for cell_index in range(2):
        cell = []
        for node in range(4):
            start = 2 * node
            source_cell = first[cell_index] if rng.random() < 0.5 else second[cell_index]
            cell.extend(source_cell[start : start + 2])
        child.append(cell)
    return child


def to_genotype(architecture):
    def convert(cell):
        return [(PRIMITIVES[int(operation)], int(source)) for source, operation in cell]

    return Genotype(
        normal=convert(architecture[0]),
        normal_concat=[2, 3, 4, 5],
        reduce=convert(architecture[1]),
        reduce_concat=[2, 3, 4, 5],
    )


def sanitize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = float(np.min(finite)), float(np.max(finite))
    epsilon = max(abs(low), abs(high), 1.0) * 1e-6
    values[np.isnan(values) | np.isneginf(values)] = low - epsilon
    values[np.isposinf(values)] = high + epsilon
    return values


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = sanitize(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return ranks


def operation_score_path(root: Path, proxy_name: str) -> Path:
    candidates = [
        root / f"nb301_zcpt_{proxy_name}" / "operation_scores.json",
        root / f"nb301_official_zcpt_{proxy_name}" / "operation_scores.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing operation scores for {proxy_name}: {candidates}")


def load_operation_tensor(root: Path, proxy_name: str):
    path = operation_score_path(root, proxy_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tensor = np.full((2, 14, 7), np.nan, dtype=float)
    cell_index = {"normal": 0, "reduce": 1}
    for row in payload["records"]:
        value = row.get("operation_score")
        tensor[cell_index[str(row["cell_type"])], int(row["edge_id"]), int(row["op_id"])] = (
            float("nan") if value is None else float(value)
        )
    if np.isnan(tensor).all(axis=None):
        raise ValueError(f"{path} contains no finite operation score")
    tensor = sanitize(tensor.reshape(-1)).reshape(2, 14, 7)
    if not np.isfinite(tensor).all():
        raise ValueError(f"failed to sanitize {path}")
    metadata = {
        "path": str(path),
        "protocol_revision": payload.get("protocol_revision"),
        "predictor": payload.get("predictor"),
        "batch_size": payload.get("batch_size"),
    }
    return tensor, metadata


def component_indices(architecture):
    edge_starts = [0, 2, 5, 9]
    indices = []
    for cell_index, cell in enumerate(architecture):
        for item_index, (source, operation) in enumerate(cell):
            node = item_index // 2
            indices.append((cell_index, edge_starts[node] + int(source), int(operation)))
    return indices


def proxy_columns(proxy_tensors, architectures):
    return [
        np.asarray([sum(float(tensor[index]) for index in component_indices(arch)) for arch in architectures])
        for _, tensor in proxy_tensors
    ]


def aggregate_proxy_columns(columns, aggregation: str) -> np.ndarray:
    rank_matrix = np.vstack([average_ranks(column) for column in columns])
    if aggregation == "log_rank":
        return np.log(np.maximum(rank_matrix, 1.0) / rank_matrix.shape[1]).sum(axis=0)
    return -rank_matrix.mean(axis=0)


def score_cache(cache, proxy_tensors, aggregation: str) -> None:
    architectures = [item["architecture"] for item in cache.values()]
    scores = aggregate_proxy_columns(proxy_columns(proxy_tensors, architectures), aggregation)
    for item, score in zip(cache.values(), scores):
        item["score"] = float(score)


def finite_score(item) -> float:
    score = float(item["score"])
    return score if math.isfinite(score) else float("-inf")


def load_nb301_model(runtime_root: Path):
    package_root = runtime_root.parent if runtime_root.name == "ZeroCostNAS" else runtime_root
    os.environ["NAS_RUNTIME_PACKAGE_ROOT"] = str(package_root)
    evaluator_source = Path(__file__).resolve().parents[1] / "nas_v2" / "src"
    sys.path.insert(0, str(evaluator_source))
    from reevaluate_free_selected_arch import get_nasbench301_api

    return get_nasbench301_api("cifar10")["nb301_model"][0]


def main() -> None:
    args = parse_args()
    proxy_names = [clean_proxy_name(name) for name in args.proxy_names.split(",") if name.strip()]
    if not proxy_names:
        raise ValueError("at least one proxy is required")
    proxy_tensors = []
    input_metadata = {}
    for name in proxy_names:
        tensor, metadata = load_operation_tensor(args.operation_score_root, name)
        proxy_tensors.append((name, tensor))
        input_metadata[name] = metadata

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    cache = {}
    population = [random_architecture(rng) for _ in range(args.population_size)]
    history = []
    started = time.time()

    for generation in range(args.generation_count):
        for architecture in population:
            key = architecture_key(architecture)
            cache.setdefault(key, {"architecture": architecture_from_key(key), "score": float("nan")})
        score_cache(cache, proxy_tensors, args.aggregation)
        ranked = sorted(cache.values(), key=finite_score, reverse=True)
        parents = [item["architecture"] for item in ranked[: args.parent_count]]
        history.append(
            {
                "generation": generation,
                "evaluated_unique": len(cache),
                "best_score": float(ranked[0]["score"]),
                "best_architecture": ranked[0]["architecture"],
            }
        )
        print(
            f"generation={generation} unique={len(cache)} best_score={ranked[0]['score']}",
            flush=True,
        )
        if generation + 1 == args.generation_count:
            break
        next_population = []
        while len(next_population) < args.crossover_count:
            first, second = rng.sample(parents, 2)
            next_population.append(crossover_architecture(first, second, rng))
        while len(next_population) < args.crossover_count + args.mutation_count:
            next_population.append(mutate_architecture(rng.choice(parents), rng, args.mutation_rate))
        while len(next_population) < args.population_size:
            next_population.append(random_architecture(rng))
        population = next_population

    score_cache(cache, proxy_tensors, args.aggregation)
    ranked = sorted(cache.values(), key=finite_score, reverse=True)
    selected = ranked[0]
    model = load_nb301_model(args.nas_runtime_root)
    genotype = to_genotype(selected["architecture"])
    selected_accuracy = float(model.predict(config=genotype, representation="genotype", with_noise=False))
    pool_payload = torch.load(args.fixed_architecture_file, map_location="cpu")
    fixed_pool_accuracies = [float(value) for value in pool_payload[1]]
    fixed_pool_rank = 1 + sum(value > selected_accuracy for value in fixed_pool_accuracies)

    result = {
        "definition": "NB301 evolution with population-relative operation-proxy rank aggregation",
        "seed": args.seed,
        "operation_score_root": str(args.operation_score_root),
        "proxy_names": proxy_names,
        "aggregation": args.aggregation,
        "population_size": args.population_size,
        "parent_count": args.parent_count,
        "generation_count": args.generation_count,
        "mutation_rate": args.mutation_rate,
        "mutation_count": args.mutation_count,
        "crossover_count": args.crossover_count,
        "evaluated_unique": len(cache),
        "wall_time_seconds": time.time() - started,
        "selected_architecture": selected["architecture"],
        "selected_genotype": repr(genotype),
        "selected_score": float(selected["score"]),
        "selected_accuracy": selected_accuracy,
        "selected_fixed_pool_rank": fixed_pool_rank,
        "input_metadata": input_metadata,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "search_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
