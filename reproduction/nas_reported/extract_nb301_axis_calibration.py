#!/usr/bin/env python3
"""Export NB301 axis profiles and calibrated axis coefficients."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def flatten_axis_parameters(value: object) -> list[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and torch.is_tensor(value[0]) and value[0].numel() > 1:
            return flatten_axis_parameters(value[0])
        output = []
        for item in value:
            flattened = flatten_axis_parameters(item)
            if flattened:
                output.append(flattened[0])
        return output
    return [float(value)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorization-details", type=Path, required=True)
    parser.add_argument("--axis-artifact", type=Path, required=True)
    parser.add_argument("--residual-axis-scale", type=float, default=1.0)
    parser.add_argument("--out-weights-csv", type=Path, required=True)
    parser.add_argument("--out-profiles-csv", type=Path, required=True)
    args = parser.parse_args()

    details = json.loads(args.factorization_details.read_text(encoding="utf-8"))
    factorization = details.get("factorization", details)
    artifact = torch.load(args.axis_artifact, map_location="cpu")
    parameters = flatten_axis_parameters(artifact.get("axis_calibration_weights"))

    axes = factorization["oriented_axes"]
    if len(parameters) < len(axes) + 1:
        raise RuntimeError(
            f"axis artifact contains {len(parameters)} parameters, but {len(axes) + 1} are required"
        )

    weight_rows = [
        {
            "coordinate": "base_readout",
            "raw_parameter": parameters[0],
            "effective_coefficient": 1.0,
            "strongest_proxy": "",
            "strongest_abs_corr": "",
        }
    ]
    profile_rows = []
    for index, axis in enumerate(axes, start=1):
        raw_parameter = parameters[index]
        weight_rows.append(
            {
                "coordinate": f"axis_{index}",
                "raw_parameter": raw_parameter,
                "effective_coefficient": args.residual_axis_scale * math.tanh(raw_parameter),
                "strongest_proxy": axis["strongest_proxy"],
                "strongest_abs_corr": axis["strongest_abs_corr"],
            }
        )
        for proxy_name, correlation in axis["oriented_corr"].items():
            profile_rows.append(
                {
                    "coordinate": f"axis_{index}",
                    "proxy": proxy_name,
                    "oriented_spearman": correlation,
                    "abs_spearman": abs(float(correlation)),
                }
            )

    write_csv(args.out_weights_csv, weight_rows)
    write_csv(args.out_profiles_csv, profile_rows)
    print(f"weights={args.out_weights_csv}")
    print(f"profiles={args.out_profiles_csv}")


if __name__ == "__main__":
    main()
