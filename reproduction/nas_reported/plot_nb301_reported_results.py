#!/usr/bin/env python3
"""Render the NB301 paper panels from clean reproduction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = (55, 105, 170)
RED = (196, 82, 74)
PURPLE = (117, 100, 160)
ORANGE = (221, 143, 62)
GREEN = (74, 145, 111)
DARK = (0, 0, 0)
MID = (95, 100, 105)
GRID = (222, 226, 230)
PALE = (247, 248, 250)


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F7 = font(19)
F8 = font(22)
F8B = font(22, True)
F9 = font(25)
F9B = font(25, True)
F10 = font(28)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reported-summary", type=Path, required=True)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--full-decode", type=Path, required=True)
    parser.add_argument("--axis-weights", type=Path, required=True)
    parser.add_argument("--axis-profiles", type=Path, required=True)
    parser.add_argument("--subset-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def text_center(draw, xy, value, text_font, fill=DARK):
    box = draw.textbbox((0, 0), value, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), value, font=text_font, fill=fill)


def save(image, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / name
    image.save(output, dpi=(300, 300))
    print(output)


def read_reported(path: Path):
    values = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values[(row["group"], row["name"])] = row["value"]
    return values


def read_main(path: Path):
    with path.open(encoding="utf-8") as stream:
        rows = {row["row"]: row for row in csv.DictReader(stream)}
    row = rows["full_proxy_pool"]
    return {
        "prior": float(row["prior_acc"]),
        "axis": float(row["axis_acc"]),
        "final": float(row["refinement_acc"]),
    }


def draw_check(draw, center_x, center_y):
    draw.line((center_x - 8, center_y, center_x - 2, center_y + 7), fill=DARK, width=2)
    draw.line((center_x - 2, center_y + 7, center_x + 10, center_y - 8), fill=DARK, width=2)


def component_ablation(output_dir: Path, reported):
    items = [
        ("Single proxy", float(reported[("component_ablation", "best_single_proxy_acc")]), [0, 0, 0]),
        ("Multi-proxy", float(reported[("component_ablation", "raw_multi_proxy_direct_acc")]), [1, 0, 0]),
        (
            "Multi-proxy + task-conditioned",
            float(reported[("component_ablation", "raw_multi_proxy_task_conditioned_acc")]),
            [1, 0, 1],
        ),
        ("ProxyDiff", float(reported[("component_ablation", "factorized_full_acc")]), [1, 1, 1]),
    ]
    base = items[0][1]
    image = Image.new("RGB", (840, 430), "white")
    draw = ImageDraw.Draw(image)
    column_x = [250, 385, 530]
    x0, x1 = 610, 800
    y0, row_height = 140, 55
    minimum = math.floor((min(item[1] for item in items) - 0.05) * 10) / 10
    maximum = math.ceil((max(item[1] for item in items) + 0.05) * 10) / 10
    headers = ["Multi-\nproxy", "Proxy\nDisentanglement", "Task-\nconditioned\nrefinement"]
    for center_x, header in zip(column_x, headers):
        for index, part in enumerate(header.split("\n")):
            text_center(draw, (center_x, 28 + index * 20), part, F7)
    for tick in [value for value in (94.0, 94.4, 94.8) if minimum <= value <= maximum]:
        x = x0 + (tick - minimum) / (maximum - minimum) * (x1 - x0)
        draw.text((x - 24, y0 + row_height * len(items) + 8), f"{tick:.1f}", font=F7, fill=DARK)
    draw.line((x0, y0 + row_height * len(items) + 2, x1, y0 + row_height * len(items) + 2), fill=DARK, width=2)
    draw.text((x0 + 30, y0 + row_height * len(items) + 42), "surrogate acc.", font=F8, fill=DARK)
    base_x = x0 + (base - minimum) / (maximum - minimum) * (x1 - x0)
    draw.line((base_x, y0 - 45, base_x, y0 + row_height * len(items) + 2), fill=DARK, width=2)
    draw.text((base_x - 105, y0 - 60), "best single", font=F7, fill=DARK)
    for index, (label, accuracy, checks) in enumerate(items):
        y = y0 + index * row_height
        x = max(x0, min(x1, x0 + (accuracy - minimum) / (maximum - minimum) * (x1 - x0)))
        if label == "Multi-proxy + task-conditioned":
            draw.text((35, y - 24), "Multi-proxy +", font=F8, fill=DARK)
            draw.text((35, y + 2), "task-conditioned", font=F8, fill=DARK)
        else:
            draw.text((35, y - 13), label, font=F8B if label == "ProxyDiff" else F8, fill=DARK)
        for center_x, enabled in zip(column_x, checks):
            if enabled:
                draw_check(draw, center_x, y)
        draw.line((x0, y, x, y), fill=DARK, width=5)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=DARK, outline="white", width=2)
        gain = accuracy - base if abs(accuracy - base) >= 0.005 else 0.0
        text_x = x + 16 if x < x1 - 45 else x - 70
        draw.text((text_x, y - 24), f"{accuracy:.2f}", font=F8B if label == "ProxyDiff" else F8, fill=DARK)
        draw.text((text_x, y + 2), f"{gain:+.2f}", font=F7, fill=GREEN if gain > 0 else RED if gain < 0 else DARK)
    save(image, output_dir, "fig_nb301_component_ablation.png")


def proxy_geometry(output_dir: Path, reported):
    raw_corr = float(reported[("proxy_geometry", "raw_proxy_scores_mean_abs_spearman")])
    axis_corr = float(reported[("proxy_geometry", "factorized_axes_mean_abs_spearman")])
    raw_rank = float(reported[("proxy_geometry", "raw_proxy_scores_effective_rank")])
    axis_rank = float(reported[("proxy_geometry", "factorized_axes_effective_rank")])
    image = Image.new("RGB", (640, 350), "white")
    draw = ImageDraw.Draw(image)
    top, bottom = 68, 252
    left, right = 78, 568
    draw.line((left, bottom, right, bottom), fill=DARK, width=2)
    for fraction in (0.25, 0.50, 0.75, 1.00):
        y = bottom - int(fraction * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=1)
    for center, values, maximum, name in [
        (220, (raw_corr, axis_corr), 0.35, "Correlation"),
        (426, (raw_rank, axis_rank), 10.0, "Effective Rank"),
    ]:
        for offset, value, color in [(-18, values[0], BLUE), (18, values[1], RED)]:
            y = bottom - int(float(value) / maximum * (bottom - top))
            draw.rectangle((center + offset - 14, y, center + offset + 14, bottom), fill=color)
            text_center(draw, (center + offset, y - 28), f"{value:.2f}", F7)
        text_center(draw, (center, bottom + 17), name, F8)
    text_center(draw, (image.width / 2, 18), "NAS on CNN", F8B)
    draw.rectangle((156, 324, 174, 336), fill=BLUE)
    draw.text((184, 320), "Raw Proxy", font=F7, fill=DARK)
    draw.rectangle((350, 324, 368, 336), fill=RED)
    draw.text((378, 320), "Disentangled Axis", font=F7, fill=DARK)
    save(image, output_dir, "fig_nb301_proxy_geometry.png")


def exact_trajectory(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    values = {}
    for row in rows:
        method = str(row.get("method", ""))
        selected = row.get("free_top_score_arch") or {}
        if "acc" not in selected:
            continue
        if method == "proxydiff_epoch_-01_score_params":
            values[0] = float(selected["acc"])
        elif method == "proxydiff_epoch_000_score_params":
            values[60] = float(selected["acc"])
        else:
            match = re.search(r"step_(\d+)", method)
            if match:
                values[int(match.group(1))] = float(selected["acc"])
    required = (0, 10, 20, 30, 40, 50, 60)
    missing = [step for step in required if step not in values]
    if missing:
        raise ValueError(f"missing exact trajectory checkpoints: {missing}")
    return [(step, values[step]) for step in required]


def trajectory(output_dir: Path, main_values, decode_path: Path):
    points = exact_trajectory(decode_path)
    image = Image.new("RGB", (880, 360), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 95, 70, 790, 270
    split = left + (right - left) // 2
    draw.rectangle((left, top - 8, split, bottom), fill=(248, 250, 253))
    draw.rectangle((split, top - 8, right, bottom), fill=(253, 250, 247))
    draw.line((left, bottom, right, bottom), fill=DARK, width=2)
    draw.line((left, top, left, bottom), fill=DARK, width=2)
    draw.text((left - 45, top - 35), "acc.", font=F9, fill=DARK)
    draw.text((right - 25, bottom + 28), "step", font=F9, fill=DARK)

    def x_for(step):
        return left + int(step / 60 * (right - left))

    def y_for(accuracy):
        return bottom - int((accuracy - 93.5) / (94.7 - 93.5) * (bottom - top))

    for tick in (93.5, 94.0, 94.5):
        y = y_for(tick)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 58, y - 11), f"{tick:.1f}", font=F8, fill=DARK)
    for tick in range(0, 61, 10):
        x = x_for(tick)
        draw.line((x, bottom, x, bottom + 7), fill=DARK, width=1)
        draw.text((x - 13, bottom + 12), str(tick), font=F8, fill=DARK)
    draw.line((split, top - 12, split, bottom), fill=DARK, width=2)
    text_center(draw, ((left + split) / 2, top - 34), "axis calibration", F8)
    text_center(draw, ((split + right) / 2, top - 34), "component correction", F8)
    draw.line([(x_for(step), y_for(acc)) for step, acc in points if step <= 30], fill=BLUE, width=4)
    draw.line([(x_for(step), y_for(acc)) for step, acc in points if step >= 30], fill=RED, width=4)
    for step, accuracy in points:
        color = BLUE if step <= 30 else RED
        x, y = x_for(step), y_for(accuracy)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=1)
    labels = [
        (0, main_values["prior"], "prior", BLUE, 12, 12),
        (30, main_values["axis"], "axis", BLUE, -136, 12),
        (60, main_values["final"], "final", RED, -118, 32),
    ]
    for step, accuracy, label, color, dx, dy in labels:
        draw.text((x_for(step) + dx, y_for(accuracy) + dy), f"{label} {accuracy:.2f}", font=F8, fill=color)
    save(image, output_dir, "fig_nb301_refinement_stagewise.png")


def axis_calibration(output_dir: Path, weights_path: Path, profiles_path: Path):
    with weights_path.open(encoding="utf-8") as stream:
        weights = list(csv.DictReader(stream))
    profiles = defaultdict(list)
    with profiles_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            profiles[row["coordinate"]].append(row)
    image = Image.new("RGB", (1750, 740), "white")
    draw = ImageDraw.Draw(image)
    y0, row_height = 105, 58
    profile_left, profile_right = 315, 735
    zero_x, scale = 1290, 155
    headers = [("Score Direction", 150), ("Proxy Profile", 505), ("Nearest Proxy", 840), ("Axis-Level Weight", zero_x), ("Effect", 1575)]
    for label, x in headers:
        text_center(draw, (x, y0 - 42), label, F9B)
    draw.line((82, y0 - 12, 1665, y0 - 12), fill=DARK, width=2)
    draw.line((zero_x, y0 - 5, zero_x, y0 + len(weights) * row_height - 18), fill=DARK, width=4)
    for index, row in enumerate(weights):
        y = y0 + index * row_height
        if index % 2 == 0:
            draw.rectangle((82, y - 8, 1665, y + row_height - 10), fill=PALE)
        coordinate = "consensus score" if row["coordinate"] == "base_readout" else row["coordinate"]
        coefficient = float(row["effective_coefficient"])
        if row["coordinate"] == "base_readout":
            role, color, nearest = "fixed anchor", GREEN, "consensus score"
        else:
            role = "suppressed" if abs(coefficient) < 0.05 else "reversed" if coefficient < 0 else "strengthened"
            color = MID if role == "suppressed" else RED if coefficient < 0 else GREEN
            nearest = row["strongest_proxy"]
        draw.text((95, y + 4), coordinate, font=F10, fill=DARK)
        if row["coordinate"] != "base_readout":
            draw.line((profile_left, y + 22, profile_right, y + 22), fill=GRID, width=2)
            midpoint = (profile_left + profile_right) / 2
            draw.line((midpoint, y + 12, midpoint, y + 32), fill=DARK, width=1)
            for profile in profiles[row["coordinate"]]:
                correlation = float(profile["oriented_spearman"])
                x = profile_left + (correlation + 1) / 2 * (profile_right - profile_left)
                dot_color = ORANGE if profile["proxy"] == nearest else BLUE
                draw.ellipse((x - 5, y + 17, x + 5, y + 27), fill=dot_color, outline="white", width=1)
        else:
            draw.text((profile_left, y + 4), "consensus score", font=F9, fill=DARK)
        draw.text((770, y + 4), nearest, font=F9, fill=DARK)
        endpoint = zero_x + coefficient * scale
        draw.line((zero_x, y + 22, endpoint, y + 22), fill=color, width=7)
        draw.ellipse((endpoint - 8, y + 14, endpoint + 8, y + 30), fill=color, outline="white", width=1)
        draw.text((1488, y + 4), f"{coefficient:+.2f}", font=F10, fill=color)
        draw.text((1585, y + 4), role, font=F10, fill=color)
    legend_y = y0 + len(weights) * row_height + 12
    draw.ellipse((315, legend_y + 6, 327, legend_y + 18), fill=BLUE)
    draw.text((342, legend_y), "retained proxy", font=F9, fill=DARK)
    draw.ellipse((595, legend_y + 6, 607, legend_y + 18), fill=ORANGE)
    draw.text((625, legend_y), "nearest proxy", font=F9, fill=DARK)
    save(image, output_dir, "fig_nb301_axis_calibration.png")


def subset_stability(output_dir: Path, path: Path):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    variants = [("rank_mean", "rank-mean", BLUE), ("log_rank", "AZ-NAS", PURPLE), ("proxydiff", "ProxyDiff", RED)]
    groups = [("retained_pool", 3, "Gate-9"), ("retained_pool", 5, "Gate-9"), ("full_proxy_pool", 3, "All-19"), ("full_proxy_pool", 5, "All-19")]
    image = Image.new("RGB", (1500, 650), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 285, 1390, 65, 485
    minimum, maximum = 90.8, 94.8

    def x_for(accuracy):
        clipped = max(minimum, min(maximum, float(accuracy)))
        return left + (clipped - minimum) / (maximum - minimum) * (right - left)

    draw.line((left, bottom, right, bottom), fill=DARK, width=2)
    text_center(draw, ((left + right) / 2, 22), "NB301 surrogate accuracy", F10)
    for tick in (91, 92, 93, 94, 94.5):
        x = x_for(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=DARK, width=2)
        text_center(draw, (x, bottom + 17), f"{tick:g}", F9)
    offsets = {"rank_mean": 28, "log_rank": 0, "proxydiff": -28}
    for group_index, (pool, size, label) in enumerate(groups):
        center_y = 112 + group_index * 98
        if group_index % 2 == 0:
            draw.rectangle((60, center_y - 48, right + 35, center_y + 44), fill=PALE)
        draw.text((82, center_y - 16), f"{label} k={size}", font=F10, fill=DARK)
        draw.line((left, center_y + 46, right, center_y + 46), fill=(232, 235, 238), width=1)
        for protocol, _, color in variants:
            values = sorted(
                float(row["accuracy"])
                for row in rows
                if row["pool"] == pool
                and int(row["subset_size"]) == size
                and row["protocol"] == protocol
            )
            if not values:
                raise ValueError(f"missing subset results for {pool}, k={size}, {protocol}")
            median_value = values[len(values) // 2]
            y = center_y + offsets[protocol]
            draw.line((x_for(min(values)), y, x_for(max(values)), y), fill=color, width=5)
            x = x_for(median_value)
            draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline="white", width=2)
    for index, (_, label, color) in enumerate(variants):
        x = 320 + index * 300
        draw.line((x - 25, 566, x + 45, 566), fill=color, width=5)
        draw.ellipse((x, 555, x + 22, 577), fill=color, outline="white", width=2)
        draw.text((x + 56, 550), label, font=F10, fill=DARK)
    save(image, output_dir, "fig_nb301_subset_stability.png")


def main():
    args = parse_args()
    reported = read_reported(args.reported_summary)
    main_values = read_main(args.main_summary)
    component_ablation(args.output_dir, reported)
    proxy_geometry(args.output_dir, reported)
    trajectory(args.output_dir, main_values, args.full_decode)
    axis_calibration(args.output_dir, args.axis_weights, args.axis_profiles)
    subset_stability(args.output_dir, args.subset_results)


if __name__ == "__main__":
    main()
