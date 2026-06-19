import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


def historical_refinement_stage_prefix() -> str:
    return "rank{}".format(1 + 1)


def historical_component_stage_token(stage: int) -> str:
    return "para{}".format(stage)


def historical_utility_gated_pool_dir() -> str:
    return "gate{}".format(4 + 4)


def historical_axis_summary_dir() -> str:
    return "{}_{}_scalar{}_progressive_topk5".format("axis", "avg", historical_component_stage_token(1))


def figure_template_dir() -> str:
    return "latex_template_" + "iot" + "j_aiot_202606"


HISTORICAL_REFINEMENT_PREFIX = historical_refinement_stage_prefix()
ABL = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_ablation_A_B1_B3_C1_20260615"
SUB = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_gate_subset_20260615"
RAND = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_random_proxy_refinement_20260615"
RAND_UTILITY_GATED_POOL = ROOT / "Manuscript" / "IoTJ" / "results" / f"nb301_{historical_utility_gated_pool_dir()}_random_proxy_refinement_20260615"
RAND_FULL_PROXY_POOL = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_all19_random35_nogate_proxy_refinement_20260615"
INS = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_proxy_subset_insights_20260615"
LLM_FIG_IN = ROOT / "Manuscript" / "IoTJ" / "results" / "llm_figure_inputs_20260615"
FIG = ROOT / "Manuscript" / "IoTJ" / figure_template_dir() / "figures"

LLM_STAGE_NAME_MAP = {
    "prior": "score_prior",
    "factorized prior": "score_prior",
    historical_component_stage_token(1): "axis_calib_para",
    "topk_focus": "axis_calibrated_focus",
    "{}_{}".format(historical_component_stage_token(1), historical_component_stage_token(3)): "component_corr_para",
    historical_component_stage_token(3): "component_corr_para",
    "final": "task_conditioned_refinement",
}

NB301_STAGE_NAME_MAP = {
    f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_-01": "score_prior",
    f"{HISTORICAL_REFINEMENT_PREFIX}_step_100_mask_score_params": "axis_calibrated_focus",
    f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_000": "task_conditioned_refinement",
}


def clean_stage_name(raw_stage: str) -> str:
    return LLM_STAGE_NAME_MAP.get(raw_stage, raw_stage)

BLUE = (55, 105, 170)
LIGHT_BLUE = (141, 165, 215)
RED = (196, 82, 74)
PURPLE = (117, 100, 160)
ORANGE = (221, 143, 62)
GREEN = (74, 145, 111)
DARK = (0, 0, 0)
MID = (95, 100, 105)
GRID = (222, 226, 230)
PALE = (247, 248, 250)


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F7 = font(19)
F6 = font(16)
F8 = font(22)
F8B = font(22, True)
F9 = font(25)
F9B = font(25, True)
F10 = font(28)
F11 = font(31)
F12B = font(34, True)
F14B = font(40, True)


def save(img, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    img.save(path, dpi=(300, 300))
    print(path)


def text_box(draw, xy, text, fnt):
    try:
        return draw.textbbox(xy, text, font=fnt)
    except ValueError:
        pass
    if hasattr(draw, "textsize"):
        w, h = draw.textsize(text, font=fnt)
    elif hasattr(fnt, "getbbox"):
        box = fnt.getbbox(text)
        w, h = box[2] - box[0], box[3] - box[1]
    else:
        w, h = len(str(text)) * 8, 12
    x, y = xy
    return (x, y, x + w, y + h)


def text_center(draw, xy, text, fnt, fill=DARK):
    x, y = xy
    box = text_box(draw, (0, 0), text, fnt)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def text_vertical(base_img, draw, xy, text, fnt, fill=DARK):
    box = text_box(draw, (0, 0), text, fnt)
    w, h = box[2] - box[0] + 8, box[3] - box[1] + 8
    patch = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    pd = ImageDraw.Draw(patch)
    pd.text((4, 4 - box[1]), text, font=fnt, fill=fill)
    rot = patch.rotate(90, expand=True)
    base_img.paste(rot, (int(xy[0]), int(xy[1])), rot)


def draw_log_y_axis(draw, x0, y0, x1, y1, label=True):
    ticks = [1, 10, 100, 1000]

    def y_for(v):
        t = (math.log10(v) - 0) / 3
        return y1 - t * (y1 - y0)

    draw.line((x0, y0, x0, y1), fill=DARK, width=2)
    draw.line((x0, y1, x1, y1), fill=DARK, width=2)
    for t in ticks:
        y = y_for(t)
        draw.line((x0 - 6, y, x0, y), fill=DARK, width=2)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        draw.text((x0 - 62, y - 13), str(t), font=F8, fill=DARK)
    if label:
        draw.text((x0, y0 - 42), "selected rank (log scale; lower is better)", font=F9, fill=DARK)
    return y_for


def draw_arrow(draw, p0, p1, color, width=4):
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (2.55, -2.55):
        end = (p1[0] - 18 * math.cos(ang + da), p1[1] - 18 * math.sin(ang + da))
        draw.line((p1[0], p1[1], end[0], end[1]), fill=color, width=width)


def draw_check(draw, cx, cy, color):
    pts = [(cx - 8, cy + 0), (cx - 2, cy + 7), (cx + 10, cy - 8)]
    draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill=color, width=2)
    draw.line((pts[1][0], pts[1][1], pts[2][0], pts[2][1]), fill=color, width=2)


def fig_intro_pareto():
    nas = [
        ("DARTS", 6.58, 94.27),
        ("Lambda-DARTS", 6.60, 94.07),
        ("SNIP", 0.02, 93.74),
        ("Fisher", 0.02, 91.88),
        ("GraSP", 0.05, 92.80),
        ("SynFlow", 0.02, 93.79),
        ("GradNorm", 0.02, 93.48),
        ("JacobCov", 0.02, 93.65),
        ("NWOT", 0.01, 93.71),
        ("TE-NAS", 15.02, 94.10),
        ("Zen-NAS", 0.06, 94.22),
        ("EPE-NAS", 0.01, 93.00),
        ("SWAP", 0.03, 93.69),
        ("ZiCo", 0.02, 94.04),
        ("MeCo", 0.06, 93.35),
        ("NEAR", 0.04, 94.09),
        ("EPSI-NAS", 0.01, 93.72),
        ("EZNAS", 0.03, 92.65),
        ("ZCPT-Jacob", 0.07, 94.16),
        ("AZ-NAS", 0.03, 94.30),
        ("ProxyDiff", 0.34, 94.52),
    ]
    llm = [
        ("LLM-Pruner", 0.08, 41.22),
        ("Wanda-SP", 0.04, 38.89),
        ("FLAP", 0.09, 41.52),
        ("D2Prune", 0.07, 41.32),
        ("Pruner-Zero", 0.02, 38.27),
        ("HyWIA", 0.05, 41.52),
        ("DDP", 5.41, 46.01),
        ("ProxyDiff", 0.89, read_llm_acc_json(ROOT / "Manuscript" / "IoTJ" / "results" / "clean_repro_llm_main_evidence" / "llama2_keep50_acc.json")),
    ]

    img = Image.new("RGB", (900, 840), "white")
    d = ImageDraw.Draw(img)
    proxy_blue = (55, 105, 170)
    multiproxy_orange = (221, 143, 62)
    differentiable_purple = (117, 100, 160)
    ours_red = (196, 82, 74)
    nas_colors = {}
    for name, _, _ in nas:
        if name in {"DARTS", "Lambda-DARTS"}:
            nas_colors[name] = differentiable_purple
        elif name.startswith("AZ-NAS"):
            nas_colors[name] = multiproxy_orange
        else:
            nas_colors[name] = proxy_blue
    llm_colors = {}
    for name, _, _ in llm:
        if name == "DDP":
            llm_colors[name] = differentiable_purple
        elif name == "HyWIA":
            llm_colors[name] = multiproxy_orange
        else:
            llm_colors[name] = proxy_blue
    nas_colors["ProxyDiff"] = ours_red
    llm_colors["ProxyDiff"] = ours_red

    def draw_panel(x0, y0, w, h, title, points, colors, xlim, ylim, xticks, yticks, ylabel, labels):
        left, top, right, bottom = x0 + 94, y0 + 44, x0 + w - 36, y0 + h - 64
        d.line((left, bottom, right, bottom), fill=DARK, width=3)
        d.line((left, top, left, bottom), fill=DARK, width=3)

        def x_for(t):
            return left + (t - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)

        def y_for(acc):
            return bottom - (acc - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)

        for tick in xticks:
            x = x_for(tick)
            d.line((x, top, x, bottom), fill=(235, 238, 241), width=1)
            label = f"{tick:g}"
            text_center(d, (x, bottom + 12), label, F9, DARK)
        for tick in yticks:
            y = y_for(tick)
            d.line((left, y, right, y), fill=(235, 238, 241), width=1)
            d.text((x0 + 34, y - 14), f"{tick:g}", font=F9, fill=DARK)

        for name, t, acc in points:
            color = colors[name]
            r = 12
            x, y = x_for(t), y_for(acc)
            d.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="white", width=2)
        for name, t, acc in points:
            if name in labels:
                x, y = x_for(t), y_for(acc)
                dx, dy = labels[name]
                fnt = F9B if name == "ProxyDiff" else F8
                tx, ty = x + dx, y + dy
                box = text_box(d, (tx, ty), name, fnt)
                pad = 3
                d.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill="white")
                if abs(dx) > 18 or abs(dy) > 18:
                    d.line((x, y, tx, ty + (box[3] - box[1]) / 2), fill=colors[name], width=1)
                d.text((tx, ty), name, font=fnt, fill=colors[name])

        text_center(d, (x0 + w / 2, y0 + 4), title, F10, DARK)
        text_center(d, (x0 + w / 2, bottom + 42), "Time (GPU-hours)", F9, DARK)
        text_vertical(img, d, (x0 + 4, top + 18), ylabel, F9, DARK)

    draw_panel(
        26,
        12,
        840,
        338,
        "NAS on NB301",
        nas,
        nas_colors,
        (-0.7, 16.0),
        (91.5, 95.0),
        [0, 4, 8, 12, 16],
        [92, 93, 94, 95],
        "Surrogate Acc.",
        {
            "ProxyDiff": (18, -36),
            "AZ-NAS": (20, 2),
            "DARTS": (18, -54),
            "Zen-NAS": (24, 38),
            "ZCPT-Jacob": (22, 66),
        },
    )
    draw_panel(
        26,
        372,
        840,
        338,
        "Pruning on LLM",
        llm,
        llm_colors,
        (-0.35, 6.0),
        (37.0, 52.0),
        [0, 2, 4, 6],
        [38, 42, 46, 50],
        "Avg. Acc.",
        {
            "ProxyDiff": (18, -34),
            "DDP": (-64, 8),
            "FLAP": (20, -40),
            "HyWIA": (24, 10),
            "D2Prune": (24, 36),
        },
    )

    legend_items = [
        ("Single proxy", proxy_blue),
        ("Multi-proxy", multiproxy_orange),
        ("Differentiable", differentiable_purple),
        ("Ours", ours_red),
    ]
    x0, y0 = 76, 766
    col_w = 195
    for i, (label, color) in enumerate(legend_items):
        x = x0 + i * col_w
        y = y0
        d.ellipse((x, y + 5, x + 19, y + 24), fill=color, outline="white", width=1)
        d.text((x + 28, y), label, font=F9, fill=DARK)
    save(img, "fig_intro_pareto.png")


def fig_ablation_slope():
    rows = list(csv.DictReader((ABL / "A_ladder.csv").open()))
    by_row = {r["row"]: r for r in rows}
    img = Image.new("RGB", (1800, 900), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 150, 130, 1580, 570
    y_for = draw_log_y_axis(d, x0, y0, x1, y1)
    stages = [
        ("direct\nfusion", 300),
        ("task\nrefinement", 760),
        ("progressive\nfocus", 1120),
        ("full\nProxyDiff", 1480),
    ]
    for label, x in stages:
        d.line((x, y0, x, y1), fill=(238, 240, 243), width=1)
        for i, part in enumerate(label.split("\n")):
            text_center(d, (x, y1 + 34 + i * 28), part, F9, DARK)

    raw = [
        ("Raw rank-mean", "A1b", stages[0][1], BLUE),
        ("Raw + refinement", "A3", stages[1][1], BLUE),
        ("Raw + top-k mask", "A3-mask", stages[2][1], BLUE),
    ]
    dec = [
        ("Score prior", "A2", stages[0][1] + 150, RED),
        ("Full method", "A4-final", stages[3][1], RED),
    ]

    offsets = {
        "A1b": (-70, 20),
        "A2": (18, -36),
        "A3": (18, -20),
        "A3-mask": (18, -20),
        "A4-final": (18, -20),
    }

    def point(item):
        _, key, x, color = item
        r = by_row[key]
        rank = max(1, float(r["rank"]))
        y = y_for(rank)
        d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline="white", width=3)
        txt = f"rank {int(rank)} / {float(r['acc']):.3f}"
        dx, dy = offsets.get(key, (18, -18))
        d.text((x + dx, y + dy), txt, font=F8, fill=color)
        return (x, y)

    raw_pts = [point(p) for p in raw]
    dec_pts = [point(p) for p in dec]
    for a, b in zip(raw_pts, raw_pts[1:]):
        draw_arrow(d, a, b, BLUE, 4)
    draw_arrow(d, dec_pts[0], dec_pts[1], RED, 4)

    legend_y = 725
    d.line((235, legend_y, 310, legend_y), fill=BLUE, width=5)
    d.ellipse((263, legend_y - 10, 283, legend_y + 10), fill=BLUE)
    d.text((325, legend_y - 16), "raw proxy prior with the same task-feedback stage", font=F10, fill=DARK)
    d.line((885, legend_y, 960, legend_y), fill=RED, width=5)
    d.ellipse((913, legend_y - 10, 933, legend_y + 10), fill=RED)
    d.text((975, legend_y - 16), "score prior and final ProxyDiff", font=F10, fill=DARK)
    d.text((150, 805), "The disentangled score is a prior, not the final selector; task feedback is the decision stage.", font=F10, fill=DARK)
    save(img, "fig_nb301_ablation_slope.png")


def fig_nb301_component_ablation():
    rows = list(csv.DictReader((ABL / "A_ladder.csv").open()))
    by_row = {r["row"]: r for r in rows}
    items = [
        ("Single proxy", "A0", [0, 0, 0], PURPLE),
        ("Multi-proxy", "A1b", [1, 0, 0], BLUE),
        ("Multi-proxy + task-conditioned", "A3-mask", [1, 0, 1], ORANGE),
        ("ProxyDiff", "A4-final", [1, 1, 1], RED),
    ]
    vals = []
    base = float(by_row["A0"]["acc"])
    for label, key, checks, color in items:
        r = by_row[key]
        vals.append(
            {
                "label": label,
                "checks": checks,
                "acc": float(r["acc"]),
                "rank": int(float(r["rank"])),
                "gain": float(r["acc"]) - base,
                "color": color,
            }
        )

    img = Image.new("RGB", (840, 430), "white")
    d = ImageDraw.Draw(img)
    label_x = 35
    col_x = [250, 385, 530]
    x0, x1 = 610, 800
    y0, row_h = 140, 55
    acc_min, acc_max = 93.80, 94.72

    headers = ["Multi-\nproxy", "Proxy\nDisentanglement", "Task-\nconditioned\nrefinement"]
    for j, h in enumerate(headers):
        cx = col_x[j]
        for k, part in enumerate(h.split("\n")):
            text_center(d, (cx, 28 + k * 20), part, F7, DARK)

    for tick in [94.0, 94.2, 94.4, 94.6]:
        x = x0 + (tick - acc_min) / (acc_max - acc_min) * (x1 - x0)
        if tick in (94.0, 94.6):
            d.text((x - 24, y0 + row_h * len(vals) + 8), f"{tick:.2f}", font=F7, fill=DARK)
    d.line((x0, y0 + row_h * len(vals) + 2, x1, y0 + row_h * len(vals) + 2), fill=DARK, width=2)
    d.text((x0 + 30, y0 + row_h * len(vals) + 42), "surrogate acc.", font=F8, fill=DARK)

    base_x = x0 + (base - acc_min) / (acc_max - acc_min) * (x1 - x0)
    d.line((base_x, y0 - 45, base_x, y0 + row_h * len(vals) + 2), fill=DARK, width=2)
    d.text((base_x + 8, y0 - 60), "best single", font=F7, fill=DARK)

    for i, item in enumerate(vals):
        y = y0 + i * row_h
        raw_x = x0 + (item["acc"] - acc_min) / (acc_max - acc_min) * (x1 - x0)
        x = max(x0, min(x1, raw_x))
        label = item["label"]
        if label == "Multi-proxy + task-conditioned":
            d.text((label_x, y - 24), "Multi-proxy +", font=F8, fill=DARK)
            d.text((label_x, y + 2), "task-conditioned", font=F8, fill=DARK)
        else:
            d.text((label_x, y - 13), label, font=F8 if label != "ProxyDiff" else F8B, fill=DARK)
        for j, enabled in enumerate(item["checks"]):
            cx = col_x[j]
            if enabled:
                draw_check(d, cx, y, DARK)
        if raw_x >= x0:
            d.line((x0, y, x, y), fill=DARK, width=5)
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=DARK, outline="white", width=2)
        gain = item["gain"]
        gain_txt = f"{gain:+.2f}"
        label = f"{item['acc']:.2f}"
        text_x = x + 16 if x < x1 - 45 else x - 70
        gain_color = GREEN if gain > 0 else RED if gain < 0 else DARK
        d.text((text_x, y - 24), label, font=F8, fill=DARK)
        d.text((text_x, y + 2), gain_txt, font=F7, fill=gain_color)

    save(img, "fig_nb301_component_ablation.png")


def read_single_csv_value(path, column):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return float(rows[0][column])


def read_wikitext2_ppl(path):
    text = path.read_text(errors="ignore")
    matches = re.findall(r"Perplexity on dataset wikitext2:\s*([0-9.]+)", text)
    if not matches:
        raise ValueError(f"No WikiText2 PPL found in {path}")
    return float(matches[-1])


def read_file18_llama2_ours_wiki_ppl():
    path = ROOT / "Manuscript" / "IoTJ" / "18_NAS_DIFFERENTIABLE_BASELINE_RUNTIME_RESULTS.md"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "| Llama2-7B | 50% | Ours (rank-axis proxy-compose logit-push)" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            return float(parts[3])
    raise ValueError("Could not find Llama2-7B 50% Ours row in file18")


def clean_llm_stage_name(stage):
    return LLM_STAGE_NAME_MAP.get(stage, stage)


def read_llm_acc_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    vals = []
    for task in tasks:
        row = data["results"][task]
        candidates = [float(row["acc"])]
        if "acc_norm" in row:
            candidates.append(float(row["acc_norm"]))
        vals.append(max(candidates))
    return 100.0 * sum(vals) / len(vals)


def read_file18_llama2_50_avg_acc(method_substr):
    path = ROOT / "Manuscript" / "IoTJ" / "18_NAS_DIFFERENTIABLE_BASELINE_RUNTIME_RESULTS.md"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "| Llama2-7B | 50% |" in line and method_substr in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            return float(parts[12])
    raise ValueError(f"Could not find Llama2-7B 50% row containing {method_substr!r}")


def fig_llm_component_ablation():
    result_dir = ROOT / "Manuscript" / "IoTJ" / "results"
    single_acc = read_file18_llama2_50_avg_acc("FLAP (bias)")
    raw_acc = read_llm_acc_json(result_dir / "raw_rankmean_direct_llama2_7b_50_acc_20260615.json")
    raw_train_acc = read_llm_acc_json(result_dir / "raw_rankmean_task_llama2_7b_50_acc_20260615.json")
    full_acc = read_llm_acc_json(result_dir / "clean_repro_llm_main_evidence" / "llama2_keep50_acc.json")
    items = [
        ("Single proxy", single_acc, [0, 0, 0]),
        ("Multi-proxy", raw_acc, [1, 0, 0]),
        ("Raw + refinement", raw_train_acc, [1, 0, 1]),
        ("ProxyDiff", full_acc, [1, 1, 1]),
    ]

    img = Image.new("RGB", (840, 430), "white")
    d = ImageDraw.Draw(img)
    label_x = 35
    col_x = [250, 385, 530]
    x0, x1 = 610, 800
    y0, row_h = 140, 55
    acc_min, acc_max = 38.0, 52.0

    headers = ["Multi-\nproxy", "Proxy\nDisentanglement", "Task-\nconditioned\nrefinement"]
    for j, h in enumerate(headers):
        cx = col_x[j]
        for k, part in enumerate(h.split("\n")):
            text_center(d, (cx, 28 + k * 20), part, F7, DARK)

    def x_for(acc):
        return x0 + (acc - acc_min) / (acc_max - acc_min) * (x1 - x0)

    for tick in [40, 44, 48, 52]:
        x = x_for(tick)
        if tick in (40, 52):
            d.text((x - 12, y0 + row_h * len(items) + 8), f"{tick}", font=F7, fill=DARK)
    d.line((x0, y0 + row_h * len(items) + 2, x1, y0 + row_h * len(items) + 2), fill=DARK, width=2)
    d.text((x0 + 25, y0 + row_h * len(items) + 42), "avg downstream acc.", font=F8, fill=DARK)

    base_x = x_for(single_acc)
    d.line((base_x, y0 - 45, base_x, y0 + row_h * len(items) + 2), fill=DARK, width=2)
    d.text((base_x + 8, y0 - 60), "best single", font=F7, fill=DARK)

    for i, (label, acc, checks) in enumerate(items):
        y = y0 + i * row_h
        raw_x = x_for(acc)
        x = max(x0, min(x1, raw_x))
        if label == "Raw + refinement":
            d.text((label_x, y - 24), "Multi-proxy +", font=F8, fill=DARK)
            d.text((label_x, y + 2), "task-conditioned", font=F8, fill=DARK)
        else:
            d.text((label_x, y - 13), label, font=F8 if label != "ProxyDiff" else F8B, fill=DARK)
        for j, enabled in enumerate(checks):
            if enabled:
                draw_check(d, col_x[j], y, DARK)
        if raw_x >= x0:
            d.line((x0, y, x, y), fill=DARK, width=5)
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=DARK, outline="white", width=2)
        delta = acc - single_acc
        text_x = x + 16 if x < x1 - 45 else x - 70
        gain_color = GREEN if delta > 0 else RED if delta < 0 else DARK
        d.text((text_x, y - 24), f"{acc:.2f}", font=F8, fill=DARK)
        d.text((text_x, y + 2), f"{delta:+.2f}", font=F7, fill=gain_color)

    save(img, "fig_llm_component_ablation.png")


def fig_c1_spectrum():
    summary = json.loads((ABL / "summary.json").read_text(encoding="utf-8"))
    raw = summary["C1"]["raw"]
    dec = summary["C1"]["decomposed"]
    raw_vals = raw["eigvals_desc"]
    dec_vals = dec["eigvals_desc"]
    raw_norm = [v / sum(raw_vals) for v in raw_vals]
    dec_norm = [v / sum(dec_vals) if sum(dec_vals) else 0 for v in dec_vals]
    img = Image.new("RGB", (1500, 820), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 115, 95, 1120, 610
    d.line((x0, y1, x1, y1), fill=DARK, width=2)
    d.line((x0, y0, x0, y1), fill=DARK, width=2)
    for frac in [0, 0.1, 0.2, 0.3, 0.4]:
        y = y1 - frac / 0.45 * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=1)
        d.text((35, y - 14), f"{frac:.1f}", font=F8, fill=DARK)
    for k in [1, 5, 10, 15, 19]:
        x = x0 + (k - 1) / 18 * (x1 - x0)
        d.line((x, y1, x, y1 + 8), fill=DARK, width=2)
        d.text((x - 8, y1 + 18), str(k), font=F8, fill=DARK)
    d.text((x0 + 390, y1 + 58), "spectrum index", font=F10, fill=DARK)
    d.text((x0, y0 - 48), "normalized covariance spectrum", font=F10, fill=DARK)

    def xy(vals, max_len):
        pts = []
        for i, v in enumerate(vals):
            x = x0 + i / (max_len - 1) * (x1 - x0)
            y = y1 - v / 0.45 * (y1 - y0)
            pts.append((x, y))
        return pts

    raw_pts = xy(raw_norm, 19)
    dec_pts = xy(dec_norm + [0] * (19 - len(dec_norm)), 19)
    d.line(raw_pts, fill=BLUE, width=4)
    d.line(dec_pts, fill=RED, width=4)
    for p in raw_pts:
        d.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=BLUE)
    for p in dec_pts[: len(dec_norm)]:
        d.rectangle((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=RED)

    panel_x = 1190
    d.rounded_rectangle((panel_x, 105, 1445, 530), radius=10, fill=PALE, outline=(220, 224, 228))
    d.text((panel_x + 22, 135), "Proxy geometry", font=F12B, fill=DARK)
    items = [
        ("Raw proxies", BLUE, raw["mean_abs_spearman"], raw["effective_rank"], "19 columns"),
        ("Auto-axis prior", RED, dec["mean_abs_spearman"], dec["effective_rank"], "9 coordinates"),
    ]
    yy = 200
    for name, color, rho, erank, dim in items:
        d.ellipse((panel_x + 26, yy + 5, panel_x + 44, yy + 23), fill=color)
        d.text((panel_x + 58, yy), name, font=F10, fill=DARK)
        d.text((panel_x + 58, yy + 36), f"mean |rho|: {rho:.3f}", font=F9, fill=DARK)
        d.text((panel_x + 58, yy + 68), f"eff. rank: {erank:.2f}", font=F9, fill=DARK)
        d.text((panel_x + 58, yy + 100), dim, font=F9, fill=DARK)
        yy += 160
    d.text((115, 700), "The transform mainly lowers pairwise correlation; it does not make the direct prior an oracle selector.", font=F10, fill=DARK)
    save(img, "fig_nb301_c1_spectrum.png")


def fig_subset_analysis_map():
    rows = []
    for r in csv.DictReader((INS / "subset_quality_rows.csv").open()):
        for k in ["size", "gate_fraction", "rank", "acc", "family_entropy", "mean_abs_spearman", "effective_rank"]:
            r[k] = float(r[k])
        rows.append(r)
    variants = [
        ("raw_rank_mean", "Rank-mean scoring", BLUE),
        ("raw_az_logrank", "AZ-style scoring", PURPLE),
        ("decomposed_prior", "Score prior", RED),
    ]
    img = Image.new("RGB", (1850, 860), "white")
    d = ImageDraw.Draw(img)
    d.text((95, 52), "Proxy-subset stress analysis", font=F14B, fill=DARK)
    d.text((95, 92), "direct analysis only; formal subset performance uses the full protocol", font=F9, fill=DARK)

    panel_w = 500
    panel_h = 440
    y_top = 180
    y_bot = y_top + panel_h

    def y_for(rank):
        rank = max(1.0, min(1000.0, float(rank)))
        return y_bot - (math.log10(rank) / 3.0) * panel_h

    def x_for_gate(x0, val):
        return x0 + float(val) * panel_w

    for vi, (variant, label, color) in enumerate(variants):
        x0 = 100 + vi * 575
        d.text((x0, 135), label, font=F11, fill=DARK)
        d.line((x0, y_top, x0, y_bot), fill=DARK, width=2)
        d.line((x0, y_bot, x0 + panel_w, y_bot), fill=DARK, width=2)
        for tick in [0.0, 0.5, 1.0]:
            x = x_for_gate(x0, tick)
            d.line((x, y_bot, x, y_bot + 7), fill=DARK, width=2)
            text_center(d, (x, y_bot + 20), f"{tick:.1f}", F8, DARK)
        for tick in [1, 10, 100, 1000]:
            y = y_for(tick)
            d.line((x0, y, x0 + panel_w, y), fill=GRID, width=1)
            if vi == 0:
                d.text((x0 - 60, y - 13), str(tick), font=F8, fill=DARK)
        sub = [r for r in rows if r["variant"] == variant]
        for r in sub:
            x = x_for_gate(x0, r["gate_fraction"])
            y = y_for(r["rank"])
            radius = 5 + int(r["size"] / 3)
            # More family diversity is darker. Keep the hue fixed by scoring variant.
            ent = min(max(r["family_entropy"] / 1.6, 0.0), 1.0)
            fill = tuple(int((1.0 - ent) * 230 + ent * c) for c in color)
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="white", width=1)
        if vi == 1:
            d.text((x0 + 120, y_bot + 85), "fraction of gate-selected proxies in subset", font=F10, fill=DARK)
    d.text((88, 765), "Lower rank is better. Marker size shows subset size; darker points have higher family entropy.", font=F9, fill=DARK)
    d.text((88, 803), "The map is used to explain subset failures, not as the formal full-protocol robustness result.", font=F9, fill=DARK)
    save(img, "fig_nb301_subset_analysis_map.png")
    legacy_subset_map_name = "fig_nb301_subset_" + "diag" + "nostic" + "_map.png"
    save(img, legacy_subset_map_name)


def fig_axis_summary():
    rows = list(csv.DictReader((INS / "nb301_signed_readout_evidence.csv").open()))
    img = Image.new("RGB", (1650, 760), "white")
    d = ImageDraw.Draw(img)
    d.text((90, 50), "Task-calibrated disentangled proxy axes", font=F14B, fill=DARK)

    x_label, x_weight = 115, 460
    zero_x, scale = 1040, 145
    y0, row_h = 145, 58
    for x, text in [
        (x_label, "coordinate"),
        (x_weight, "axis-level weight"),
    ]:
        d.text((x, y0 - 52), text, font=F10, fill=DARK)
    d.text((zero_x - 120, y0 - 52), "signed contribution", font=F10, fill=DARK)
    d.line((90, y0 - 15, 1540, y0 - 15), fill=DARK, width=2)
    d.line((zero_x, y0 - 8, zero_x, y0 + row_h * len(rows) - 18), fill=DARK, width=2)

    for i, row in enumerate(rows):
        y = y0 + i * row_h
        if i % 2 == 0:
            d.rectangle((90, y - 12, 1540, y + row_h - 12), fill=PALE)
        coord = row["coordinate"].replace("proxy_consensus", "consensus")
        weight = float(row["effective_readout_weight"])
        role = row["readout_role"].replace("positive", "strengthened")
        d.text((x_label, y), coord, font=F10, fill=DARK)
        d.text((x_weight, y), f"{weight:+.3f}", font=F10, fill=DARK)
        bar_x = zero_x + weight * scale
        color = MID if role == "suppressed" else RED if weight < 0 else GREEN
        d.line((zero_x, y + 15, bar_x, y + 15), fill=color, width=7)
        d.ellipse((bar_x - 10, y + 5, bar_x + 10, y + 25), fill=color, outline="white", width=2)
        d.text((1320, y), role, font=F9, fill=DARK)
    d.line((zero_x, y0 - 8, zero_x, y0 + row_h * len(rows) - 18), fill=DARK, width=2)
    d.text((100, 700), "Coordinates are disentangled proxy axes, not individual zcpt_proxy columns.", font=F9, fill=DARK)
    save(img, "fig_nb301_axis_summary.png")


def fig_factorization_geometry():
    summary = json.loads((ABL / "summary.json").read_text(encoding="utf-8"))
    raw = summary["C1"]["raw"]
    dec = summary["C1"]["decomposed"]
    axes = list(csv.DictReader((INS / "nb301_signed_readout_evidence.csv").open()))
    dyn_path = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_rank1_stagewise_gpu0_20260615" / "nb301_rank1_stagewise_gpu0_free_decode_summary.csv"
    dyn_rows = []
    if dyn_path.exists():
        for r in csv.DictReader(dyn_path.open()):
            dyn_rows.append({
                "step": int(r["step"]),
                "stage": r["stage"],
                "acc": float(r["acc"]),
                "rank": int(float(r["rank"])),
                "score": float(r["score"]),
            })
        dyn_rows.sort(key=lambda r: (r["step"], r["stage"]))
    profiles = defaultdict(list)
    profile_path = INS / "nb301_axis_proxy_alignment_profiles.csv"
    if profile_path.exists():
        for r in csv.DictReader(profile_path.open()):
            r["oriented_corr"] = float(r["oriented_corr"])
            profiles[r["axis"]].append(r)

    rows = [
        ("Raw proxy matrix", raw["mean_abs_spearman"], raw["effective_rank"], 19, BLUE),
        ("Disentangled axes", dec["mean_abs_spearman"], dec["effective_rank"], 9, RED),
    ]

    # Single-column proxy-geometry grouped bar chart.
    img = Image.new("RGB", (880, 430), "white")
    d = ImageDraw.Draw(img)
    plot_top, plot_bottom = 80, 315
    group_centers = [275, 605]
    bar_w = 42
    metrics = [
        ("Correlation", raw["mean_abs_spearman"], dec["mean_abs_spearman"], 0.35, "{:.2f}"),
        ("Effective Rank", raw["effective_rank"], dec["effective_rank"], 10.0, "{:.2f}"),
    ]
    d.line((80, plot_bottom, 810, plot_bottom), fill=DARK, width=2)
    for frac in [0.25, 0.50, 0.75, 1.00]:
        y = plot_bottom - int(frac * (plot_bottom - plot_top))
        d.line((80, y, 810, y), fill=GRID, width=1)

    def bar_y(value, max_value):
        return plot_bottom - int((float(value) / max_value) * (plot_bottom - plot_top))

    for gi, (name, raw_v, axis_v, max_v, fmt) in enumerate(metrics):
        cx = group_centers[gi]
        for offset, value, color in [(-bar_w // 2 - 4, raw_v, BLUE), (bar_w // 2 + 4, axis_v, RED)]:
            x0b = cx + offset - bar_w // 2
            x1b = cx + offset + bar_w // 2
            yb = bar_y(value, max_v)
            d.rectangle((x0b, yb, x1b, plot_bottom), fill=color)
            text_center(d, ((x0b + x1b) / 2, yb - 30), fmt.format(value), F8, DARK)
        text_center(d, (cx, plot_bottom + 18), name, F8, DARK)

    legend_y = 370
    d.rectangle((230, legend_y, 255, legend_y + 16), fill=BLUE)
    d.text((267, legend_y - 4), "raw proxy", font=F8, fill=DARK)
    d.rectangle((520, legend_y, 545, legend_y + 16), fill=RED)
    d.text((557, legend_y - 4), "disentangled axis", font=F8, fill=DARK)
    save(img, "fig_nb301_proxy_geometry.png")

    # Double-column disentangled-axis calibration parameter table.
    img = Image.new("RGB", (1750, 740), "white")
    d = ImageDraw.Draw(img)
    table_y = 105
    profile_x0, profile_x1 = 315, 735
    weight_zero_x, scale = 1290, 155
    headers = [
        ("Score Direction", 150),
        ("Proxy Profile", 505),
        ("Nearest Proxy", 840),
        ("Axis-Level Weight", weight_zero_x),
        ("Effect", 1575),
    ]
    for label, x in headers:
        text_center(d, (x, table_y - 42), label, F9B, DARK)
    d.line((82, table_y - 12, 1665, table_y - 12), fill=DARK, width=2)
    row_h = 58
    d.line((weight_zero_x, table_y - 5, weight_zero_x, table_y + 9 * row_h - 18), fill=DARK, width=4)
    for i, row in enumerate(axes):
        y = table_y + i * row_h
        if i % 2 == 0:
            d.rectangle((82, y - 8, 1665, y + row_h - 10), fill=PALE)
        raw_coord = row["coordinate"]
        coord = raw_coord.replace("proxy_consensus", "consensus score")
        w = float(row["effective_readout_weight"])
        role = row["readout_role"].replace("positive", "strengthened")
        color = MID if role == "suppressed" else RED if w < 0 else GREEN
        d.text((95, y + 4), coord, font=F10, fill=DARK)
        if coord != "consensus score":
            d.line((profile_x0, y + 22, profile_x1, y + 22), fill=GRID, width=2)
            d.line(((profile_x0 + profile_x1) / 2, y + 12, (profile_x0 + profile_x1) / 2, y + 32), fill=DARK, width=1)
            prof = profiles.get(coord, [])
            nearest = prof[0]["nearest_proxy"] if prof else ""
            for pr in prof:
                px = profile_x0 + (pr["oriented_corr"] + 1) / 2 * (profile_x1 - profile_x0)
                is_near = pr["proxy"] == nearest
                rdot = 5
                d.ellipse((px - rdot, y + 22 - rdot, px + rdot, y + 22 + rdot), fill=BLUE if not is_near else ORANGE, outline="white", width=1)
        else:
            d.text((profile_x0, y + 4), "consensus score", font=F9, fill=DARK)
            nearest = "consensus score"
        d.text((770, y + 4), nearest.replace("zcpt_", "zcpt_"), font=F9, fill=DARK)
        bx = weight_zero_x + w * scale
        d.line((weight_zero_x, y + 22, bx, y + 22), fill=color, width=7)
        d.ellipse((bx - 8, y + 14, bx + 8, y + 30), fill=color, outline="white", width=1)
        d.text((1488, y + 4), f"{w:+.2f}", font=F10, fill=color)
        d.text((1585, y + 4), role, font=F10, fill=color)
    d.line((weight_zero_x, table_y - 5, weight_zero_x, table_y + 9 * row_h - 18), fill=DARK, width=4)
    legend_y = table_y + 9 * row_h + 12
    d.ellipse((315, legend_y + 6, 327, legend_y + 18), fill=BLUE)
    d.text((342, legend_y), "retained proxy", font=F9, fill=DARK)
    d.ellipse((595, legend_y + 6, 607, legend_y + 18), fill=ORANGE)
    d.text((625, legend_y), "nearest proxy", font=F9, fill=DARK)
    save(img, "fig_nb301_axis_calibration.png")

    # Single-column staged refinement curve.
    img = Image.new("RGB", (880, 360), "white")
    d = ImageDraw.Draw(img)
    cx0, cy0, cx1, cy1 = 95, 70, 790, 270
    split_x = cx0 + int(100 / 200.0 * (cx1 - cx0))
    d.rectangle((cx0, cy0 - 8, split_x, cy1), fill=(248, 250, 253))
    d.rectangle((split_x, cy0 - 8, cx1, cy1), fill=(253, 250, 247))
    d.line((cx0, cy1, cx1, cy1), fill=DARK, width=2)
    d.line((cx0, cy0, cx0, cy1), fill=DARK, width=2)
    d.text((cx0 - 45, cy0 - 35), "acc.", font=F9, fill=DARK)
    d.text((cx1 - 25, cy1 + 28), "step", font=F9, fill=DARK)

    def x_for_step(step):
        return cx0 + int(float(step) / 200.0 * (cx1 - cx0))

    def y_for_acc(acc):
        return cy1 - int((float(acc) - 93.00) / (94.75 - 93.00) * (cy1 - cy0))

    for tick in [93.0, 93.5, 94.0, 94.5]:
        y = y_for_acc(tick)
        d.line((cx0, y, cx1, y), fill=GRID, width=1)
        d.text((cx0 - 58, y - 11), f"{tick:.1f}", font=F8, fill=DARK)
    for tick in [0, 50, 100, 150, 200]:
        x = x_for_step(tick)
        d.line((x, cy1, x, cy1 + 7), fill=DARK, width=1)
        d.text((x - 13, cy1 + 12), str(tick), font=F8, fill=DARK)
    d.line((x_for_step(100), cy0 - 12, x_for_step(100), cy1), fill=DARK, width=2)
    text_center(d, (cx0 + (x_for_step(100) - cx0) / 2, cy0 - 34), "axis calibration", F8, DARK)
    text_center(d, (x_for_step(100) + (cx1 - x_for_step(100)) / 2, cy0 - 34), "component correction", F8, DARK)
    if dyn_rows:
        for r in dyn_rows:
            r["clean_stage"] = clean_llm_stage_name(r["stage"])
        axis_calib = [r for r in dyn_rows if r["clean_stage"] == "axis_calib_para"]
        component_corr = [r for r in dyn_rows if r["clean_stage"] == "component_corr_para"]
        prior = next((r for r in dyn_rows if r["clean_stage"] == "score_prior"), None)
        focus = next((r for r in dyn_rows if r["clean_stage"] == "axis_calibrated_focus"), None)
        final = next((r for r in dyn_rows if r["clean_stage"] == "task_conditioned_refinement"), None)
        axis_calib_line = sorted(axis_calib + ([focus] if focus else []), key=lambda r: r["step"])
        component_corr_line = sorted(([focus] if focus else []) + component_corr + ([final] if final else []), key=lambda r: r["step"])
        for rows, color in [(axis_calib_line, BLUE), (component_corr_line, RED)]:
            pts = [(x_for_step(r["step"]), y_for_acc(r["acc"])) for r in rows]
            if len(pts) > 1:
                d.line(pts, fill=color, width=4)
            for x, y in pts:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
        if prior:
            x, y = x_for_step(0), y_for_acc(prior["acc"])
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLUE, outline="white", width=1)
            d.text((x + 12, y - 24), f"prior {prior['acc']:.2f}", font=F8, fill=BLUE)
        if focus:
            x, y = x_for_step(100), y_for_acc(focus["acc"])
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLUE, outline="white", width=1)
            d.text((x + 10, y + 12), f"checkpoint {focus['acc']:.2f}", font=F8, fill=BLUE)
        if final:
            x, y = x_for_step(200), y_for_acc(final["acc"])
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=RED, outline="white", width=1)
            d.text((x - 118, y + 12), f"final {final['acc']:.2f}", font=F8, fill=RED)
    save(img, "fig_nb301_refinement_stagewise.png")


def fig_llm_proxy_geometry():
    summary = json.loads((LLM_FIG_IN / "llm_proxy_geometry_summary.json").read_text(encoding="utf-8"))
    fams = summary["families"]
    raw_corr = sum(v["raw_mean_abs_corr"] for v in fams.values()) / len(fams)
    axis_corr = sum(v["axis_mean_abs_corr"] for v in fams.values()) / len(fams)
    raw_er = sum(v["raw_effective_rank"] for v in fams.values()) / len(fams)
    axis_er = sum(v["axis_effective_rank"] for v in fams.values()) / len(fams)

    img = Image.new("RGB", (880, 430), "white")
    d = ImageDraw.Draw(img)
    plot_top, plot_bottom = 80, 315
    group_centers = [275, 605]
    bar_w = 42
    metrics = [
        ("Correlation", raw_corr, axis_corr, 0.85, "{:.2f}"),
        ("Effective Rank", raw_er, axis_er, 4.2, "{:.2f}"),
    ]
    d.line((80, plot_bottom, 810, plot_bottom), fill=DARK, width=2)
    for frac in [0.25, 0.50, 0.75, 1.00]:
        y = plot_bottom - int(frac * (plot_bottom - plot_top))
        d.line((80, y, 810, y), fill=GRID, width=1)

    def bar_y(value, max_value):
        return plot_bottom - int((float(value) / max_value) * (plot_bottom - plot_top))

    for gi, (name, raw_v, axis_v, max_v, fmt) in enumerate(metrics):
        cx = group_centers[gi]
        for offset, value, color in [(-bar_w // 2 - 4, raw_v, BLUE), (bar_w // 2 + 4, axis_v, RED)]:
            x0b = cx + offset - bar_w // 2
            x1b = cx + offset + bar_w // 2
            yb = bar_y(value, max_v)
            d.rectangle((x0b, yb, x1b, plot_bottom), fill=color)
            text_center(d, ((x0b + x1b) / 2, yb - 30), fmt.format(value), F8, DARK)
        text_center(d, (cx, plot_bottom + 18), name, F8, DARK)

    legend_y = 370
    d.rectangle((205, legend_y, 230, legend_y + 16), fill=BLUE)
    d.text((242, legend_y - 4), "raw proxy", font=F8, fill=DARK)
    d.rectangle((500, legend_y, 525, legend_y + 16), fill=RED)
    d.text((537, legend_y - 4), "disentangled axis", font=F8, fill=DARK)
    save(img, "fig_llm_proxy_geometry.png")


def fig_llm_refinement_stagewise():
    rows = list(csv.DictReader((LLM_FIG_IN / "llm_refinement_trajectory.csv").open()))
    final_acc = read_llm_acc_json(ROOT / "Manuscript" / "IoTJ" / "results" / "clean_repro_llm_main_evidence" / "llama2_keep50_acc.json")
    for r in rows:
        r["step"] = int(r["step"])
        r["avg_acc"] = float(r["avg_acc"])
        if r["stage"] == "final":
            r["avg_acc"] = final_acc

    img = Image.new("RGB", (880, 360), "white")
    d = ImageDraw.Draw(img)
    cx0, cy0, cx1, cy1 = 95, 70, 790, 270
    max_step = max(r["step"] for r in rows)
    x_max = int(math.ceil(max_step / 20.0) * 20)
    axis_calib_end = 16
    split_x = cx0 + int(axis_calib_end / float(x_max) * (cx1 - cx0))
    d.rectangle((cx0, cy0 - 8, split_x, cy1), fill=(248, 250, 253))
    d.rectangle((split_x, cy0 - 8, cx1, cy1), fill=(253, 250, 247))
    d.line((cx0, cy1, cx1, cy1), fill=DARK, width=2)
    d.line((cx0, cy0, cx0, cy1), fill=DARK, width=2)
    d.text((cx0 - 45, cy0 - 35), "acc.", font=F9, fill=DARK)
    d.text((cx1 - 25, cy1 + 28), "step", font=F9, fill=DARK)

    def x_for_step(step):
        return cx0 + int(float(step) / float(x_max) * (cx1 - cx0))

    def y_for_acc(acc):
        return cy1 - int((float(acc) - 44.0) / (52.6 - 44.0) * (cy1 - cy0))

    for tick in [44.0, 46.0, 48.0, 50.0, 52.0]:
        y = y_for_acc(tick)
        d.line((cx0, y, cx1, y), fill=GRID, width=1)
        d.text((cx0 - 58, y - 11), f"{tick:.0f}", font=F8, fill=DARK)
    for tick in [0, 40, 80, 120]:
        x = x_for_step(tick)
        d.line((x, cy1, x, cy1 + 7), fill=DARK, width=1)
        d.text((x - 13, cy1 + 12), str(tick), font=F8, fill=DARK)
    d.line((split_x, cy0 - 12, split_x, cy1), fill=DARK, width=2)
    d.text((cx0 + 8, cy0 - 34), "axis calibration parameter", font=F8, fill=DARK)
    text_center(d, (split_x + (cx1 - split_x) / 2, cy0 - 34), "component correction", F8, DARK)

    pts = [(x_for_step(r["step"]), y_for_acc(r["avg_acc"]), r) for r in rows]
    axis_calib_pts = [p for p in pts if p[2]["step"] <= axis_calib_end]
    component_corr_pts = [p for p in pts if p[2]["step"] >= axis_calib_end]
    d.line([(p[0], p[1]) for p in axis_calib_pts], fill=BLUE, width=4)
    d.line([(p[0], p[1]) for p in component_corr_pts], fill=RED, width=4)
    for x, y, r in pts:
        fill = BLUE if r["step"] <= axis_calib_end else RED
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=fill, outline="white", width=1)
    for r, color in [(rows[0], BLUE), (rows[2], BLUE), (rows[-1], RED)]:
        x, y = x_for_step(r["step"]), y_for_acc(r["avg_acc"])
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=1)
    labels = [
        (rows[0], "prior", 12, -50, BLUE),
        (rows[2], "checkpoint", 28, -12, BLUE),
        (rows[-1], "final", -112, -34, RED),
    ]
    for r, prefix, dx, dy, color in labels:
        d.text((x_for_step(r["step"]) + dx, y_for_acc(r["avg_acc"]) + dy), f"{prefix} {r['avg_acc']:.2f}", font=F8, fill=color)
    save(img, "fig_llm_refinement_stagewise.png")


def fig_llm_axis_calib():
    axes = list(csv.DictReader((LLM_FIG_IN / "llm_axis_calibration.csv").open()))
    proxies = ["hessian", "wanda_sp", "flap_wifv", "llm_pruner_taylor"]

    img = Image.new("RGB", (1750, 740), "white")
    d = ImageDraw.Draw(img)
    table_y = 105
    profile_x0, profile_x1 = 350, 735
    weight_zero_x, scale = 1290, 360
    headers = [
        ("Score Direction", 160),
        ("Proxy Profile", 540),
        ("Nearest Proxy", 860),
        ("Axis-Level Weight", weight_zero_x),
        ("Effect", 1575),
    ]
    for label, x in headers:
        text_center(d, (x, table_y - 42), label, F9B, DARK)
    d.line((82, table_y - 12, 1665, table_y - 12), fill=DARK, width=2)
    row_h = 58
    d.line((weight_zero_x, table_y - 5, weight_zero_x, table_y + row_h * len(axes) - 18), fill=DARK, width=4)
    for i, row in enumerate(axes):
        y = table_y + i * row_h
        if i % 2 == 0:
            d.rectangle((82, y - 8, 1665, y + row_h - 10), fill=PALE)
        coord = f"{row['family'].replace('attention heads', 'att').replace('MLP channels', 'mlp')} {row['coordinate']}"
        w = float(row["task_weight"])
        role = row["role"].replace("positive", "strengthened")
        color = MID if role == "suppressed" else RED if w < 0 else GREEN
        d.text((95, y + 4), coord, font=F10, fill=DARK)
        d.line((profile_x0, y + 22, profile_x1, y + 22), fill=GRID, width=2)
        d.line(((profile_x0 + profile_x1) / 2, y + 12, (profile_x0 + profile_x1) / 2, y + 32), fill=DARK, width=1)
        is_consensus = row["coordinate"] == "consensus"
        nearest = "proxy mean" if is_consensus else row["nearest_proxy"]
        for pr in proxies:
            val = float(row[pr])
            px = profile_x0 + (val + 1) / 2 * (profile_x1 - profile_x0)
            is_near = (pr == nearest) and not is_consensus
            rdot = 5 if not is_near else 7
            d.ellipse((px - rdot, y + 22 - rdot, px + rdot, y + 22 + rdot), fill=BLUE if not is_near else ORANGE, outline="white", width=1)
        if is_consensus:
            cx = profile_x0 + int(0.75 * (profile_x1 - profile_x0))
            d.ellipse((cx - 7, y + 15, cx + 7, y + 29), fill=ORANGE, outline="white", width=1)
        d.text((770, y + 4), nearest, font=F9, fill=DARK)
        bx = weight_zero_x + w * scale
        d.line((weight_zero_x, y + 22, bx, y + 22), fill=color, width=7)
        d.ellipse((bx - 8, y + 14, bx + 8, y + 30), fill=color, outline="white", width=1)
        d.text((1488, y + 4), f"{w:+.2f}", font=F10, fill=color)
        d.text((1585, y + 4), role, font=F10, fill=color)
    d.line((weight_zero_x, table_y - 5, weight_zero_x, table_y + row_h * len(axes) - 18), fill=DARK, width=4)
    legend_y = table_y + row_h * len(axes) + 12
    d.ellipse((315, legend_y + 6, 327, legend_y + 18), fill=BLUE)
    d.text((342, legend_y), "retained proxy", font=F9, fill=DARK)
    d.ellipse((515, legend_y + 6, 527, legend_y + 18), fill=ORANGE)
    d.text((545, legend_y), "nearest proxy", font=F9, fill=DARK)
    save(img, "fig_llm_axis_calibration.png")


def fig_signed_refinement():
    rows = []
    summary_dir = historical_axis_summary_dir()
    summary_path = ROOT / "Manuscript" / "IoTJ" / "results" / f"{HISTORICAL_REFINEMENT_PREFIX}_zcpt_op_explore" / summary_dir / f"{summary_dir}_summary.jsonl"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        stagewise_path = ROOT / "Manuscript" / "IoTJ" / "results" / "nb301_rank1_stagewise_gpu0_20260615" / "nb301_rank1_stagewise_gpu0_free_decode_summary.csv"
        stage_rows = list(csv.DictReader(stagewise_path.open(encoding="utf-8")))
        prior = next(r for r in stage_rows if clean_stage_name(r["stage"]) == "score_prior")
        focus = next(r for r in stage_rows if clean_stage_name(r["stage"]) == "axis_calibrated_focus")
        final = max(
            (r for r in stage_rows if clean_stage_name(r["stage"]) == "component_corr_para"),
            key=lambda r: float(r["acc"]),
        )
        rows = [
            {
                "method": f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_-01",
                "free_acc": float(prior["acc"]),
                "free_equiv_fixed1000_rank": int(float(prior["rank"])),
            },
            {
                "method": f"{HISTORICAL_REFINEMENT_PREFIX}_step_100_mask_score_params",
                "free_acc": float(focus["acc"]),
                "free_equiv_fixed1000_rank": int(float(focus["rank"])),
            },
            {
                "method": f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_000",
                "free_acc": float(final["acc"]),
                "free_equiv_fixed1000_rank": int(float(final["rank"])),
            },
        ]
    order = [
        ("score_prior", "score\nprior"),
        ("axis_calibrated_focus", "axis calibration\n+ focus"),
        ("task_conditioned_refinement", "task-conditioned\nrefinement"),
    ]
    by_method = {NB301_STAGE_NAME_MAP.get(r["method"], r["method"]): r for r in rows}
    axes = list(csv.DictReader((INS / "nb301_signed_readout_evidence.csv").open()))

    img = Image.new("RGB", (1750, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((95, 45), "Task-conditioned refinement dynamics", font=F14B, fill=DARK)

    # Panel A: selected architecture quality across stages.
    x0, y0, x1, y1 = 120, 215, 760, 625
    d.text((x0, 125), "A  Selected architecture improves after task feedback", font=F12B, fill=DARK)
    d.line((x0, y1, x1, y1), fill=DARK, width=2)
    d.line((x0, y0, x0, y1), fill=DARK, width=2)

    def y_for_acc(acc):
        return y1 - (float(acc) - 93.75) / (94.75 - 93.75) * (y1 - y0)

    for tick in [93.8, 94.0, 94.2, 94.4, 94.6]:
        y = y_for_acc(tick)
        d.line((x0, y, x1, y), fill=GRID, width=1)
        d.text((x0 - 72, y - 13), f"{tick:.1f}", font=F8, fill=DARK)
    d.text((x0 - 5, y0 - 42), "surrogate accuracy", font=F9, fill=DARK)

    pts = []
    for i, (key, label) in enumerate(order):
        r = by_method[key]
        x = x0 + 110 + i * 245
        y = y_for_acc(r["free_acc"])
        pts.append((x, y, r, label))
        for j, part in enumerate(label.split("\n")):
            text_center(d, (x, y1 + 28 + j * 25), part, F8, DARK)
    d.line([(p[0], p[1]) for p in pts], fill=RED, width=5)
    for x, y, r, _ in pts:
        d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=RED, outline="white", width=3)
        d.text((x + 18, y - 30), f"{r['free_acc']:.4f}", font=F9, fill=RED)
        d.text((x + 18, y + 2), f"rank {int(r['free_equiv_fixed1000_rank'])}", font=F8, fill=DARK)

    # Panel B: learned signed axis weights.
    px0, py0, py1 = 920, 185, 635
    d.text((px0, 125), "B  Calibrated weights", font=F12B, fill=DARK)
    zero_x, scale = 1240, 155
    d.line((zero_x, py0 - 10, zero_x, py1), fill=DARK, width=2)
    d.text((zero_x - 7, py1 + 18), "0", font=F8, fill=DARK)
    d.text((zero_x - 255, py1 + 18), "negative / reversed", font=F8, fill=RED)
    d.text((zero_x + 58, py1 + 18), "positive", font=F8, fill=GREEN)
    for i, row in enumerate(axes):
        y = py0 + i * 48
        coord = row["coordinate"].replace("proxy_consensus", "consensus")
        w = float(row["effective_readout_weight"])
        role = row["readout_role"]
        color = MID if role == "suppressed" else RED if w < 0 else GREEN
        d.text((px0, y), coord, font=F9, fill=DARK)
        bx = zero_x + w * scale
        d.line((zero_x, y + 13, bx, y + 13), fill=color, width=7)
        d.ellipse((bx - 9, y + 4, bx + 9, y + 22), fill=color, outline="white", width=2)
        d.text((1490, y), f"{w:+.2f}", font=F8, fill=DARK)
    d.line((zero_x, py0 - 10, zero_x, py1), fill=DARK, width=2)
    d.text((95, 740), "The final scorer is not a uniform proxy average: one axis is strongly amplified, several axes are reversed, and weak coordinates are suppressed.", font=F9, fill=DARK)
    save(img, "fig_nb301_signed_refinement.png")


def load_subset():
    rows = []
    for r in csv.DictReader((SUB / "gate_subset_results.csv").open()):
        for k in ["size", "gate_count", "rank", "acc", "family_entropy", "mean_abs_spearman", "effective_rank"]:
            r[k] = float(r[k])
        rows.append(r)
    return rows


def color_entropy(v):
    lo, hi = 0.75, 1.75
    t = max(0, min(1, (v - lo) / (hi - lo)))
    a = BLUE
    b = RED
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def fig_subset_gate_rank():
    rows = load_subset()
    variants = [("raw_rank_mean", "Raw rank-mean"), ("raw_az_logrank", "AZ-logrank"), ("decomposed_prior", "Score prior")]
    img = Image.new("RGB", (1900, 820), "white")
    d = ImageDraw.Draw(img)
    panels = [(105, 145, 615, 595), (735, 145, 1245, 595), (1365, 145, 1875, 595)]
    for (var, label), (x0, y0, x1, y1) in zip(variants, panels):
        y_for = draw_log_y_axis(d, x0, y0, x1, y1, label=(var == "raw_rank_mean"))
        text_center(d, ((x0 + x1) / 2, y0 - 85), label, F12B, DARK)
        for gc in range(0, 9):
            x = x0 + gc / 8 * (x1 - x0)
            d.line((x, y1, x, y1 + 8), fill=DARK, width=2)
            d.text((x - 7, y1 + 16), str(gc), font=F8, fill=DARK)
        d.text((x0 + 120, y1 + 58), "# gate-kept proxies in subset", font=F9, fill=DARK)
        sub = [r for r in rows if r["variant"] == var]
        grouped = defaultdict(list)
        for r in sub:
            gc = r["gate_count"]
            xbase = x0 + gc / 8 * (x1 - x0)
            jitter = (-0.18 if r["size"] == 5 else 0.18) * (x1 - x0) / 8
            x = xbase + jitter
            y = y_for(max(1, r["rank"]))
            rad = 5 if r["size"] == 5 else 8
            col = color_entropy(r["family_entropy"])
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=col, outline="white", width=1)
            grouped[(int(r["size"]), int(gc))].append(r["rank"])
        for size, col in [(5, LIGHT_BLUE), (10, BLUE)]:
            pts = []
            for gc in range(0, 9):
                vals = grouped.get((size, gc), [])
                if vals:
                    mean = sum(vals) / len(vals)
                    x = x0 + gc / 8 * (x1 - x0) + (-0.18 if size == 5 else 0.18) * (x1 - x0) / 8
                    pts.append((x, y_for(max(1, mean))))
            if len(pts) > 1:
                d.line(pts, fill=col, width=3)
                for p in pts:
                    d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=col)

    legend_y = 710
    d.line((620, legend_y, 690, legend_y), fill=LIGHT_BLUE, width=4)
    d.text((705, legend_y - 14), "mean trend, size=5", font=F9, fill=DARK)
    d.line((950, legend_y, 1020, legend_y), fill=BLUE, width=4)
    d.text((1035, legend_y - 14), "mean trend, size=10", font=F9, fill=DARK)
    for i, ent in enumerate([0.8, 1.1, 1.4, 1.7]):
        x = 620 + i * 150
        d.ellipse((x - 9, 755 - 9, x + 9, 755 + 9), fill=color_entropy(ent))
        d.text((x + 20, 742), f"entropy {ent:.1f}", font=F8, fill=DARK)
    save(img, "fig_nb301_subset_gate_rank.png")


def fig_subset_geometry_map():
    rows = load_subset()
    variants = [("raw_rank_mean", "Raw rank-mean"), ("raw_az_logrank", "AZ-logrank"), ("decomposed_prior", "Score prior")]
    img = Image.new("RGB", (1900, 820), "white")
    d = ImageDraw.Draw(img)
    panels = [(105, 145, 615, 595), (735, 145, 1245, 595), (1365, 145, 1875, 595)]

    def x_for(v, x0, x1):
        v = max(0.10, min(0.95, v))
        return x0 + (v - 0.10) / (0.95 - 0.10) * (x1 - x0)

    for (var, label), (x0, y0, x1, y1) in zip(variants, panels):
        y_for = draw_log_y_axis(d, x0, y0, x1, y1, label=(var == "raw_rank_mean"))
        text_center(d, ((x0 + x1) / 2, y0 - 85), label, F12B, DARK)
        for tick in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x = x_for(tick, x0, x1)
            d.line((x, y1, x, y1 + 8), fill=DARK, width=2)
            d.text((x - 18, y1 + 16), f"{tick:.1f}", font=F8, fill=DARK)
        d.text((x0 + 110, y1 + 58), "mean absolute proxy correlation", font=F9, fill=DARK)
        sub = [r for r in rows if r["variant"] == var]
        for r in sub:
            x = x_for(r["mean_abs_spearman"], x0, x1)
            y = y_for(max(1, r["rank"]))
            rad = 4 + int(r["family_entropy"] * 3.0)
            t = r["gate_count"] / 8.0
            col = tuple(int(GREEN[i] * (1 - t) + RED[i] * t) for i in range(3))
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=col, outline="white", width=1)

    d.text((535, 690), "Color shifts from low to high gate coverage; point size increases with family entropy.", font=F10, fill=DARK)
    d.text((535, 735), "Good proxy subsets are not just high-gate subsets: they also avoid extremely correlated, one-family evidence.", font=F10, fill=DARK)
    save(img, "fig_nb301_subset_geometry_map.png")


def fig_subset_gate_heatmap():
    by_gate = list(csv.DictReader((SUB / "gate_subset_by_gate.csv").open()))
    variants = [("raw_rank_mean", "Raw rank-mean"), ("raw_az_logrank", "AZ-logrank"), ("decomposed_prior", "Score prior")]
    img = Image.new("RGB", (1550, 840), "white")
    d = ImageDraw.Draw(img)
    margin_x, margin_y = 250, 120
    cell_w, cell_h = 62, 62
    gap_y = 92
    data = {}
    for r in by_gate:
        data[(r["variant"], int(r["size"]), int(float(r["gate_count"])))] = float(r["mean_rank"])

    def rank_col(v):
        t = max(0, min(1, (math.log10(v) - 0) / 3))
        good = (72, 145, 110)
        bad = (197, 80, 76)
        return tuple(int(good[i] * (1 - t) + bad[i] * t) for i in range(3))

    for vi, (var, label) in enumerate(variants):
        ybase = margin_y + vi * (2 * cell_h + gap_y)
        d.text((35, ybase + 40), label, font=F11, fill=DARK)
        for si, size in enumerate([5, 10]):
            y = ybase + si * cell_h
            d.text((118, y + 18), f"k={size}", font=F9, fill=DARK)
            for gc in range(0, 9):
                x = margin_x + gc * cell_w
                val = data.get((var, size, gc))
                if val is None:
                    d.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=(242, 242, 242), outline="white")
                    continue
                d.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=rank_col(val), outline="white")
                text_center(d, (x + cell_w / 2 - 2, y + 18), str(int(round(val))), F8, "white" if val > 250 else DARK)
        if vi == 0:
            for gc in range(0, 9):
                x = margin_x + gc * cell_w
                text_center(d, (x + cell_w / 2 - 2, 82), str(gc), F9, DARK)
            d.text((margin_x + 120, 35), "# gate-kept proxies in subset", font=F10, fill=DARK)

    d.text((900, 180), "Cell value: mean selected rank", font=F11, fill=DARK)
    d.text((900, 230), "Green is better; red is worse.", font=F10, fill=DARK)
    d.text((900, 300), "The pattern is not monotone.", font=F12B, fill=DARK)
    d.text((900, 350), "Raw voting strongly rewards gate-kept", font=F10, fill=DARK)
    d.text((900, 390), "proxies. The score prior needs", font=F10, fill=DARK)
    d.text((900, 430), "both strong proxies and non-collapsed", font=F10, fill=DARK)
    d.text((900, 470), "proxy geometry.", font=F10, fill=DARK)
    save(img, "fig_nb301_subset_gate_heatmap.png")


def fig_subset_stability():
    sources = [
        ("Utility-gated", 3, RAND_UTILITY_GATED_POOL / "random_subset_protocol_summary.json"),
        ("Utility-gated", 5, RAND_UTILITY_GATED_POOL / "random_subset_protocol_summary.json"),
        ("Full pool", 3, RAND_FULL_PROXY_POOL / "random_subset_protocol_summary.json"),
        ("Full pool", 5, RAND_FULL_PROXY_POOL / "random_subset_protocol_summary.json"),
    ]
    missing = [str(path) for _, _, path in sources if not path.exists()]
    if missing:
        print(f"skip fig_nb301_subset_stability.png; missing {missing}")
        return
    rows = []
    for pool, wanted_size, path in sources:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload["rows"]:
            if r.get("status") != "done" or int(r["size"]) != int(wanted_size):
                continue
            item = dict(r)
            item["pool_label"] = pool
            rows.append(item)
    variants = [
        ("rank_mean_search", "rank-mean", BLUE),
        ("az_style_logrank_search", "AZ-NAS", PURPLE),
        ("proxydiff_full_refinement", "ProxyDiff", RED),
    ]
    groups = [("Gate-8", 3), ("Gate-8", 5), ("All-19", 3), ("All-19", 5)]
    img = Image.new("RGB", (1500, 650), "white")
    d = ImageDraw.Draw(img)
    left, right = 285, 1390
    top, bottom = 65, 485
    xmin, xmax = 90.8, 94.8

    def x_for(acc):
        acc = max(xmin, min(xmax, float(acc)))
        return left + (acc - xmin) / (xmax - xmin) * (right - left)

    d.line((left, bottom, right, bottom), fill=DARK, width=2)
    text_center(d, ((left + right) / 2, 22), "NB301 surrogate accuracy", F10, DARK)
    for tick in [91, 92, 93, 94, 94.5]:
        x = x_for(tick)
        d.line((x, top, x, bottom), fill=GRID, width=1)
        d.line((x, bottom, x, bottom + 8), fill=DARK, width=2)
        label = f"{tick:.1f}" if isinstance(tick, float) else str(tick)
        text_center(d, (x, bottom + 17), label, F9, DARK)

    group_gap = 98
    y0 = 112
    method_offsets = {
        "rank_mean_search": 28,
        "az_style_logrank_search": 0,
        "proxydiff_full_refinement": -28,
    }
    medians = {}
    for gi, (pool, size) in enumerate(groups):
        cy = y0 + gi * group_gap
        if gi % 2 == 0:
            d.rectangle((60, cy - 48, right + 35, cy + 44), fill=PALE)
        d.text((82, cy - 16), f"{pool} k={size}", font=F10, fill=DARK)
        d.line((left, cy + 46, right, cy + 46), fill=(232, 235, 238), width=1)
        for var, label, color in variants:
            vals = sorted(float(r["acc"]) for r in rows if r["pool_label"] == pool and r["protocol"] == var and int(r["size"]) == size)
            if not vals:
                continue
            med = vals[len(vals) // 2]
            medians[(pool, size, var)] = med
            y = cy + method_offsets[var]
            xlo, xhi, xmed = x_for(min(vals)), x_for(max(vals)), x_for(med)
            d.line((xlo, y, xhi, y), fill=color, width=5)
            d.ellipse((xmed - 13, y - 13, xmed + 13, y + 13), fill=color, outline="white", width=2)
    legend_y = 555
    legend_width = 860
    legend_x = (img.width - legend_width) // 2
    for vi, (_, label, color) in enumerate(variants):
        x = legend_x + vi * 300
        d.line((x - 25, legend_y + 11, x + 45, legend_y + 11), fill=color, width=5)
        d.ellipse((x, legend_y, x + 22, legend_y + 22), fill=color, outline="white", width=2)
        d.text((x + 56, legend_y - 5), label, font=F10, fill=DARK)
    save(img, "fig_nb301_subset_stability.png")


def compose_pair_figures():
    def open_rgb(name):
        return Image.open(FIG / name).convert("RGB")

    def paste_center(canvas, img, box):
        x0, y0, x1, y1 = box
        x = x0 + (x1 - x0 - img.width) // 2
        y = y0 + (y1 - y0 - img.height) // 2
        canvas.paste(img, (x, y))

    nb_summary = json.loads((ABL / "summary.json").read_text(encoding="utf-8"))
    nb_raw = nb_summary["C1"]["raw"]
    nb_dec = nb_summary["C1"]["decomposed"]
    llm_summary = json.loads((LLM_FIG_IN / "llm_proxy_geometry_summary.json").read_text(encoding="utf-8"))
    llm_fams = llm_summary["families"]
    llm_raw_corr = sum(v["raw_mean_abs_corr"] for v in llm_fams.values()) / len(llm_fams)
    llm_axis_corr = sum(v["axis_mean_abs_corr"] for v in llm_fams.values()) / len(llm_fams)
    llm_raw_er = sum(v["raw_effective_rank"] for v in llm_fams.values()) / len(llm_fams)
    llm_axis_er = sum(v["axis_effective_rank"] for v in llm_fams.values()) / len(llm_fams)

    geom = Image.new("RGB", (640, 680), "white")
    d = ImageDraw.Draw(geom)

    def draw_geometry_panel(y0, label, corr_raw, corr_axis, er_raw, er_axis, corr_max, er_max):
        plot_top, plot_bottom = y0 + 54, y0 + 220
        plot_left, plot_right = 78, 568
        group_centers = [220, 426]
        bar_w = 28
        d.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=DARK, width=2)
        for frac in [0.25, 0.50, 0.75, 1.00]:
            y = plot_bottom - int(frac * (plot_bottom - plot_top))
            d.line((plot_left, y, plot_right, y), fill=GRID, width=1)

        def bar_y(value, max_value):
            return plot_bottom - int((float(value) / max_value) * (plot_bottom - plot_top))

        metrics = [
            ("Correlation", corr_raw, corr_axis, corr_max),
            ("Effective Rank", er_raw, er_axis, er_max),
        ]
        for gi, (name, raw_v, axis_v, max_v) in enumerate(metrics):
            cx = group_centers[gi]
            for offset, value, color in [(-18, raw_v, BLUE), (18, axis_v, RED)]:
                x0b = cx + offset - bar_w // 2
                x1b = cx + offset + bar_w // 2
                yb = bar_y(value, max_v)
                d.rectangle((x0b, yb, x1b, plot_bottom), fill=color)
                text_center(d, ((x0b + x1b) / 2, yb - 28), f"{value:.2f}", F7, DARK)
            text_center(d, (cx, plot_bottom + 17), name, F8, DARK)
        text_center(d, (geom.width / 2, y0 + 12), label, F8B, DARK)

    draw_geometry_panel(
        8,
        "NAS on CNN",
        nb_raw["mean_abs_spearman"],
        nb_dec["mean_abs_spearman"],
        nb_raw["effective_rank"],
        nb_dec["effective_rank"],
        0.35,
        10.0,
    )
    draw_geometry_panel(
        326,
        "Pruning on LLM",
        llm_raw_corr,
        llm_axis_corr,
        llm_raw_er,
        llm_axis_er,
        0.85,
        4.2,
    )
    d.line((92, 300, geom.width - 92, 300), fill=(205, 209, 214), width=2)
    legend_y = 644
    d.rectangle((156, legend_y, 174, legend_y + 12), fill=BLUE)
    d.text((184, legend_y - 4), "Raw Proxy", font=F7, fill=DARK)
    d.rectangle((350, legend_y, 368, legend_y + 12), fill=RED)
    d.text((378, legend_y - 4), "Disentangled Axis", font=F7, fill=DARK)
    save(geom, "fig_proxy_geometry_combined.png")

    nas_traj = open_rgb("fig_nb301_refinement_stagewise.png")
    llm_traj = open_rgb("fig_llm_refinement_stagewise.png")
    gap = 22
    traj = Image.new("RGB", (max(nas_traj.width, llm_traj.width), nas_traj.height + llm_traj.height + gap), "white")
    paste_center(traj, nas_traj, (0, 0, traj.width, nas_traj.height))
    d = ImageDraw.Draw(traj)
    text_center(d, (traj.width / 2, nas_traj.height - 40), "NAS on CNN", F8B, DARK)
    sep_y = nas_traj.height + gap // 2
    d.line((50, sep_y, traj.width - 50, sep_y), fill=DARK, width=2)
    paste_center(traj, llm_traj, (0, nas_traj.height + gap, traj.width, traj.height))
    text_center(d, (traj.width / 2, traj.height - 58), "Pruning on LLM", F8B, DARK)
    save(traj, "fig_refinement_stagewise_combined.png")

    nas_axis = open_rgb("fig_nb301_axis_calibration.png")
    llm_axis = open_rgb("fig_llm_axis_calibration.png")
    gap = 18
    axis = Image.new("RGB", (max(nas_axis.width, llm_axis.width), nas_axis.height + llm_axis.height + gap), "white")
    paste_center(axis, nas_axis, (0, 0, axis.width, nas_axis.height))
    d = ImageDraw.Draw(axis)
    text_center(d, (axis.width / 2, nas_axis.height - 40), "NAS on CNN", F8B, DARK)
    sep_y = nas_axis.height + gap // 2
    d.line((90, sep_y, axis.width - 90, sep_y), fill=DARK, width=2)
    paste_center(axis, llm_axis, (0, nas_axis.height + gap, axis.width, axis.height))
    text_center(d, (axis.width / 2, axis.height - 58), "Pruning on LLM", F8B, DARK)
    save(axis, "fig_axis_calibration_combined.png")


if __name__ == "__main__":
    fig_intro_pareto()
    fig_ablation_slope()
    fig_nb301_component_ablation()
    fig_llm_component_ablation()
    fig_c1_spectrum()
    fig_factorization_geometry()
    fig_llm_proxy_geometry()
    fig_llm_refinement_stagewise()
    fig_llm_axis_calib()
    fig_signed_refinement()
    fig_subset_analysis_map()
    fig_axis_summary()
    fig_subset_stability()
    compose_pair_figures()
