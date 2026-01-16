#!/usr/bin/env python3
"""Generate a time-depth (T-D) diagram from data/dives.csv."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial", "Heiti TC", "sans-serif"]


def load_dives(
    csv_path: Path, time_col: str, depth_col: str
) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if time_col not in row or depth_col not in row:
                raise KeyError(
                    f"Missing columns in CSV. Expected '{time_col}' and '{depth_col}'."
                )
            try:
                rows.append((float(row[time_col]), float(row[depth_col])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric value in columns '{time_col}' or '{depth_col}'."
                ) from exc
    return rows


@dataclass
class ExamplePoint:
    label: str
    time_s: float
    depth_m: float
    color: str


EXAMPLE_POINT_DATA = [
    ("personal_best_fim", 108.0, 45.0, "#ff7f0e"),
    ("slow_dive_30m_fim", 144.0, 30.0, "#2ca02c"),
    ("sta_4min", 244.0, 0.0, "#1f4e79"),
    ("sprint_dive_20m_cwtb", 28.0, 20.0, "#d62728"),
]

TRANSLATIONS = {
    "en": {
        "title": "Dive Time vs. Max Depth",
        "xlabel": "Dive Time (s)",
        "ylabel": "Max Depth (m)",
        "legend_records": "Dive Records",
        "legend_feasible": "Feasible Region",
        "legend_infeasible": "Infeasible Region",
        "examples": {
            "personal_best_fim": "Personal Best Dive (FIM)",
            "slow_dive_30m_fim": "Slow Dive ~30 M (FIM)",
            "sta_4min": "STA ~4 Min",
            "sprint_dive_20m_cwtb": "Sprint Dive ~20 M (CWTB)",
        },
    },
    "zh-tw": {
        "title": "潛水時間與最大深度",
        "xlabel": "潛水時間（秒）",
        "ylabel": "最大深度（米）",
        "legend_records": "潛水紀錄",
        "legend_feasible": "能力可及區域",
        "legend_infeasible": "能力不可及區域",
        "examples": {
            "personal_best_fim": "個人最佳潛水（FIM）",
            "slow_dive_30m_fim": "慢速潛水 約30米（FIM）",
            "sta_4min": "靜態閉氣 約4分鐘",
            "sprint_dive_20m_cwtb": "衝刺潛水 約20米（CWTB）",
        },
    },
}


def example_points_for_lang(lang: str) -> List[ExamplePoint]:
    labels = TRANSLATIONS[lang]["examples"]
    return [
        ExamplePoint(labels[key], time_s=time_s, depth_m=depth_m, color=color)
        for key, time_s, depth_m, color in EXAMPLE_POINT_DATA
    ]


def isotonic_regression(values: List[float], increasing: bool = True) -> List[float]:
    if not values:
        return []
    if not increasing:
        values = [-v for v in values]

    blocks = []
    for idx, val in enumerate(values):
        blocks.append([val, 1.0, idx, idx])
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]
            if (prev[0] / prev[1]) > (curr[0] / curr[1]):
                merged = [
                    prev[0] + curr[0],
                    prev[1] + curr[1],
                    prev[2],
                    curr[3],
                ]
                blocks[-2:] = [merged]
            else:
                break

    fitted = [0.0] * len(values)
    for total, weight, start, end in blocks:
        avg = total / weight
        for i in range(start, end + 1):
            fitted[i] = avg

    if not increasing:
        fitted = [-v for v in fitted]
    return fitted


def monotone_envelope_upper(values: List[float]) -> List[float]:
    if not values:
        return []
    envelope = [0.0] * len(values)
    current = float("inf")
    for idx in range(len(values) - 1, -1, -1):
        current = min(current, values[idx])
        envelope[idx] = current
    return envelope


def monotone_envelope_lower_decreasing(values: List[float]) -> List[float]:
    if not values:
        return []
    envelope = [0.0] * len(values)
    current = float("-inf")
    for idx in range(len(values) - 1, -1, -1):
        current = max(current, values[idx])
        envelope[idx] = current
    return envelope


def frontier_by_depth(
    rows: List[Tuple[float, float]], depth_bin: float, mode: str
) -> Tuple[List[float], List[float]]:
    if depth_bin <= 0:
        raise ValueError("depth_bin must be positive.")

    bins = {}
    for time_s, depth_m in rows:
        bin_idx = int(depth_m // depth_bin)
        bins.setdefault(bin_idx, []).append(time_s)

    depths = []
    times = []
    for bin_idx in sorted(bins):
        bin_times = bins[bin_idx]
        if mode == "fast":
            time_s = min(bin_times)
        elif mode == "slow":
            time_s = max(bin_times)
        else:
            raise ValueError("mode must be 'fast' or 'slow'.")
        depth_m = (bin_idx + 0.5) * depth_bin
        depths.append(depth_m)
        times.append(time_s)

    if mode == "fast":
        fitted = isotonic_regression(times, increasing=True)
        constrained = [min(f, t) for f, t in zip(fitted, times)]
        envelope = monotone_envelope_upper(constrained)
    else:
        fitted = isotonic_regression(times, increasing=False)
        constrained = [max(f, t) for f, t in zip(fitted, times)]
        envelope = monotone_envelope_lower_decreasing(constrained)
    return envelope, depths


def main():
    parser = argparse.ArgumentParser(description="Plot a T-D diagram from dives.csv.")
    parser.add_argument(
        "--input",
        default="data/dives.csv",
        help="Path to dives CSV (default: data/dives.csv)",
    )
    parser.add_argument(
        "--output",
        default="images/td-diagram.pdf",
        help="Output image path (default: images/td-diagram.pdf)",
    )
    parser.add_argument(
        "--time-col",
        default="duration",
        help="Column for dive time (default: duration)",
    )
    parser.add_argument(
        "--depth-col",
        default="max_depth",
        help="Column for max depth (default: max_depth)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=sorted(TRANSLATIONS.keys()),
        help="Language for plot labels (default: en)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title (default: language-specific)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Disable representative example markers.",
    )
    parser.add_argument(
        "--no-frontiers",
        action="store_true",
        help="Disable isotonic regression frontiers.",
    )
    parser.add_argument(
        "--depth-bin",
        type=float,
        default=1.0,
        help="Depth bin size in meters for frontiers (default: 1.0).",
    )
    parser.add_argument(
        "--figsize",
        default="10,4.0",
        help="Figure size as 'width,height' in inches (default: 10,4.0).",
    )
    args = parser.parse_args()

    try:
        fig_width, fig_height = (
            float(value.strip()) for value in args.figsize.split(",", maxsplit=1)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("figsize must be two numbers like '10,4.0'.") from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strings = TRANSLATIONS[args.lang]
    title = args.title or strings["title"]
    rows = load_dives(input_path, args.time_col, args.depth_col)
    times = [t for t, _ in rows]
    depths = [d for _, d in rows]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    max_depth = max(depths) if depths else 0.0
    ax.scatter(
        times,
        depths,
        s=18,
        alpha=0.55,
        color="#b8b8b8",
        edgecolors="none",
        label=strings["legend_records"],
    )
    if not args.no_examples:
        for example in example_points_for_lang(args.lang):
            ax.scatter(
                example.time_s,
                example.depth_m,
                s=60,
                color=example.color,
                alpha=0.85,
                edgecolors="none",
            )
            ax.annotate(
                example.label,
                (example.time_s, example.depth_m),
                xytext=(6, -6),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="top",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc=example.color,
                    ec=example.color,
                    lw=0.6,
                    alpha=0.9,
                ),
            )
    if not args.no_frontiers:
        fast_times, fast_depths = frontier_by_depth(rows, args.depth_bin, mode="fast")
        slow_times, slow_depths = frontier_by_depth(rows, args.depth_bin, mode="slow")
        ax.fill_betweenx(
            fast_depths,
            fast_times,
            slow_times,
            color="#cfe9cf",
            alpha=0.25,
            label=strings["legend_feasible"],
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        ax.fill_betweenx(
            fast_depths,
            [0.0 for _ in fast_times],
            fast_times,
            color="#f2b6b6",
            alpha=0.2,
            label=strings["legend_infeasible"],
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        max_time = max(times) if times else 0.0
        max_time = max(max_time, 250.0)
        ax.fill_betweenx(
            fast_depths,
            slow_times,
            [max_time for _ in slow_times],
            color="#f2b6b6",
            alpha=0.2,
            label="_nolegend_",
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        depth_tail_start = max(fast_depths) if fast_depths else max_depth
        ax.fill_betweenx(
            [depth_tail_start, max_depth + 2.0],
            [0.0, 0.0],
            [max_time, max_time],
            color="#f2b6b6",
            alpha=0.2,
            label="_nolegend_",
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        ax.plot(
            fast_times,
            fast_depths,
            linestyle=(0, (1, 2)),
            linewidth=1.6,
            color="#4a4a4a",
            label="_nolegend_",
        )
        ax.plot(
            slow_times,
            slow_depths,
            linestyle=(0, (1, 2)),
            linewidth=1.6,
            color="#4a4a4a",
            label="_nolegend_",
        )
    ax.set_title(title)
    ax.set_xlabel(strings["xlabel"])
    ax.set_ylabel(strings["ylabel"])
    ax.set_xlim(left=0, right=250.0)
    ax.set_ylim(-1.0, max_depth + 2.0)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if not args.no_frontiers:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            framealpha=1.0,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
