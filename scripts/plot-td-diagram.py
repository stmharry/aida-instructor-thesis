#!/usr/bin/env python3
"""Generate a time-depth (T-D) diagram from data/dives.csv."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


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


EXAMPLE_POINTS = [
    ExamplePoint(
        "Personal Best Dive (FIM)", time_s=108.0, depth_m=45.0, color="#ff7f0e"
    ),
    ExamplePoint("Slow Dive ~30 M (FIM)", time_s=144.0, depth_m=30.0, color="#2ca02c"),
    ExamplePoint("STA ~4 Min", time_s=244.0, depth_m=0.0, color="#1f4e79"),
    ExamplePoint(
        "Sprint Dive ~20 M (CWTB)", time_s=28.0, depth_m=20.0, color="#d62728"
    ),
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
        "--title",
        default="Dive Time vs. Max Depth",
        help="Plot title",
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_dives(input_path, args.time_col, args.depth_col)
    times = [t for t, _ in rows]
    depths = [d for _, d in rows]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    max_depth = max(depths) if depths else 0.0
    ax.scatter(
        times,
        depths,
        s=18,
        alpha=0.55,
        color="#b8b8b8",
        edgecolors="none",
        label="Dive Records",
    )
    if not args.no_examples:
        for example in EXAMPLE_POINTS:
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
            label="Feasible Region",
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
            label="Infeasible Region",
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
    ax.set_title(args.title)
    ax.set_xlabel("Dive Time (s)")
    ax.set_ylabel("Max Depth (m)")
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
