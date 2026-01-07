#!/usr/bin/env python3
"""Plot a single dive profile with labeled T and D definitions."""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


def load_profile(csv_path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    depths: List[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times.append(float(row["time"]))
                depths.append(float(row["depth"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path.name}") from exc
    return times, depths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a single dive profile with labeled dive time T and max depth D."
    )
    parser.add_argument(
        "--input",
        default="data/63475a65dda5494297ffc915edc937e4.csv",
        help="Dive CSV to plot (default: data/63475a65dda5494297ffc915edc937e4.csv)",
    )
    parser.add_argument(
        "--output",
        default="images/example-profile.pdf",
        help="Output image path (default: images/example-profile.pdf)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise SystemExit(f"Input file not found: {csv_path}")

    times, depths = load_profile(csv_path)
    if not times or not depths:
        raise SystemExit("No data points found in the input CSV.")

    time_zero = times[0]
    times = [t - time_zero for t in times]
    t_start = 0.0
    t_end = times[-1]
    max_depth = max(depths)
    max_idx = depths.index(max_depth)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, depths, color="#1f77b4", linewidth=1.4)
    ax.axhline(0.0, color="#4c4c4c", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(max_depth, color="#4c4c4c", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(t_start, color="#4c4c4c", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(t_end, color="#4c4c4c", linewidth=0.8, linestyle="--", alpha=0.6)

    t_label_y = max_depth * 0.6
    ax.annotate(
        "",
        xy=(t_end, t_label_y),
        xytext=(t_start, t_label_y),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="#222222", mutation_scale=16),
    )
    ax.text(
        0.5 * (t_start + t_end),
        t_label_y + 0.6,
        "Dive time T",
        ha="center",
        va="top",
        fontsize=9,
        color="#222222",
    )

    max_x = times[max_idx]
    ax.scatter([max_x], [max_depth], s=30, color="#111111", zorder=3)
    ax.annotate(
        "",
        xy=(max_x, max_depth),
        xytext=(max_x, 0.0),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="#222222", mutation_scale=16),
    )
    ax.text(
        max_x + 0.02 * t_end,
        15.0,
        "Max depth D",
        ha="left",
        va="center",
        fontsize=9,
        color="#222222",
    )

    ax.text(
        10.0,
        0.6,
        "z = 0 at surface, positive downward",
        fontsize=8.5,
        color="#444444",
        va="top",
        ha="left",
    )
    ax.text(
        10.0,
        max_depth,
        "z = D",
        fontsize=8.5,
        color="#444444",
        va="bottom",
        ha="left",
    )

    x_label_pad = 0.008 * t_end
    ax.text(
        t_start + x_label_pad,
        15.0,
        "t = 0",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    ax.text(
        t_end - x_label_pad,
        15.0,
        "t = T",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#333333",
    )

    ax.set_title("Example Dive Profile and Definitions", pad=10)
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Depth z(t) (m)")
    ax.set_ylim(-1.4, max_depth + 2.0)
    x_pad = 0.03 * t_end if t_end > 0 else 0.0
    ax.set_xlim(left=-x_pad, right=t_end + x_pad)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
