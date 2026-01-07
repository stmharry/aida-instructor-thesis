#!/usr/bin/env python3
"""Plot depth vs time profiles for extracted dives in data/."""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

PROFILE_LABELS = {
    "63475a65dda5494297ffc915edc937e4": "Personal Best Dive (FIM)",
    "74918b030ad842a5a66103d4aa1c56d3": "Slow Dive ~30 M (FIM)",
    "fabd899bd41a4c8c97849fba45034bf9": "Sprint Dive ~20 M (CWTB)",
}

PROFILE_COLORS = {
    "63475a65dda5494297ffc915edc937e4": "#ff7f0e",
    "74918b030ad842a5a66103d4aa1c56d3": "#2ca02c",
    "fabd899bd41a4c8c97849fba45034bf9": "#d62728",
}


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
        description="Plot dive profiles (depth vs time) from CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with per-dive CSVs (default: data)",
    )
    parser.add_argument(
        "--output",
        default="images/dive-profiles.pdf",
        help="Output image path (default: images/dive-profiles.pdf)",
    )
    parser.add_argument(
        "--exclude",
        default="dives.csv",
        help="Comma-separated filenames to exclude (default: dives.csv)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exclude = {name.strip() for name in args.exclude.split(",") if name.strip()}
    csv_paths = sorted(p for p in input_dir.glob("*.csv") if p.name not in exclude)
    if not csv_paths:
        raise SystemExit("No dive profile CSV files found.")

    profiles = [(p, *load_profile(p)) for p in csv_paths]
    max_depth = 0.0
    for _path, _times, _depths in profiles:
        if _depths:
            max_depth = max(max_depth, max(_depths))

    fig, ax = plt.subplots(figsize=(10, 4.0))
    for path, times, depths in profiles:
        color = PROFILE_COLORS.get(path.stem, "#1f77b4")
        ax.plot(times, depths, linewidth=1.0, color=color)
        label = PROFILE_LABELS.get(path.stem, f"Dive Profile: {path.stem}")
        anchor_idx = int(0.75 * (len(times) - 1))
        ax.annotate(
            label,
            (times[anchor_idx], depths[anchor_idx]),
            xytext=(20, -20),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc=color,
                ec=color,
                lw=0.6,
                alpha=0.9,
            ),
            color="white",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=color),
        )
    sta_color = "#1f4e79"
    sta_time = [float(t) for t in range(0, 245, 27)]
    sta_depth = [0.0] * len(sta_time)
    ax.plot(sta_time, sta_depth, linewidth=1.0, color=sta_color, linestyle="-")
    sta_anchor_idx = int(0.75 * (len(sta_time) - 1))
    ax.annotate(
        "STA ~4 Min",
        (sta_time[sta_anchor_idx], sta_depth[sta_anchor_idx]),
        xytext=(20, -20),
        textcoords="offset points",
        fontsize=8.5,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc=sta_color,
            ec=sta_color,
            lw=0.6,
            alpha=0.9,
        ),
        color="white",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=sta_color),
    )
    ax.set_title("Dive Profiles")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Depth (m)")
    if max_depth > 0.0:
        ax.set_ylim(-1.0, max_depth + 2.0)
    ax.set_xlim(left=0, right=250.0)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
