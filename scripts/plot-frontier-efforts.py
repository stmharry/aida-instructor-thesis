#!/usr/bin/env python3
"""Plot frontier contours for multiple effort levels."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


def load_dives(csv_path: str) -> Optional[pd.DataFrame]:
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "duration" in cols and "max_depth" in cols:
        return df.rename(
            columns={cols["duration"]: "duration", cols["max_depth"]: "max_depth"}
        )
    if "t" in cols and "d" in cols:
        return df.rename(columns={cols["t"]: "duration", cols["d"]: "max_depth"})
    raise ValueError("dives_csv must have columns (duration,max_depth) or (t,d).")


def interpolate_frontier(
    depths: np.ndarray, times: np.ndarray, depth_vals: np.ndarray, T_sta: float | None
) -> np.ndarray:
    interp = np.interp(depth_vals, depths, times, left=np.nan, right=times[-1])
    min_depth = float(depths.min())
    time_at_min = float(times[0])
    shallow_mask = depth_vals < min_depth
    if shallow_mask.any() and min_depth > 0.0:
        if T_sta is None:
            interp[shallow_mask] = time_at_min * (depth_vals[shallow_mask] / min_depth)
        else:
            interp[shallow_mask] = T_sta + (
                (time_at_min - T_sta) * (depth_vals[shallow_mask] / min_depth)
            )
    return interp


def plot_effort_contours(
    outpath: str,
    dives: Optional[pd.DataFrame],
    frontiers_df: pd.DataFrame,
    effort_levels: np.ndarray,
    title: str,
    T_sta: float,
) -> None:
    plt.figure(figsize=(10.0, 4.0))

    max_depth = 0.0
    max_time = 0.0
    if dives is not None:
        max_depth = max(max_depth, float(dives["max_depth"].max()))
        max_time = max(max_time, float(dives["duration"].max()))
        plt.scatter(
            dives["duration"],
            dives["max_depth"],
            s=18,
            alpha=0.55,
            color="#b8b8b8",
            edgecolors="none",
            zorder=3,
            label="Dive Records",
        )

    cmap = plt.get_cmap("RdYlGn_r")
    colors = cmap(np.linspace(0.1, 0.9, max(len(effort_levels), 2)))

    entries = []
    for color, effort in zip(colors, effort_levels):
        subset = frontiers_df[np.isclose(frontiers_df["effort"], effort)]
        if subset.empty:
            continue
        fast_df = subset[subset["frontier"] == "fast"].dropna(subset=["T", "D"])
        slow_df = subset[subset["frontier"] == "slow"].dropna(subset=["T", "D"])
        effort_T_sta = T_sta * effort

        if not fast_df.empty and not slow_df.empty:
            fast_depths = fast_df["D"].to_numpy()
            fast_values = fast_df["T"].to_numpy()
            slow_depths = slow_df["D"].to_numpy()
            slow_values = slow_df["T"].to_numpy()
            max_depth = max(max_depth, float(fast_depths.max()), float(slow_depths.max()))
            max_time = max(max_time, float(fast_values.max()), float(slow_values.max()))
            depth_vals = np.linspace(
                0.0,
                float(min(fast_depths.max(), slow_depths.max())),
                max(len(fast_values), len(slow_values), 2),
            )
            fast_times = interpolate_frontier(
                fast_depths, fast_values, depth_vals, T_sta=None
            )
            slow_times = interpolate_frontier(
                slow_depths, slow_values, depth_vals, T_sta=effort_T_sta
            )
            entries.append(
                {
                    "effort": effort,
                    "color": color,
                    "depth_vals": depth_vals,
                    "fast_times": fast_times,
                    "slow_times": slow_times,
                }
            )
        elif not fast_df.empty:
            fast_depths = fast_df["D"].to_numpy()
            fast_values = fast_df["T"].to_numpy()
            max_depth = max(max_depth, float(fast_depths.max()))
            max_time = max(max_time, float(fast_values.max()))
            depth_vals = np.linspace(
                0.0, float(fast_depths.max()), max(len(fast_values), 2)
            )
            fast_times = interpolate_frontier(
                fast_depths, fast_values, depth_vals, T_sta=None
            )
            entries.append(
                {
                    "effort": effort,
                    "color": color,
                    "depth_vals": depth_vals,
                    "fast_times": fast_times,
                    "slow_times": None,
                }
            )
        elif not slow_df.empty:
            slow_depths = slow_df["D"].to_numpy()
            slow_values = slow_df["T"].to_numpy()
            max_depth = max(max_depth, float(slow_depths.max()))
            max_time = max(max_time, float(slow_values.max()))
            depth_vals = np.linspace(
                0.0, float(slow_depths.max()), max(len(slow_values), 2)
            )
            slow_times = interpolate_frontier(
                slow_depths, slow_values, depth_vals, T_sta=effort_T_sta
            )
            entries.append(
                {
                    "effort": effort,
                    "color": color,
                    "depth_vals": depth_vals,
                    "fast_times": None,
                    "slow_times": slow_times,
                }
            )

    for entry in reversed(entries):
        if entry["fast_times"] is None or entry["slow_times"] is None:
            continue

        plt.fill_betweenx(
            entry["depth_vals"],
            entry["fast_times"],
            entry["slow_times"],
            color=entry["color"],
            alpha=0.85,
            linewidth=0.0,
            edgecolor="none",
            zorder=1,
        )

    handles = []
    labels = []
    sorted_entries = sorted(entries, key=lambda item: item["effort"])
    for idx, entry in enumerate(sorted_entries):
        start = sorted_entries[idx - 1]["effort"] if idx > 0 else 0.0
        end = entry["effort"]
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=entry["color"], edgecolor="none")
        )
        labels.append(f"{start * 100:.0f}–{end * 100:.0f}%")

    plt.title(title)
    plt.xlabel("Dive Time (s)")
    plt.ylabel("Max Depth (m)")
    plt.xlim(left=0, right=max(max_time, 250.0))
    plt.ylim(-1.0, max_depth + 2.0)

    plt.gca().invert_yaxis()
    plt.grid(True, linewidth=0.5, alpha=0.5)
    if handles:
        plt.legend(
            handles=handles,
            labels=labels,
            title="Effort Level",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontiers-csv",
        type=str,
        default="data/frontiers.csv",
        help="Path to frontiers CSV",
    )
    parser.add_argument(
        "--dives-csv",
        type=str,
        default="data/dives.csv",
        help="CSV with columns duration,max_depth (default: data/dives.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images/frontiers-efforts.pdf",
        help="Output image path.",
    )
    parser.add_argument(
        "--efforts",
        type=str,
        default="",
        help="Comma-separated effort levels (fractions). Default: all in CSV.",
    )
    parser.add_argument("--T-sta", type=float, default=240.0)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    frontiers_df = pd.read_csv(args.frontiers_csv)
    if "effort" not in frontiers_df.columns:
        raise ValueError("frontiers_csv must include an effort column.")

    if args.efforts.strip():
        effort_levels = np.array(
            [float(x) for x in args.efforts.split(",") if x.strip()],
            dtype=float,
        )
    else:
        effort_levels = np.sort(frontiers_df["effort"].dropna().unique())

    dives = load_dives(args.dives_csv) if args.dives_csv else None

    plot_effort_contours(
        args.output,
        dives,
        frontiers_df,
        effort_levels,
        title="Effort Level Bands for Dives",
        T_sta=args.T_sta,
    )


if __name__ == "__main__":
    main()
