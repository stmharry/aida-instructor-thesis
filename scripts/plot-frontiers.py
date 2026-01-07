#!/usr/bin/env python3
"""Plot fast/slow frontiers from CSV outputs."""

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


def plot_td(
    outpath: str,
    dives: Optional[pd.DataFrame],
    fast_df: pd.DataFrame,
    slow_df: pd.DataFrame,
    title: str,
    slow_label: str,
    T_sta: float,
) -> None:
    plt.figure(figsize=(10.0, 4.0))

    max_depth = 0.0
    if dives is not None:
        max_depth = max(max_depth, float(dives["max_depth"].max()))
        plt.scatter(
            dives["duration"],
            dives["max_depth"],
            s=18,
            alpha=0.55,
            color="#b8b8b8",
            edgecolors="none",
            label="Dive Records",
        )

    f = fast_df.dropna(subset=["T", "D"])
    s = slow_df.dropna(subset=["T", "D"])
    if not f.empty:
        max_depth = max(max_depth, float(f["D"].max()))
    if not s.empty:
        max_depth = max(max_depth, float(s["D"].max()))

    if not f.empty:
        fast_depths = f["D"].to_numpy()
        fast_values = f["T"].to_numpy()
        grid_size = max(len(f), len(s), 2)
        if not s.empty:
            slow_depths = s["D"].to_numpy()
            depth_limit = min(float(fast_depths.max()), float(slow_depths.max()))
            max_depth_grid = depth_limit
        else:
            max_depth_grid = float(fast_depths.max())
        depth_vals = np.linspace(0.0, max_depth_grid, grid_size)
        fast_times = np.interp(
            depth_vals,
            fast_depths,
            fast_values,
            left=np.nan,
            right=float(fast_values[-1]),
        )
        fast_min_depth = float(fast_depths.min())
        fast_time_at_min = float(fast_values[0])
        fast_mask = depth_vals < fast_min_depth
        if fast_mask.any() and fast_min_depth > 0.0:
            fast_times[fast_mask] = fast_time_at_min * (
                depth_vals[fast_mask] / fast_min_depth
            )

        slow_times = np.full_like(fast_times, np.nan)
        if not s.empty:
            slow_values = s["T"].to_numpy()
            slow_times = np.interp(
                depth_vals,
                slow_depths,
                slow_values,
                left=np.nan,
                right=slow_values[-1],
            )
            slow_min_depth = float(slow_depths.min())
            slow_time_at_min = float(slow_values[0])
            shallow_mask = depth_vals < slow_min_depth
            if shallow_mask.any() and slow_min_depth > 0.0:
                slow_times[shallow_mask] = T_sta + (
                    (slow_time_at_min - T_sta)
                    * (depth_vals[shallow_mask] / slow_min_depth)
                )

        if not s.empty:
            feasible_mask = np.isfinite(fast_times) & np.isfinite(slow_times)
            if feasible_mask.any():
                plt.fill_betweenx(
                    depth_vals[feasible_mask],
                    fast_times[feasible_mask],
                    slow_times[feasible_mask],
                    color="#cfe9cf",
                    alpha=0.25,
                    label="Feasible Region",
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=0,
                )
        max_time = (
            max(float(f["T"].max()), float(s["T"].max()))
            if not f.empty and not s.empty
            else 0.0
        )
        max_time = max(max_time, 250.0)
        plt.fill_betweenx(
            depth_vals,
            np.zeros_like(fast_times),
            fast_times,
            color="#f2b6b6",
            alpha=0.2,
            label="Infeasible Region",
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        plt.fill_betweenx(
            depth_vals,
            slow_times,
            np.full_like(slow_times, max_time),
            color="#f2b6b6",
            alpha=0.2,
            label="_nolegend_",
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        depth_tail_start = float(depth_vals.max())
        plt.fill_betweenx(
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

    plt.plot(
        f["T"],
        f["D"],
        linewidth=1.6,
        linestyle=(0, (6, 3)),
        color="#4a4a4a",
        label="Fast Frontier",
    )
    plt.plot(
        s["T"],
        s["D"],
        linewidth=1.6,
        linestyle=(0, (2, 2)),
        color="#4a4a4a",
        label=slow_label,
    )

    plt.title(title)
    plt.xlabel("Dive Time (s)")
    plt.ylabel("Max Depth (m)")
    plt.xlim(left=0, right=250.0)
    plt.ylim(-1.0, max_depth + 2.0)

    plt.gca().invert_yaxis()
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
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
        default="images/frontiers.pdf",
        help="Output image path.",
    )
    parser.add_argument("--T-sta", type=float, default=240.0)
    parser.add_argument(
        "--effort",
        type=float,
        default=1.0,
        help="Effort level (fraction of oxygen budget) to plot.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    frontiers_df = pd.read_csv(args.frontiers_csv)
    if "effort" in frontiers_df.columns:
        frontiers_df = frontiers_df[
            np.isclose(frontiers_df["effort"], args.effort)
        ]
    fast_df = frontiers_df[frontiers_df["frontier"] == "fast"]
    slow_df = frontiers_df[frontiers_df["frontier"] == "slow"]
    dives = load_dives(args.dives_csv) if args.dives_csv else None

    plot_td(
        args.output,
        dives,
        fast_df,
        slow_df,
        title="Dive Time vs. Max Depth (Performance Frontiers)",
        slow_label="Slow Frontier",
        T_sta=args.T_sta,
    )


if __name__ == "__main__":
    main()
