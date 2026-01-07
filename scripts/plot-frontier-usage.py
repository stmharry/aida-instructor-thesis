#!/usr/bin/env python3
"""Plot resource usage from frontier CSV outputs."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


def plot_usage(
    outpath: str,
    fast_df: pd.DataFrame,
    slow_df: pd.DataFrame,
    title: str,
    max_depth: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    style_handles = [
        plt.Line2D([0], [0], color="#111111", linestyle=(0, (6, 3)), label="Fast"),
        plt.Line2D([0], [0], color="#111111", linestyle=(0, (2, 2)), label="Slow"),
    ]

    f = fast_df.dropna(subset=["D", "F_frac", "P_frac"])
    s = slow_df.dropna(subset=["D", "F_frac", "P_frac"])
    axes[0].plot(
        f["F_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#1f77b4",
    )
    axes[0].plot(
        s["F_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#1f77b4",
    )
    axes[0].plot(
        f["P_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#ff7f0e",
    )
    axes[0].plot(
        s["P_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#ff7f0e",
    )
    axes[0].set_xlabel("Peak Force / Power Utilization (%)")
    axes[0].set_ylabel("Max Depth (m)")
    axes[0].set_xlim(left=-2.0, right=102.0)
    axes[0].grid(True, linewidth=0.5, alpha=0.5)
    usage_handles = [
        plt.Line2D([0], [0], color="#1f77b4", linestyle="-", label="Force"),
        plt.Line2D([0], [0], color="#ff7f0e", linestyle="-", label="Power"),
    ]
    legend_usage = axes[0].legend(
        handles=usage_handles, fontsize=8, title="Constraint Type"
    )
    axes[0].add_artist(legend_usage)

    f = fast_df.dropna(subset=["D", "O_base_frac", "O_act_frac", "O_mech_frac"])
    s = slow_df.dropna(subset=["D", "O_base_frac", "O_act_frac", "O_mech_frac"])
    axes[1].plot(
        f["O_base_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#2ca02c",
    )
    axes[1].plot(
        s["O_base_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#2ca02c",
    )
    axes[1].plot(
        f["O_act_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#d62728",
    )
    axes[1].plot(
        s["O_act_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#d62728",
    )
    axes[1].plot(
        f["O_mech_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#8c564b",
    )
    axes[1].plot(
        s["O_mech_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#8c564b",
    )
    axes[1].plot(
        f["O_used_frac"] * 100,
        f["D"],
        linestyle=(0, (6, 3)),
        color="#000000",
    )
    axes[1].plot(
        s["O_used_frac"] * 100,
        s["D"],
        linestyle=(0, (2, 2)),
        color="#000000",
    )
    axes[1].set_xlabel("$O_2$ Budget Share (%)")
    axes[1].set_xlim(left=-2.0, right=102.0)
    axes[1].grid(True, linewidth=0.5, alpha=0.5)

    axes[0].set_ylim(max_depth + 2.0, -1.0)
    axes[1].set_ylim(max_depth + 2.0, -1.0)
    o2_handles = [
        plt.Line2D([0], [0], color="#2ca02c", linestyle="-", label="Basal"),
        plt.Line2D([0], [0], color="#d62728", linestyle="-", label="Activation"),
        plt.Line2D([0], [0], color="#8c564b", linestyle="-", label="Mechanical"),
        plt.Line2D([0], [0], color="#000000", linestyle="-", label="Total"),
    ]
    legend_o2 = axes[1].legend(handles=o2_handles, fontsize=8, title="$O_2$ Share")
    axes[1].add_artist(legend_o2)
    fig.legend(
        handles=style_handles,
        title="Frontier",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    fig.savefig(outpath, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontiers-csv",
        type=str,
        default="data/frontiers.csv",
        help="Path to frontiers CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images/frontier-usage.pdf",
        help="Output image path.",
    )
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

    max_depth = 0.0
    fast_depths = fast_df.dropna(subset=["T", "D"])
    slow_depths = slow_df.dropna(subset=["T", "D"])
    if not fast_depths.empty:
        max_depth = max(max_depth, float(fast_depths["D"].max()))
    if not slow_depths.empty:
        max_depth = max(max_depth, float(slow_depths["D"].max()))

    plot_usage(
        args.output,
        fast_df,
        slow_df,
        title="Resource Usage (at Performance Frontiers)",
        max_depth=max_depth,
    )


if __name__ == "__main__":
    main()
