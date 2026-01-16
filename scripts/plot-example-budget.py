#!/usr/bin/env python3
"""Redraw the integrated resource bookkeeping figure programmatically."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Arial"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot O2/CO2 budget schematic.")
    parser.add_argument(
        "--output",
        default="images/example-budget.pdf",
        help="Output image path (default: images/example-budget.pdf)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.5, 6.0), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.5, wspace=0.3)

    t = np.linspace(0, 1, 200)
    o2 = t**2.5
    co2 = t**0.9

    # Top-left: O2 budget
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, o2, color="#2b6da3", linewidth=3)
    ax1.set_title("Oxygen ($O_2$) Budget")
    ax1.set_xlabel("Elapsed Time ($t$)")
    ax1.set_ylabel("Used $O_2$")
    ax1.set_xlim(0, 1.35)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks([0.0, 1.0])
    ax1.set_xticklabels(["$t=0$", "$t=T_{O_2}^{\\mathrm{total}}$"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["0", "$R_{O_2}^{\\mathrm{total}}$"])
    ax1.axhline(1.0, color="#2b6da3", linestyle="--", linewidth=1)
    ax1.axvline(1.0, color="#9a9a9a", linestyle=":", linewidth=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.annotate(
        "Total Usable $O_2$ Store",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        color="#2b6da3",
        ha="left",
        va="top",
    )
    ax1.annotate(
        "$R_{O_2}(t)$",
        xy=(0.75, float(0.75**2.5)),
        color="#2b6da3",
        fontsize=11,
        ha="right",
        va="bottom",
    )
    ax1.annotate(
        "$O_2$ Limit\nReached\n($E_{O_2}=1$)",
        xy=(1.0, 1.0),
        xycoords="data",
        xytext=(0.80, 0.65),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="left",
        va="top",
    )
    ax1.text(
        0.02,
        0.05,
        "Basal rate + Work-dependent rate\n(reduced by diving response)",
        transform=ax1.transAxes,
        fontsize=10,
    )

    # Top-right: CO2 budget
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, co2, color="#b0372f", linewidth=3)
    ax2.set_title("Carbon Dioxide ($CO_2$) Budget")
    ax2.set_xlabel("Elapsed Time ($t$)")
    ax2.set_ylabel("Accumulated $CO_2$")
    ax2.set_xlim(0, 1.35)
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks([0.0, 1.0])
    ax2.set_xticklabels(["$t=0$", "$t=T_{CO_2}^{\\mathrm{total}}$"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["0", "$R_{CO_2}^{\\mathrm{total}}$"])
    ax2.axhline(1.0, color="#b0372f", linestyle="--", linewidth=1)
    ax2.axvline(1.0, color="#9a9a9a", linestyle=":", linewidth=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.annotate(
        "Effective $CO_2$ Tolerance Threshold",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        color="#b0372f",
        ha="left",
        va="top",
    )
    ax2.annotate(
        "$R_{CO_2}(t)$",
        xy=(0.40, 0.60),
        color="#b0372f",
        fontsize=11,
        ha="center",
        va="bottom",
    )
    ax2.annotate(
        "$CO_2$ Limit\nReached\n($E_{CO_2}=1$)",
        xy=(1.0, 1.0),
        xycoords="data",
        xytext=(0.80, 0.65),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        ha="left",
        va="top",
    )
    ax2.text(
        0.02,
        0.05,
        "Metabolic production\n(buffered, proxy for urge-to-breathe)",
        transform=ax2.transAxes,
        fontsize=10,
    )

    # Bottom: dual constraint trajectory
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("Dual Constraint Dive Profiles & Active Limiter")
    ax3.set_xlabel("Accumulated $CO_2$")
    ax3.set_ylabel("Used $O_2$")
    ax3.set_xlim(0, 1.35)
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["0", "$R_{CO_2}^{\\mathrm{total}}$"])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["0", "$R_{O_2}^{\\mathrm{total}}$"])
    ax3.axhline(1.0, color="#9a9a9a", linestyle=":", linewidth=1)
    ax3.axvline(1.0, color="#9a9a9a", linestyle=":", linewidth=1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.plot([0, 1.0], [0, 0.6], color="#6b6b6b", linewidth=2.5)
    ax3.plot([0, 0.6], [0, 1.0], color="#6b6b6b", linewidth=2.5)
    ax3.text(
        0.62,
        0.38,
        "Dive Profile A",
        color="#6b6b6b",
        fontsize=11,
        ha="right",
        va="bottom",
    )
    ax3.text(
        0.38,
        0.64,
        "Dive Profile B",
        color="#6b6b6b",
        fontsize=11,
        ha="right",
        va="bottom",
    )

    ax3.scatter([1.0], [0.6], marker="x", s=140, color="#b0372f", linewidths=3)
    ax3.scatter([0.6], [1.0], marker="x", s=140, color="#2b6da3", linewidths=3)
    ax3.axvline(1.0, color="#b0372f", linestyle="--", linewidth=1)
    ax3.text(
        1.02,
        0.02,
        "Effective $CO_2$ Tolerance Threshold",
        color="#b0372f",
        rotation=90,
        va="bottom",
    )
    ax3.axhline(1.0, color="#2b6da3", linestyle="--", linewidth=1)
    ax3.text(
        0.02,
        1.02,
        "Total Usable $O_2$ Store",
        color="#2b6da3",
        va="bottom",
    )

    ax3.annotate(
        (
            "$CO_2$-Limited Termination\n"
            "Less-adapted diver example.\n"
            "Other resource slack ($E_{O_2}<1$)."
        ),
        xy=(1.0, 0.6),
        xytext=(0.80, 0.15),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=10,
        ha="left",
    )
    ax3.annotate(
        (
            "$O_2$-Limited Termination\n"
            "Elite specialist example.\n"
            "Other resource slack ($E_{CO_2}<1$)."
        ),
        xy=(0.6, 1.0),
        xytext=(0.45, 0.60),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=10,
        ha="left",
    )
    ax3.annotate(
        "Start of All Dives\n($t=0$)",
        xy=(0.0, 0.0),
        xytext=(0.08, 0.30),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=11,
        ha="center",
        va="bottom",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
