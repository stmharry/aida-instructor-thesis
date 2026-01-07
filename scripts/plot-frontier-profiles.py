#!/usr/bin/env python3
"""Plot optimal dive profiles (force, depth, speed) for selected depths."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


def load_dives(csv_path: str) -> float:
    if not csv_path:
        return float("nan")
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "max_depth" in cols:
        return float(df[cols["max_depth"]].max())
    if "d" in cols:
        return float(df[cols["d"]].max())
    return float("nan")


def max_frontier_depth(profiles_df: pd.DataFrame) -> float:
    depths = profiles_df["D"].dropna().unique()
    if depths.size == 0:
        return float("nan")
    return float(np.nanmax(depths))


def nearest_depth(available: np.ndarray, target: float) -> float:
    if available.size == 0:
        return float("nan")
    return float(available[np.argmin(np.abs(available - target))])


def build_time_series(profile: pd.DataFrame) -> Dict[str, np.ndarray]:
    z = profile["z"].to_numpy()
    u_d = profile["u_d"].to_numpy()
    u_u = profile["u_u"].to_numpy()
    v_d = profile["v_d"].to_numpy()
    v_u = profile["v_u"].to_numpy()

    if z.size < 2:
        return {}

    dz_desc = np.diff(z)
    t_desc = np.concatenate([[0.0], np.cumsum(dz_desc / v_d[:-1])])
    t_desc_end = float(t_desc[-1])

    z_rev = z[::-1]
    u_u_rev = u_u[::-1]
    v_u_rev = v_u[::-1]
    dz_asc = np.abs(np.diff(z_rev))
    t_asc = t_desc_end + np.cumsum(dz_asc / v_u_rev[:-1])

    t_full = np.concatenate([t_desc, t_asc])
    z_full = np.concatenate([z, z_rev[1:]])
    force_full = np.concatenate([u_d, -u_u_rev[1:]])
    speed_full = np.concatenate([v_d, -v_u_rev[1:]])
    u_mag = np.abs(force_full)
    v_mag = np.abs(speed_full)

    return {
        "t": t_full,
        "z": z_full,
        "force": force_full,
        "speed": speed_full,
        "u_mag": u_mag,
        "v_mag": v_mag,
    }


def compute_o2_fraction(
    series: Dict[str, np.ndarray],
    effort: float,
    T_sta: float,
    V_vc: float,
    z_failure: float,
    o2_air_percentage: float,
    o2_multiplier: float,
    rho: float,
    g: float,
    P_atm: float,
    alpha: float,
    F_ref: float,
    p: float,
    beta: float,
) -> np.ndarray:
    t = series["t"]
    u_mag = series["u_mag"]
    v_mag = series["v_mag"]
    if t.size < 2:
        return np.array([])

    L_atm = P_atm / (rho * g)
    V_tlc = V_vc * (1 + L_atm / z_failure)
    V_o2 = V_tlc * o2_air_percentage * o2_multiplier
    dotV_o2 = V_o2 / T_sta
    budget = V_o2 * effort
    if budget <= 0.0:
        return np.full_like(t, np.nan)

    o2_rate = dotV_o2 + alpha * (u_mag / F_ref) ** p + beta * u_mag * v_mag
    dt = np.diff(t)
    o2_cum = np.concatenate([[0.0], np.cumsum(0.5 * (o2_rate[1:] + o2_rate[:-1]) * dt)])
    return o2_cum / budget


def plot_profiles(
    outpath: str,
    profiles_df: pd.DataFrame,
    depths: List[float],
    title: str,
    frontier: str,
    effort: float,
    T_sta: float,
    V_vc: float,
    z_failure: float,
    o2_air_percentage: float,
    o2_multiplier: float,
    rho: float,
    g: float,
    P_atm: float,
    alpha: float,
    F_ref: float,
    p: float,
    beta: float,
) -> None:
    rows = len(depths)
    fig, axes = plt.subplots(rows, 1, figsize=(10.0, 2.75 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    metric_colors = {"depth": "#1f77b4", "speed": "#ff7f0e", "force": "#2ca02c"}
    line_style = (0, (6, 3)) if frontier == "fast" else (0, (2, 2))

    max_time = 0.0

    for ax, target_depth in zip(axes, depths):
        ax.set_title(f"Depth Target: {target_depth:.1f} m")
        ax_speed = ax.twinx()
        ax_force = ax.twinx()
        ax_o2 = ax.twinx()
        ax_force.spines["right"].set_position(("axes", 1.10))
        ax_force.set_frame_on(True)
        ax_force.patch.set_visible(False)
        ax_o2.spines["right"].set_position(("axes", 1.20))
        ax_o2.set_frame_on(True)
        ax_o2.patch.set_visible(False)
        ax_speed.set_label("speed")
        ax_force.set_label("force")
        ax_o2.set_label("o2")
        subset = profiles_df[profiles_df["frontier"] == frontier]
        available = subset["D"].dropna().unique()
        chosen = nearest_depth(available, target_depth)
        if not np.isfinite(chosen):
            continue
        prof = subset[subset["D"] == chosen].sort_values("z")
        series = build_time_series(prof)
        if not series:
            continue

        ax.plot(
            series["t"],
            series["z"],
            color=metric_colors["depth"],
            linestyle=line_style,
        )

        ax_speed.plot(
            series["t"],
            series["speed"],
            color=metric_colors["speed"],
            linestyle=line_style,
        )

        ax_force.plot(
            series["t"],
            series["force"],
            color=metric_colors["force"],
            linestyle=line_style,
        )
        o2_frac = compute_o2_fraction(
            series,
            effort=effort,
            T_sta=T_sta,
            V_vc=V_vc,
            z_failure=z_failure,
            o2_air_percentage=o2_air_percentage,
            o2_multiplier=o2_multiplier,
            rho=rho,
            g=g,
            P_atm=P_atm,
            alpha=alpha,
            F_ref=F_ref,
            p=p,
            beta=beta,
        )
        if o2_frac.size:
            ax_o2.plot(
                series["t"],
                o2_frac * 100.0,
                color="#8c564b",
                linestyle=line_style,
            )

        max_time = max(max_time, float(np.nanmax(series["t"])))

        ax.set_ylabel("Depth (m)")
        ax_speed.set_ylabel("Speed (m/s)")
        ax_force.set_ylabel("Force (N)")
        ax_o2.set_ylabel("$O_2$ Used (%)")
        ax.invert_yaxis()
        ax.grid(True, linewidth=0.5, alpha=0.5)

        max_speed = float(np.nanmax(np.abs(series["speed"])))
        max_force = float(np.nanmax(np.abs(series["force"])))
        if np.isfinite(max_speed) and max_speed > 0:
            ax_speed.set_ylim(-max_speed, max_speed)
        if np.isfinite(max_force) and max_force > 0:
            ax_force.set_ylim(-max_force, max_force)
        ax_o2.set_ylim(0.0, 105.0)

        handles = [
            plt.Line2D(
                [0], [0], color=metric_colors["depth"], linestyle="-", label="Depth"
            ),
            plt.Line2D(
                [0], [0], color=metric_colors["speed"], linestyle="-", label="Speed"
            ),
            plt.Line2D(
                [0], [0], color=metric_colors["force"], linestyle="-", label="Force"
            ),
            plt.Line2D([0], [0], color="#8c564b", linestyle="-", label="$O_2$ Used"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    if max_time > 0:
        for ax in axes:
            ax.set_xlim(0.0, max_time)

    axes[-1].set_xlabel("Elapsed Dive Time (s)")

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outpath, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles-csv",
        type=str,
        default="data/frontier-profiles.csv",
        help="Path to frontier profiles CSV",
    )
    parser.add_argument(
        "--dives-csv",
        type=str,
        default="data/dives.csv",
        help="CSV with max depth for PB depth selection",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="images/frontier-profiles-{frontier}.pdf",
        help="Output path template (use {frontier}).",
    )
    parser.add_argument("--pb-depth", type=float, default=np.nan)
    parser.add_argument(
        "--effort",
        type=float,
        default=1.0,
        help="Effort level (fraction of oxygen budget) to plot.",
    )
    parser.add_argument("--T-sta", type=float, default=240.0)
    parser.add_argument("--V-vc", type=float, default=5.0)
    parser.add_argument("--z-failure", type=float, default=30.0)
    parser.add_argument("--o2-air", type=float, default=0.2095)
    parser.add_argument("--o2-multiplier", type=float, default=1250.0)
    parser.add_argument("--rho", type=float, default=1025.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--P-atm", type=float, default=101325.0)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--F-ref", type=float, default=100.0)
    parser.add_argument("--p", type=float, default=1.6)
    parser.add_argument("--beta", type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_template) or ".", exist_ok=True)

    profiles_df = pd.read_csv(args.profiles_csv)
    if "effort" in profiles_df.columns:
        profiles_df = profiles_df[np.isclose(profiles_df["effort"], args.effort)]
    pb_depth = args.pb_depth
    if not np.isfinite(pb_depth):
        pb_depth = max_frontier_depth(profiles_df)
    if not np.isfinite(pb_depth):
        pb_depth = load_dives(args.dives_csv)
    if not np.isfinite(pb_depth):
        pb_depth = 45.0

    depths = [10.0, 20.0, 30.0, 40.0, float(pb_depth)]
    unique_depths = list(dict.fromkeys(depths))
    for frontier in ["fast", "slow"]:
        outpath = args.output_template.format(frontier=frontier)
        plot_profiles(
            outpath,
            profiles_df,
            unique_depths,
            title=f"Optimal Dive Profiles at Performance Frontiers ({frontier.title()})",
            frontier=frontier,
            effort=args.effort,
            T_sta=args.T_sta,
            V_vc=args.V_vc,
            z_failure=args.z_failure,
            o2_air_percentage=args.o2_air,
            o2_multiplier=args.o2_multiplier,
            rho=args.rho,
            g=args.g,
            P_atm=args.P_atm,
            alpha=args.alpha,
            F_ref=args.F_ref,
            p=args.p,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
