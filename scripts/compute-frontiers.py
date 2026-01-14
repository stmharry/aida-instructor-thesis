#!/usr/bin/env python3
"""Compute fast/slow frontiers and write CSV outputs."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit


@dataclass
class Params:
    P_atm: float = 101325  # atmospheric pressure (kg/m/s^2)
    rho: float = 1025.0  # seawater density (kg/m^3)
    g: float = 9.81  # gravity (m/s^2)
    eps: float = 1e-9
    o2_air_percentage: float = 0.2095  # fraction of O2 in air

    z_neutral: float = 12.0  # neutral depth (m)
    z_failure: float = 30.0  # mechanical failure depth (m)
    v_infinity: float = 0.8  # asymptotic terminal speed scale (m)
    T_sta: float = 240.0  # STA time
    V_vc: float = 5.0  # L
    F_max: float = 200.0  # force cap (N)
    P_max: float = 200.0  # power cap (W)
    o2_multiplier: float = 1250

    # mL/s
    alpha: float = 15.0  # activation overhead scale
    F_ref: float = 100.0  # reference force for activation term
    p: float = 1.6  # activation exponent
    beta: float = 0.20  # power-to-oxygen conversion coefficient

    @property
    def L_atm(self) -> float:
        return self.P_atm / (self.rho * self.g)

    @property
    def V_tlc(self) -> float:
        return self.V_vc * (1 + self.L_atm / self.z_failure)

    @property
    def F_infinity(self) -> float:
        return (
            self.rho * self.g * (self.V_tlc / 1000) / (1 + self.z_neutral / self.L_atm)
        )

    @property
    def k(self) -> float:
        return self.F_infinity / self.v_infinity**2

    @property
    def V_o2(self) -> float:
        return self.V_tlc * self.o2_air_percentage * self.o2_multiplier

    @property
    def dotV_o2(self) -> float:
        return self.V_o2 / self.T_sta


@njit(cache=True)
def _optimal_profile_numba(
    z: np.ndarray,
    u_grid: np.ndarray,
    lam: float,
    time_sign: float,
    Lp: float,
    zn: float,
    voo: float,
    k: float,
    Fmax: float,
    Pmax: float,
    V0: float,
    alpha: float,
    p: float,
    Fref: float,
    beta: float,
    eps: float,
    refine_span: float,
    refine_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Nz = z.shape[0]
    Mu = u_grid.shape[0]
    u_d = np.empty(Nz)
    u_u = np.empty(Nz)
    v_d = np.empty(Nz)
    v_u = np.empty(Nz)
    voo2 = voo * voo
    for i in range(Nz):
        zi = z[i]
        s_z = voo2 * (zi - zn) / (Lp + zi)
        best_phi_d = 1e30
        best_phi_u = 1e30
        best_ud = np.nan
        best_uu = np.nan
        best_vd = np.nan
        best_vu = np.nan
        for j in range(Mu):
            u = u_grid[j]
            if u < 0.0 or u > Fmax:
                continue
            vd = u / k + s_z
            if vd > eps:
                vd = math.sqrt(vd)
                if u * vd <= Pmax + 1e-12:
                    vterm = V0 + alpha * math.pow(u / Fref, p)
                    phi = (time_sign + lam * vterm) / vd + lam * beta * u
                    if phi < best_phi_d:
                        best_phi_d = phi
                        best_ud = u
                        best_vd = vd
            vu = u / k - s_z
            if vu > eps:
                vu = math.sqrt(vu)
                if u * vu <= Pmax + 1e-12:
                    vterm = V0 + alpha * math.pow(u / Fref, p)
                    phi = (time_sign + lam * vterm) / vu + lam * beta * u
                    if phi < best_phi_u:
                        best_phi_u = phi
                        best_uu = u
                        best_vu = vu
        # local refinement around the best coarse u
        if not math.isnan(best_ud) and refine_steps > 1:
            u_min = best_ud - refine_span
            if u_min < 0.0:
                u_min = 0.0
            u_max = best_ud + refine_span
            if u_max > Fmax:
                u_max = Fmax
            step = (u_max - u_min) / (refine_steps - 1)
            for r in range(refine_steps):
                u = u_min + step * r
                vd = u / k + s_z
                if vd > eps:
                    vd = math.sqrt(vd)
                    if u * vd <= Pmax + 1e-12:
                        vterm = V0 + alpha * math.pow(u / Fref, p)
                        phi = (time_sign + lam * vterm) / vd + lam * beta * u
                        if phi < best_phi_d:
                            best_phi_d = phi
                            best_ud = u
                            best_vd = vd
        if not math.isnan(best_uu) and refine_steps > 1:
            u_min = best_uu - refine_span
            if u_min < 0.0:
                u_min = 0.0
            u_max = best_uu + refine_span
            if u_max > Fmax:
                u_max = Fmax
            step = (u_max - u_min) / (refine_steps - 1)
            for r in range(refine_steps):
                u = u_min + step * r
                vu = u / k - s_z
                if vu > eps:
                    vu = math.sqrt(vu)
                    if u * vu <= Pmax + 1e-12:
                        vterm = V0 + alpha * math.pow(u / Fref, p)
                        phi = (time_sign + lam * vterm) / vu + lam * beta * u
                        if phi < best_phi_u:
                            best_phi_u = phi
                            best_uu = u
                            best_vu = vu
        u_d[i] = best_ud
        u_u[i] = best_uu
        v_d[i] = best_vd
        v_u[i] = best_vu
    return u_d, u_u, v_d, v_u


@njit(cache=True)
def _optimal_profile_min_speed_numba(
    z: np.ndarray,
    u_grid: np.ndarray,
    Lp: float,
    zn: float,
    voo: float,
    k: float,
    Fmax: float,
    Pmax: float,
    eps: float,
    refine_span: float,
    refine_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Nz = z.shape[0]
    Mu = u_grid.shape[0]
    u_d = np.empty(Nz)
    u_u = np.empty(Nz)
    v_d = np.empty(Nz)
    v_u = np.empty(Nz)
    voo2 = voo * voo
    for i in range(Nz):
        zi = z[i]
        s_z = voo2 * (zi - zn) / (Lp + zi)
        best_vd = 1e30
        best_vu = 1e30
        best_ud = np.nan
        best_uu = np.nan
        for j in range(Mu):
            u = u_grid[j]
            if u < 0.0 or u > Fmax:
                continue
            vd = u / k + s_z
            if vd > eps:
                vd = math.sqrt(vd)
                if u * vd <= Pmax + 1e-12 and vd < best_vd:
                    best_vd = vd
                    best_ud = u
            vu = u / k - s_z
            if vu > eps:
                vu = math.sqrt(vu)
                if u * vu <= Pmax + 1e-12 and vu < best_vu:
                    best_vu = vu
                    best_uu = u
        if not math.isnan(best_ud) and refine_steps > 1:
            u_min = best_ud - refine_span
            if u_min < 0.0:
                u_min = 0.0
            u_max = best_ud + refine_span
            if u_max > Fmax:
                u_max = Fmax
            step = (u_max - u_min) / (refine_steps - 1)
            for r in range(refine_steps):
                u = u_min + step * r
                vd = u / k + s_z
                if vd > eps:
                    vd = math.sqrt(vd)
                    if u * vd <= Pmax + 1e-12 and vd < best_vd:
                        best_vd = vd
                        best_ud = u
        if not math.isnan(best_uu) and refine_steps > 1:
            u_min = best_uu - refine_span
            if u_min < 0.0:
                u_min = 0.0
            u_max = best_uu + refine_span
            if u_max > Fmax:
                u_max = Fmax
            step = (u_max - u_min) / (refine_steps - 1)
            for r in range(refine_steps):
                u = u_min + step * r
                vu = u / k - s_z
                if vu > eps:
                    vu = math.sqrt(vu)
                    if u * vu <= Pmax + 1e-12 and vu < best_vu:
                        best_vu = vu
                        best_uu = u
        u_d[i] = best_ud
        u_u[i] = best_uu
        v_d[i] = best_vd
        v_u[i] = best_vu
    return u_d, u_u, v_d, v_u


def optimal_u_profile_for_lambda(
    lam: float,
    params: Params,
    z: np.ndarray,
    u_grid: np.ndarray,
    time_sign: float,
) -> Dict[str, np.ndarray]:
    """Depth-local argmin of phi(z,u;lambda) using numba acceleration."""
    u_d, u_u, v_d, v_u = _optimal_profile_numba(
        z,
        u_grid,
        lam,
        time_sign,
        params.L_atm,
        params.z_neutral,
        params.v_infinity,
        params.k,
        params.F_max,
        params.P_max,
        params.dotV_o2,
        params.alpha,
        params.p,
        params.F_ref,
        params.beta,
        params.eps,
        params.F_max * 0.08,
        21,
    )
    return {"z": z, "u_d": u_d, "u_u": u_u, "v_d": v_d, "v_u": v_u}


def optimal_u_profile_min_speed(
    params: Params,
    z: np.ndarray,
    u_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Max-time profile under feasibility/caps by minimizing speed pointwise."""
    u_d, u_u, v_d, v_u = _optimal_profile_min_speed_numba(
        z,
        u_grid,
        params.L_atm,
        params.z_neutral,
        params.v_infinity,
        params.k,
        params.F_max,
        params.P_max,
        params.eps,
        params.F_max * 0.08,
        21,
    )
    return {"z": z, "u_d": u_d, "u_u": u_u, "v_d": v_d, "v_u": v_u}


def integrate_T_O(
    profile: Dict[str, np.ndarray], params: Params
) -> Tuple[float, float]:
    """Compute total time and oxygen usage from depth profiles."""
    z = profile["z"]
    u_d = profile["u_d"]
    u_u = profile["u_u"]
    vd = profile["v_d"]
    vu = profile["v_u"]

    if np.any(~np.isfinite(u_d)) or np.any(~np.isfinite(u_u)):
        return (np.nan, np.nan)

    Td = np.trapezoid(1.0 / vd, z)  # T_d = ∫ dz / v_d
    Tu = np.trapezoid(1.0 / vu, z)  # T_u = ∫ dz / v_u
    T = Td + Tu

    Vd = params.dotV_o2 + params.alpha * (u_d / params.F_ref) ** params.p
    Vu = params.dotV_o2 + params.alpha * (u_u / params.F_ref) ** params.p

    Od = np.trapezoid(Vd / vd, z) + params.beta * np.trapezoid(u_d, z)
    Ou = np.trapezoid(Vu / vu, z) + params.beta * np.trapezoid(u_u, z)
    O = Od + Ou

    return (float(T), float(O))


def compute_usage(
    profile: Dict[str, np.ndarray], params: Params, effort: float
) -> Dict[str, float]:
    """Compute resource usage metrics from a depth profile."""
    z = profile["z"]
    u_d = profile["u_d"]
    u_u = profile["u_u"]
    vd = profile["v_d"]
    vu = profile["v_u"]

    if np.any(~np.isfinite(u_d)) or np.any(~np.isfinite(u_u)):
        return {
            "F_frac": np.nan,
            "P_frac": np.nan,
            "O_base_frac": np.nan,
            "O_act_frac": np.nan,
            "O_mech_frac": np.nan,
            "O_used_frac": np.nan,
        }

    max_force = float(np.nanmax([np.nanmax(u_d), np.nanmax(u_u)]))
    max_power = float(np.nanmax([np.nanmax(u_d * vd), np.nanmax(u_u * vu)]))

    O_base = np.trapezoid(params.dotV_o2 / vd, z) + np.trapezoid(params.dotV_o2 / vu, z)
    O_act = np.trapezoid(
        params.alpha * (u_d / params.F_ref) ** params.p / vd, z
    ) + np.trapezoid(params.alpha * (u_u / params.F_ref) ** params.p / vu, z)
    O_mech = params.beta * (np.trapezoid(u_d, z) + np.trapezoid(u_u, z))
    O_total = O_base + O_act + O_mech

    budget = params.V_o2 * effort
    denom = budget if budget > 0.0 else np.nan
    return {
        "F_frac": max_force / params.F_max if params.F_max > 0.0 else np.nan,
        "P_frac": max_power / params.P_max if params.P_max > 0.0 else np.nan,
        "O_base_frac": O_base / denom,
        "O_act_frac": O_act / denom,
        "O_mech_frac": O_mech / denom,
        "O_used_frac": O_total / denom,
    }


def solve_lambda_for_depth(
    D: float,
    params: Params,
    Nz: int,
    Mu: int,
    time_sign: float,
    lam_seed: Optional[float],
    effort: float,
    max_expand: int = 50,
    bisect_iter: int = 60,
) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
    """Bisection on lambda to satisfy oxygen budget O<=B with depth-local optimal u(z)."""
    z = np.linspace(0.0, D, Nz)
    u_grid = np.linspace(0.0, params.F_max, max(20, Mu // 4))
    budget = params.V_o2 * effort

    prof0 = optimal_u_profile_for_lambda(0.0, params, z, u_grid, time_sign)
    T0, O0 = integrate_T_O(prof0, params)
    if not np.isfinite(T0):
        return (np.nan, np.nan, np.nan, {})

    if O0 <= budget:
        return (0.0, T0, O0, prof0)

    lam_lo = 0.0
    lam_hi = 1e-4 if lam_seed is None else max(lam_seed, 1e-6)
    prof_hi = None
    T_hi = None
    O_hi = None

    for _ in range(max_expand):
        prof = optimal_u_profile_for_lambda(lam_hi, params, z, u_grid, time_sign)
        Tt, Oo = integrate_T_O(prof, params)
        if np.isfinite(Oo) and Oo <= budget:
            prof_hi, T_hi, O_hi = prof, Tt, Oo
            break
        lam_hi *= 2.0

    if prof_hi is None:
        return (np.nan, np.nan, np.nan, {})

    prof_mid_best = prof_hi
    T_mid_best = T_hi
    O_mid_best = O_hi
    tol = 1e-3
    for _ in range(bisect_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        prof_mid = optimal_u_profile_for_lambda(lam_mid, params, z, u_grid, time_sign)
        Tm, Om = integrate_T_O(prof_mid, params)

        if not np.isfinite(Om):
            lam_lo = lam_mid
            continue

        if Om > budget:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
            prof_mid_best = prof_mid
            T_mid_best = Tm
            O_mid_best = Om
            if abs(O_mid_best - budget) / budget < tol:
                break

    lam_star = lam_hi
    return (float(lam_star), float(T_mid_best), float(O_mid_best), prof_mid_best)


def profile_rows_from_result(
    frontier: str, effort: float, D: float, profile: Dict[str, np.ndarray]
) -> list:
    rows = []
    z_vals = profile["z"]
    u_d = profile["u_d"]
    u_u = profile["u_u"]
    v_d = profile["v_d"]
    v_u = profile["v_u"]
    for idx, z in enumerate(z_vals):
        rows.append(
            {
                "frontier": frontier,
                "effort": effort,
                "D": D,
                "z": float(z),
                "u_d": float(u_d[idx]),
                "u_u": float(u_u[idx]),
                "v_d": float(v_d[idx]),
                "v_u": float(v_u[idx]),
            }
        )
    return rows


def compute_fast_frontier(
    params: Params,
    D_grid: np.ndarray,
    Nz: int,
    Mu: int,
    shallow_depth: float,
    collect_profiles: bool,
    effort: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fast frontier: min-time under oxygen budget."""
    rows = []
    profile_rows = []
    lam_seed = None
    for D in D_grid:
        Nz_use = int(Nz * 1.5) if D < shallow_depth else Nz
        Mu_use = int(Mu * 2.0) if D < shallow_depth else Mu
        lam, T, O, prof = solve_lambda_for_depth(
            float(D),
            params,
            Nz=Nz_use,
            Mu=Mu_use,
            time_sign=1.0,
            lam_seed=lam_seed,
            effort=effort,
        )
        if not np.isfinite(T):
            rows.append(
                {
                    "effort": effort,
                    "D": D,
                    "T": np.nan,
                    "lambda": np.nan,
                    "O_used": np.nan,
                    "F_frac": np.nan,
                    "P_frac": np.nan,
                    "O_base_frac": np.nan,
                    "O_act_frac": np.nan,
                    "O_mech_frac": np.nan,
                    "O_used_frac": np.nan,
                }
            )
            continue
        usage = compute_usage(prof, params, effort)
        rows.append(
            {
                "effort": effort,
                "D": D,
                "T": T,
                "lambda": lam,
                "O_used": O,
                **usage,
            }
        )
        if collect_profiles:
            profile_rows.extend(profile_rows_from_result("fast", effort, D, prof))
        lam_seed = lam if np.isfinite(lam) else lam_seed
    return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def compute_slow_frontier(
    params: Params,
    D_grid: np.ndarray,
    Nz: int,
    Mu: int,
    slow_start: float,
    shallow_depth: float,
    collect_profiles: bool,
    effort: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Slow frontier: max-time under oxygen budget, starting at D >= z_n."""
    rows = []
    profile_rows = []
    lam_seed = None
    budget = params.V_o2 * effort
    for D in D_grid:
        if D < slow_start:
            rows.append(
                {
                    "effort": effort,
                    "D": D,
                    "T": np.nan,
                    "lambda": np.nan,
                    "O_used": np.nan,
                    "F_frac": np.nan,
                    "P_frac": np.nan,
                    "O_base_frac": np.nan,
                    "O_act_frac": np.nan,
                    "O_mech_frac": np.nan,
                    "O_used_frac": np.nan,
                }
            )
            continue
        Nz_use = int(Nz * 1.5) if D < shallow_depth else Nz
        Mu_use = int(Mu * 2.0) if D < shallow_depth else Mu
        z = np.linspace(0.0, float(D), Nz_use)
        u_grid = np.linspace(0.0, params.F_max, max(20, Mu_use // 4))
        prof_max = optimal_u_profile_min_speed(params, z, u_grid)
        T_max, O_max = integrate_T_O(prof_max, params)
        if np.isfinite(T_max) and np.isfinite(O_max) and O_max <= budget:
            lam = 0.0
            T = T_max
            O = O_max
            prof = prof_max
        else:
            lam, T, O, prof = solve_lambda_for_depth(
                float(D),
                params,
                Nz=Nz_use,
                Mu=Mu_use,
                time_sign=-1.0,
                lam_seed=lam_seed,
                effort=effort,
            )
        if not np.isfinite(T):
            rows.append(
                {
                    "effort": effort,
                    "D": D,
                    "T": np.nan,
                    "lambda": np.nan,
                    "O_used": np.nan,
                    "F_frac": np.nan,
                    "P_frac": np.nan,
                    "O_base_frac": np.nan,
                    "O_act_frac": np.nan,
                    "O_mech_frac": np.nan,
                    "O_used_frac": np.nan,
                }
            )
            continue
        usage = compute_usage(prof, params, effort)
        rows.append(
            {
                "effort": effort,
                "D": D,
                "T": T,
                "lambda": lam,
                "O_used": O,
                **usage,
            }
        )
        if collect_profiles:
            profile_rows.extend(profile_rows_from_result("slow", effort, D, prof))
        lam_seed = lam if np.isfinite(lam) else lam_seed
    return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontiers-csv",
        type=str,
        default="data/frontiers.csv",
        help="Output path for frontiers CSV.",
    )
    parser.add_argument(
        "--profiles-csv",
        type=str,
        default="data/frontier-profiles.csv",
        help="Output path for frontier profiles CSV.",
    )
    parser.add_argument("--D-min", type=float, default=5.0)
    parser.add_argument("--D-max", type=float, default=55.0)
    parser.add_argument("--Dstep", type=float, default=1.0)
    parser.add_argument("--Nz", type=int, default=140)
    parser.add_argument("--Mu", type=int, default=320)
    parser.add_argument("--slow-start", type=float, default=15.0)
    parser.add_argument("--shallow-depth", type=float, default=20.0)
    parser.add_argument(
        "--efforts",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated effort levels (fractions of oxygen budget).",
    )
    parser.add_argument("--v-infinity", type=float, default=Params.v_infinity)
    parser.add_argument("--T-sta", type=float, default=Params.T_sta)
    parser.add_argument("--V-vc", type=float, default=Params.V_vc)
    parser.add_argument("--F-max", type=float, default=Params.F_max)
    parser.add_argument("--P-max", type=float, default=Params.P_max)
    parser.add_argument("--alpha", type=float, default=Params.alpha)
    parser.add_argument("--F-ref", type=float, default=Params.F_ref)
    parser.add_argument("--beta", type=float, default=Params.beta)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.frontiers_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.profiles_csv) or ".", exist_ok=True)

    effort_levels = [float(x) for x in args.efforts.split(",") if x.strip()]
    for effort in effort_levels:
        if effort <= 0.0 or effort > 1.0:
            raise ValueError("Effort levels must be in (0, 1].")

    D_grid = np.arange(args.D_min, args.D_max + 1e-9, args.Dstep)

    frontier_frames = []
    profile_frames = []

    for effort in effort_levels:
        params = Params(
            v_infinity=args.v_infinity,
            T_sta=args.T_sta,
            V_vc=args.V_vc,
            F_max=args.F_max,
            P_max=args.P_max,
            alpha=args.alpha,
            F_ref=args.F_ref,
            beta=args.beta,
        )
        print(f"[effort {effort:.2f}] Using model parameters:")
        for key in sorted(vars(params).keys()):
            print(f"  {key} = {getattr(params, key)}")
        for key in ["L_atm", "V_tlc", "V_o2", "dotV_o2", "F_infinity", "k"]:
            print(f"  {key} = {getattr(params, key)}")

        print("[1/2] Computing fast frontier...")
        fast_df, fast_profiles = compute_fast_frontier(
            params,
            D_grid,
            Nz=args.Nz,
            Mu=args.Mu,
            shallow_depth=args.shallow_depth,
            collect_profiles=True,
            effort=effort,
        )
        fast_df = fast_df.copy()
        fast_df.insert(0, "frontier", "fast")

        print("[2/2] Computing slow frontier...")
        slow_df, slow_profiles = compute_slow_frontier(
            params,
            D_grid,
            Nz=args.Nz,
            Mu=args.Mu,
            slow_start=args.slow_start,
            shallow_depth=args.shallow_depth,
            collect_profiles=True,
            effort=effort,
        )
        slow_df = slow_df.copy()
        slow_df.insert(0, "frontier", "slow")

        fast_depths = fast_df.dropna(subset=["T", "D"])["D"].to_numpy()
        slow_depths = slow_df.dropna(subset=["T", "D"])["D"].to_numpy()
        if fast_depths.size and slow_depths.size:
            depth_limit = float(min(fast_depths.max(), slow_depths.max()))
            step = args.Dstep
            if depth_limit < args.D_max - 1e-9:
                refine_depths = np.arange(depth_limit, depth_limit + step, step / 100)
                refine_depths = np.setdiff1d(refine_depths, D_grid)
                if refine_depths.size:
                    print("[refine] Computing additional depths near depth limit...")
                    fast_refine, fast_profiles_refine = compute_fast_frontier(
                        params,
                        refine_depths,
                        Nz=args.Nz,
                        Mu=args.Mu,
                        shallow_depth=args.shallow_depth,
                        collect_profiles=True,
                        effort=effort,
                    )
                    slow_refine, slow_profiles_refine = compute_slow_frontier(
                        params,
                        refine_depths,
                        Nz=args.Nz,
                        Mu=args.Mu,
                        slow_start=args.slow_start,
                        shallow_depth=args.shallow_depth,
                        collect_profiles=True,
                        effort=effort,
                    )
                    fast_refine = fast_refine.copy()
                    fast_refine.insert(0, "frontier", "fast")
                    slow_refine = slow_refine.copy()
                    slow_refine.insert(0, "frontier", "slow")
                    fast_df = (
                        pd.concat([fast_df, fast_refine], ignore_index=True)
                        .drop_duplicates(subset=["frontier", "effort", "D"])
                        .sort_values("D")
                    )
                    slow_df = (
                        pd.concat([slow_df, slow_refine], ignore_index=True)
                        .drop_duplicates(subset=["frontier", "effort", "D"])
                        .sort_values("D")
                    )
                    fast_profiles = pd.concat(
                        [fast_profiles, fast_profiles_refine], ignore_index=True
                    ).drop_duplicates(subset=["frontier", "effort", "D", "z"])
                    slow_profiles = pd.concat(
                        [slow_profiles, slow_profiles_refine], ignore_index=True
                    ).drop_duplicates(subset=["frontier", "effort", "D", "z"])

        frontier_frames.extend([fast_df, slow_df])
        profile_frames.extend([fast_profiles, slow_profiles])

    frontiers_csv = args.frontiers_csv
    frontiers_df = (
        pd.concat(frontier_frames, ignore_index=True)
        .drop_duplicates(subset=["frontier", "effort", "D"])
        .sort_values(["effort", "frontier", "D"])
    )
    frontiers_df.to_csv(frontiers_csv, index=False)
    print("  wrote:", frontiers_csv)

    profiles_csv = args.profiles_csv
    profiles_df = pd.concat(profile_frames, ignore_index=True)
    if not profiles_df.empty:
        profiles_df = profiles_df.sort_values(["effort", "frontier", "D", "z"])
    profiles_df.to_csv(profiles_csv, index=False)
    print("  wrote:", profiles_csv)


if __name__ == "__main__":
    main()
