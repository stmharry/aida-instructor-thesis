#import "@preview/fancy-units:0.1.1": unit, qty

#import "../lib/macro.typ": O2, CO2, hfrac

#pagebreak()

= Glossary of Symbols <sec:glossary>

#table(
  columns: (auto, 1fr),
  inset: (x: 8pt, y: 4pt),
  align: (left, left),
  [*Symbol*], [*Meaning / typical values*],

  [$z$], [Depth (positive downward).],
  [$v$], [Vertical speed, $v = dot(z)$.],
  [$dot(v)$], [Vertical acceleration.],
  [$t$], [Time.],
  [$m$], [Diver + gear mass.],
  [$g$], [Gravitational acceleration.],
  [$F_"mech" (t)$], [Self-generated thrust (downward positive).],
  [$F_"mech"^"max"$], [Maximum instantaneous thrust capacity, $F_"mech"^"max" approx$ #qty[200][N].],
  [$B(z)$], [Buoyancy, $B(z) = rho_w g V_"disp" (z)$.],
  [$V_"disp" (z)$], [Total displaced volume.],
  [$V_"const"$], [Near-incompressible displaced volume at depth (after absorbing suit deep-limit when used).],
  [$V_"suit" (z)$], [Suit compressible volume (optional).],
  // [$V_("suit", 0)$], [Suit volume at surface.],
  // [$V_("suit", oo)$], [Suit volume at infinite depth (deep limit).],
  // [$n_"suit"$], [Suit compression exponent (tunable).],
  [$V_"gas" (z)$], [Compressible gas volume at depth (Boyle-compressed).],
  [$V_("gas", 0)$], [Surface gas volume (lungs + communicating air spaces).],
  [$P(z)$], [Hydrostatic pressure, $P(z) = P_0 + rho_w g z$.],
  [$P_0$], [Surface pressure.],
  [$L_p$], [Pressure length, $L_p := hfrac(P_0, (rho_w g))$.],
  [$rho_w$], [Seawater density.],

  [$F_"drag" (v)$], [Hydrodynamic drag opposing motion.],
  [$k$], [Drag constant, $k := (hfrac(1, 2)) rho_w C_arrow.b A$.],
  [$C_arrow.b$], [Drag coefficient, $C_arrow.b approx$ 0.6--0.8.],
  [$A$], [Effective frontal area, $A approx$ 0.05--0.07 #unit[m^2].],
  // [$F_"drag"^prime (v)$], [Smoothed drag surrogate, $F_"drag"^prime (v) = k v sqrt(v^2 + epsilon^2)$.],
  // [$epsilon$], [Smoothing scale, $epsilon approx$ 0.02 #unit[m/s].],

  [$Delta F_oo$], [Deep net load, $Delta F_oo := m g - rho_w g V_"const"$.],
  [$z_n$], [Neutral depth where $m g = B(z_n)$.],
  [$z_"failure"$], [Failure depth where lung compression forces full exhale (used to estimate $V_"TLC"$ from $V_"VC"$).],
  [$v_oo$], [Deep terminal velocity (if $Delta F_oo > 0$), $v_oo := sqrt(hfrac(Delta F_oo, k))$.],
  [$Delta F(z)$], [Net load vs depth, $Delta F(z) := m g - B(z)$.],
  [$tilde(z)$], [Dimensionless depth, $tilde(z) := hfrac(z, L_p)$.],
  [$tilde(z)_n$], [Dimensionless neutral depth, $tilde(z)_n := hfrac(z_n, L_p)$.],
  [$tilde(v)$], [Dimensionless speed, $tilde(v) := hfrac(v, v_oo)$.],
  [$tilde(t)$], [Dimensionless time, $tilde(t) := hfrac(v_oo, L_p) t$.],
  [$tilde(F)_"mech" (tilde(t))$], [Dimensionless thrust, $tilde(F)_"mech" := hfrac(F_"mech" (t), Delta F_oo)$.],
  [$lambda$], [Dimensionless inertia parameter, $lambda := hfrac(m, (k L_p))$.],

  [$dot(V)_#O2$], [Rate of depletion of the usable oxygen reserve.],
  [$dot(V)_(#O2,"rest")$], [Basal depletion rate of the usable oxygen reserve during apnea, $dot(V)_(#O2,"rest")$ $approx$ 2.5--4 #unit[mL/s].],
  [$alpha$], [Activation / isometric overhead coefficient.],
  [$F_"ref"$], [Reference force for activation term.],
  [$T_"STA"$], [Static apnea reference time used to anchor basal oxygen usage.],
  [$V_"VC"$], [Vital capacity (lung volume) used in the oxygen budget calibration.],
  [$p$], [Activation exponent.],
  [$beta$], [Power-to-#O2 conversion coefficient, $beta := hfrac(1, (eta e_#O2))$.],
  [$eta$], [Gross mechanical efficiency, $eta approx$ 0.05--0.095.],
  [$e_#O2$], [Energy per mL #O2, $e_#O2 approx$ 20.1 #unit[J/mL].],
  [$P_"mech" (t)$], [Positive mechanical power, $P_"mech" (t) := (F_"mech" (t) dot.op v(t))_+$.],
  [$P_"mech"^"max"$], [Maximum instantaneous mechanical power capacity, $P_"mech"^"max" approx$ #qty[120][W].],

  [$dot(V)_#CO2$], [Instantaneous #CO2 generation rate.],
  [$gamma$], [#CO2 to #O2 conversion factor, $gamma approx$ 0.85--0.95.],

  [$R_#O2 (T)$], [Accumulated oxygen consumption, $R_#O2 (T) := integral_0^T dot(V)_#O2 dif t$.],
  [$R_#CO2 (T)$], [Accumulated #CO2 generation, $R_#CO2 (T) := integral_0^T dot(V)_#CO2 dif t$.],
  [$R_#O2^"total"$], [Usable oxygen store (budget).],
  [$R_#CO2^"total"$], [Effective #CO2 tolerance budget.],
  [$E_#O2$], [#O2 based effort level, $E_#O2 := hfrac(R_#O2 (T), R_#O2^"total")$.],
  [$E_#CO2$], [#CO2 based effort level, $E_#CO2 := hfrac(R_#CO2 (T), R_#CO2^"total")$.],
)

= Theoretical Frontiers <sec:appendix-frontier-solvers>

This appendix derives model-predicted fast and slow frontiers $T_"fast" (D)$ and $T_"slow" (D)$, subject to instantaneous capacity limits and an oxygen budget.
We work under the oxygen-limited approximation (ignore #CO2 for now), and the quasi-steady re-parameterized mechanics already introduced in the manuscript.

== Common Assumptions and Notation

*Depth and Phases.*
Depth $z >= 0$ is positive downward, with surface $z = 0$.
A dive has a single turning point at maximum depth $D$, with monotone descent then monotone ascent.

*Control.*
Let thrust magnitude $f(t) := abs(F_"mech" (t)) >= 0$.
Thrust direction is aligned with motion: descent uses $F_"mech" (t) = +f$ and ascent uses $F_"mech" (t) = -f$.

*Quasi-Steady Mechanics (Drag-Dominated).*
Define the passive drift term
$
s(z) := v_oo^2 (z - z_n)/(L_p + z),
$
where $v_oo$ is the terminal speed scale at depth, $z_n$ is neutral depth, and $L_p$ is pressure length.
With constant $k$ (thrust-to-drag coefficient), the quasi-steady balance is
$
v abs(v) = s(z) + frac(F_"mech", k).
$
Therefore the speed magnitudes are
$
v_arrow.b (z; f) = sqrt(frac(f, k) + s(z)),
quad
v_arrow.t (z; f) = sqrt(frac(f, k) - s(z)).
$
Feasibility requires the square-root arguments be nonnegative over the relevant depth ranges.

*Instantaneous Limits (Hard Caps).*
$
0 <= f <= F_"mech"^"max",
quad
f v_arrow.b (z; f) <= P_"mech"^"max" ("descent"),
quad
f v_arrow.t (z; f) <= P_"mech"^"max" ("ascent").
$
These define admissible sets $cal(F)_arrow.b (z)$ and $cal(F)_arrow.t (z)$ at each depth.

*Oxygen Consumption Model (Oxygen-Limited).*
$
dot(V)_#O2
=
dot(V)_(#O2,"rest")
+ alpha (frac(f, F_"ref"))^p
+ beta P_"mech",
quad
P_"mech" = F_"mech" dot.op v.
$
Define the effective oxygen budget as $R^"total"$.

=== Depth-Domain Formulation of Time

Parameterize each monotone segment by depth.
Since $dif t = (dif z) / v$, the total time for a dive reaching depth $D$ is
$
T[f_arrow.b, f_arrow.t]
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b (z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t (z))).
$

=== Depth-Domain Formulation of Oxygen Usage

Split oxygen into basal/force terms integrated over time, and the power term integrated over time.

*Basal + Force Term.*

On each segment,
$
integral (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p) dif t
=
integral frac(dot(V)_(#O2,"rest") + alpha (frac(f(z), F_"ref"))^p, v(z; f(z))) dif z.
$

*Power Term Identity.*

With aligned thrust and monotone travel, $P_"mech" = f abs(v)$ and $dif z = abs(v) dif t$, so
$
integral beta P_"mech" dif t
=
beta integral f dif z.
$

*Full Oxygen Functional.*

Thus the total oxygen usage becomes
$
R[f_arrow.b, f_arrow.t]
=
& integral_0^D frac(dot(V)_(#O2,"rest") + alpha (frac(f_arrow.b (z), F_"ref"))^p, v_arrow.b (z; f_arrow.b (z))) dif z
+ beta integral_0^D f_arrow.b (z) dif z \
&+ integral_0^D frac(dot(V)_(#O2,"rest") + alpha (frac(f_arrow.t (z), F_"ref"))^p, v_arrow.t (z; f_arrow.t (z))) dif z
+ beta integral_0^D f_arrow.t (z) dif z.
$
Feasibility requires
$
R[f_arrow.b, f_arrow.t] <= R^"total",
quad
f_arrow.b (z) in cal(F)_arrow.b (z),
quad
f_arrow.t (z) in cal(F)_arrow.t (z)
quad
"for" z in [0, D].
$

== Fast Frontier Derivation <sec:appendix-fast-frontier>

=== Fast Frontier as a Constrained Minimum-Time Problem

For each depth $D$, the theoretical fast frontier is
$
T_"fast" (D)
:=
min_(f_arrow.b (·), f_arrow.t (·)) T[f_arrow.b, f_arrow.t]
quad
"s.t."
quad
R[f_arrow.b, f_arrow.t] <= R^"total",
quad
f_arrow.b (z) in cal(F)_arrow.b (z),
quad
f_arrow.t (z) in cal(F)_arrow.t (z).
$

=== Lagrangian Relaxation and KKT Conditions

Introduce $lambda >= 0$ and define
$
cal(J)_"fast" [f_arrow.b, f_arrow.t; lambda]
:=
T[f_arrow.b, f_arrow.t] + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$
KKT conditions at optimum $(f_arrow.b^star, f_arrow.t^star, lambda^star)$:
$
lambda^star >= 0,
quad
R[f_arrow.b^star, f_arrow.t^star] <= R^"total",
quad
lambda^star (R[f_arrow.b^star, f_arrow.t^star] - R^"total") = 0.
$
So either:
- *budget inactive:* $R < R^"total" arrow.r lambda^star = 0$,
- *budget active:* $R = R^"total" arrow.r lambda^star > 0$.

=== Pointwise Optimality in Depth (Key Simplification)

Because $cal(J)_"fast"$ contains no derivatives $f'(z)$, for fixed $lambda$ the minimization decouples pointwise in $z$.
Define the per-depth integrands:

*Descent.*
$
phi_arrow.b (z, f; lambda)
=
frac(1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p), v_arrow.b (z; f))
+ lambda beta f,
quad f in cal(F)_arrow.b (z).
$

*Ascent.*
$
phi_arrow.t (z, f; lambda)
=
frac(1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p), v_arrow.t (z; f))
+ lambda beta f,
quad f in cal(F)_arrow.t (z).
$

Then for each depth $z$,
$
f_arrow.b^lambda (z) in arg min_(f in cal(F)_arrow.b (z)) phi_arrow.b (z, f; lambda),
quad
f_arrow.t^lambda (z) in arg min_(f in cal(F)_arrow.t (z)) phi_arrow.t (z, f; lambda).
$
This provides an analytic characterization of the optimal control: it is the minimizer of a one-dimensional function per depth, with saturation at the admissible boundaries when needed.

=== Interior Stationarity Condition

When the minimizer is interior (no active saturation), it satisfies $partial_arrow.t phi = 0$.
Let
$
g (f; lambda)
=
1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p),
quad
g'(f; lambda) = lambda alpha p frac(f^(p-1), F_"ref"^p).
$
Also,
$
 (dif v_arrow.b (z; f)) / (dif f) = frac(1, 2 k v_arrow.b),
quad
 (dif v_arrow.t (z; f)) / (dif f) = frac(1, 2 k v_arrow.t).
$
Then the stationarity equation has the unified form (choose $v = v_arrow.b$ for descent or $v = v_arrow.t$ for ascent):
$
lambda beta
+
frac(g'(f; lambda), v(z; f))
-
frac(g (f; lambda), 2 k v(z; f)^3)
= 0.
$
In general this is an implicit equation for $f$ (closed form in the sense of an analytic stationarity condition), and the actual solution is piecewise:
- interior solution from the stationarity equation when it lies in $cal(F)(z)$,
- otherwise clamped to the active boundary ($f = F_"mech"^"max"$) or $f v = P_"mech"^"max"$ (and/or square-root feasibility boundary).

=== Determining $lambda^star (D)$

Define the oxygen usage induced by $lambda$:
$
R (lambda; D) := R[f_arrow.b^lambda, f_arrow.t^lambda].
$
Then
$
lambda^star (D)
=
cases(
  0 & "if" R (0; D) <= R^"total",
  "solve " R (lambda; D) = R^"total" " for " lambda > 0 & "if" R (0; D) > R^"total",
)
$
Under this formulation, increasing $lambda$ penalizes oxygen more heavily, typically increasing time and decreasing oxygen usage, so $R (lambda; D)$ is monotone enough in practice to permit robust root finding (e.g., bisection).

=== Final Evaluation of the Fast Frontier Curve

Once $lambda^star (D)$ is determined,
$
T_"fast" (D)
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b^ (lambda^star)(z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t^ (lambda^star)(z))).
$
Sweeping $D$ yields the model-predicted fast frontier $T_"fast" (D)$.
Depths where no admissible ascent solution exists define the predicted depth limit.

== Slow Frontier Derivation <sec:appendix-slow-frontier>

For each depth $D$, the theoretical slow frontier is defined as a maximum-time problem under the same mechanics, caps, and oxygen budget used for the fast frontier:
$
T_"slow" (D)
:=
max_(f_arrow.b (·), f_arrow.t (·)) T[f_arrow.b, f_arrow.t]
quad
"s.t."
quad
R[f_arrow.b, f_arrow.t] <= R^"total",
quad
f_arrow.b (z) in cal(F)_arrow.b (z),
quad
f_arrow.t (z) in cal(F)_arrow.t (z).
$

As in the fast frontier derivation, we work in the depth domain and enforce feasibility pointwise via the admissible sets $cal(F)_arrow.b (z)$ and $cal(F)_arrow.t (z)$.

Introduce a multiplier $lambda >= 0$ for the oxygen constraint and consider the equivalent minimization problem
$
max T
quad
"subject to"
quad
R <= R^"total"
quad
<==>
quad
min (-T)
quad
"subject to"
quad
R <= R^"total".
$

Define the slow-frontier Lagrangian
$
cal(J)_"slow" [f_arrow.b, f_arrow.t; lambda]
:=
(-T[f_arrow.b, f_arrow.t]) + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$

In the depth formulation, the only change relative to the fast frontier is the sign in front of the time functional.
Consequently, all steps in the fast-frontier derivation that yield depth-local optimality carry over, with the replacement
$
(+1) " in the time integrand" -> (-1).
$
