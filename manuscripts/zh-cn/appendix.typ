#import "@preview/fancy-units:0.1.1": unit, qty

#import "../lib/macro.typ": O2, CO2, hfrac

#pagebreak()

= 符号表 <sec:glossary>

#table(
  columns: (auto, 1fr),
  inset: (x: 8pt, y: 4pt),
  align: (left, left),
  [*符号*], [*意义／典型值*],

  [$z$], [深度（向下为正）。],
  [$v$], [垂直速度，$v = dot(z)$.],
  [$dot(v)$], [垂直加速度。],
  [$t$], [时间。],
  [$m$], [潜水员 + 装备质量。],
  [$g$], [重力加速度。],
  [$F_"mech" (t)$], [自我产生的推进力（向下为正）。],
  [$F_"mech"^"max"$], [最大瞬时推进力上限，$F_"mech"^"max" approx$ #qty[200][N].],
  [$B(z)$], [浮力，$B(z) = rho_w g V_"disp" (z)$.],
  [$V_"disp" (z)$], [总排水体积。],
  [$V_"const"$], [深度处近乎不可压缩的排水体积（使用防寒衣深度上限时吸收其效应）。],
  [$V_"suit" (z)$], [防寒衣可压缩体积（可选）。],
  // [$V_("suit", 0)$], [Suit volume at surface.],
  // [$V_("suit", oo)$], [Suit volume at infinite depth (deep limit).],
  // [$n_"suit"$], [Suit compression exponent (tunable).],
  [$V_"gas" (z)$], [深度处可压缩气体体积（波以耳压缩）。],
  [$V_("gas", 0)$], [水面气体体积（肺 + 相通气腔）。],
  [$P(z)$], [静水压，$P(z) = P_0 + rho_w g z$.],
  [$P_0$], [水面压力。],
  [$L_p$], [压力长度，$L_p := hfrac(P_0, (rho_w g))$.],
  [$rho_w$], [海水密度。],

  [$F_"drag" (v)$], [流体阻力（与运动方向相反）。],
  [$k$], [阻力常数，$k := (hfrac(1, 2)) rho_w C_arrow.b A$.],
  [$C_arrow.b$], [阻力系数，$C_arrow.b approx$ 0.6--0.8.],
  [$A$], [有效正向截面积，$A approx$ 0.05--0.07 #unit[m^2].],
  // [$F_"drag"^prime (v)$], [Smoothed drag surrogate, $F_"drag"^prime (v) = k v sqrt(v^2 + epsilon^2)$.],
  // [$epsilon$], [Smoothing scale, $epsilon approx$ 0.02 #unit[m/s].],

  [$Delta F_oo$], [深处净载重，$Delta F_oo := m g - rho_w g V_"const"$.],
  [$z_n$], [中性浮力深度，$m g = B(z_n)$.],
  [$z_"failure"$], [失效深度：肺压缩迫使完全吐气的深度（用于由 $V_"VC"$ 估计 $V_"TLC"$）。],
  [$v_oo$], [深处终端速度（若 $Delta F_oo > 0$），$v_oo := sqrt(hfrac(Delta F_oo, k))$.],
  [$Delta F(z)$], [随深度变化的净载重，$Delta F(z) := m g - B(z)$.],
  [$tilde(z)$], [无因次深度，$tilde(z) := hfrac(z, L_p)$.],
  [$tilde(z)_n$], [无因次中性深度，$tilde(z)_n := hfrac(z_n, L_p)$.],
  [$tilde(v)$], [无因次速度，$tilde(v) := hfrac(v, v_oo)$.],
  [$tilde(t)$], [无因次时间，$tilde(t) := hfrac(v_oo, L_p) t$.],
  [$tilde(F)_"mech" (tilde(t))$], [无因次推进力，$tilde(F)_"mech" := hfrac(F_"mech" (t), Delta F_oo)$.],
  [$lambda$], [无因次惯性参数，$lambda := hfrac(m, (k L_p))$.],

  [$dot(V)_#O2$], [可用氧储量的耗竭速率。],
  [$dot(V)_(#O2,"rest")$], [闭气时可用氧储量的基础耗竭速率，$dot(V)_(#O2,"rest")$ $approx$ 2.5--4 #unit[mL/s].],
  [$alpha$], [启动／等长额外成本系数。],
  [$F_"ref"$], [启动项的参考力。],
  [$T_"STA"$], [用于锚定基础耗氧的静态闭气参考时间。],
  [$V_"VC"$], [用于氧气预算校准的肺活量。],
  [$p$], [启动指数。],
  [$beta$], [功率转换为 #O2 的系数，$beta := hfrac(1, (eta e_#O2))$.],
  [$eta$], [总效率，$eta approx$ 0.05--0.095.],
  [$e_#O2$], [每 mL #O2 的能量，$e_#O2 approx$ 20.1 #unit[J/mL].],
  [$P_"mech" (t)$], [正机械功率，$P_"mech" (t) := (F_"mech" (t) dot.op v(t))_+$.],
  [$P_"mech"^"max"$], [最大瞬时机械功率上限，$P_"mech"^"max" approx$ #qty[120][W].],

  [$dot(V)_#CO2$], [瞬时 #CO2 生成率。],
  [$gamma$], [#CO2 与 #O2 的转换因子，$gamma approx$ 0.85--0.95.],

  [$R_#O2 (T)$], [累积耗氧，$R_#O2 (T) := integral_0^T dot(V)_#O2 dif t$.],
  [$R_#CO2 (T)$], [累积 #CO2 生成，$R_#CO2 (T) := integral_0^T dot(V)_#CO2 dif t$.],
  [$R_#O2^"total"$], [可用氧储量（预算）。],
  [$R_#CO2^"total"$], [有效 #CO2 耐受预算。],
  [$E_#O2$], [以 #O2 为基础的努力程度，$E_#O2 := hfrac(R_#O2 (T), R_#O2^"total")$.],
  [$E_#CO2$], [以 #CO2 为基础的努力程度，$E_#CO2 := hfrac(R_#CO2 (T), R_#CO2^"total")$.],
)

= 理论前沿 <sec:appendix-frontier-solvers>

本附录推导在瞬时能力限制与氧气预算下的模型预测最快与最慢前沿 $T_"fast" (D)$ 与 $T_"slow" (D)$。
我们在氧气限制近似下进行（暂不考虑 #CO2），并使用本文已介绍的准稳态、重新参数化力学。

== 共通假设与记号

*深度与阶段。*
深度 $z >= 0$ 向下为正，水面 $z = 0$。
一次下潜在最大深度 $D$ 有单一转折点，先单调下潜再单调上浮。

*控制。*
令推进力大小 $f(t) := abs(F_"mech" (t)) >= 0$。
推进力方向与运动一致：下潜用 $F_"mech" (t) = +f$，上浮用 $F_"mech" (t) = -f$。

*准稳态力学（阻力主导）。*
定义被动漂移项
$
s(z) := v_oo^2 (z - z_n)/(L_p + z),
$
其中 $v_oo$ 为深处终端速度尺度，$z_n$ 为中性浮力深度，$L_p$ 为压力长度。
在 $k$ 常数（推进力对阻力系数）下，准稳态平衡为
$
v abs(v) = s(z) + frac(F_"mech", k).
$
因此速度大小为
$
v_arrow.b (z; f) = sqrt(frac(f, k) + s(z)),
quad
v_arrow.t (z; f) = sqrt(frac(f, k) - s(z)).
$
可行性要求平方根内在相关深度范围非负。

*瞬时限制（硬上限）。*
$
0 <= f <= F_"mech"^"max",
quad
f v_arrow.b (z; f) <= P_"mech"^"max" ("descent"),
quad
f v_arrow.t (z; f) <= P_"mech"^"max" ("ascent").
$
这些定义各深度上的可行集合 $cal(F)_arrow.b (z)$ 与 $cal(F)_arrow.t (z)$。

*耗氧模型（氧气限制）。*
$
dot(V)_#O2
=
dot(V)_(#O2,"rest")
+ alpha (frac(f, F_"ref"))^p
+ beta P_"mech",
quad
P_"mech" = F_"mech" dot.op v.
$
定义有效氧气预算为 $R^"total"$。

=== 时间的深度域表述

以深度参数化每个单调段。
因为 $dif t = (dif z) / v$，到达深度 $D$ 的下潜总时间为
$
T[f_arrow.b, f_arrow.t]
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b (z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t (z))).
$

=== 耗氧的深度域表述

将氧气分为时间积分的基础／力量项，与时间积分的功率项。

*基础 + 力量项。*

在每一段，
$
integral (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p) dif t
=
integral frac(dot(V)_(#O2,"rest") + alpha (frac(f(z), F_"ref"))^p, v(z; f(z))) dif z.
$

*功率项恒等式。*

在推进力对齐且单调运动时，$P_"mech" = f abs(v)$ 且 $dif z = abs(v) dif t$，因此
$
integral beta P_"mech" dif t
=
beta integral f dif z.
$

*完整耗氧泛函。*

因此总耗氧为
$
R[f_arrow.b, f_arrow.t]
=
& integral_0^D frac(dot(V)_(#O2,"rest") + alpha (frac(f_arrow.b (z), F_"ref"))^p, v_arrow.b (z; f_arrow.b (z))) dif z
+ beta integral_0^D f_arrow.b (z) dif z \
&+ integral_0^D frac(dot(V)_(#O2,"rest") + alpha (frac(f_arrow.t (z), F_"ref"))^p, v_arrow.t (z; f_arrow.t (z))) dif z
+ beta integral_0^D f_arrow.t (z) dif z.
$
可行性要求
$
R[f_arrow.b, f_arrow.t] <= R^"total",
quad
f_arrow.b (z) in cal(F)_arrow.b (z),
quad
f_arrow.t (z) in cal(F)_arrow.t (z)
quad
"for" z in [0, D].
$

== 最快前沿推导 <sec:appendix-fast-frontier>

=== 以受约束最小时间问题定义最快前沿

对每个深度 $D$，理论最快前沿为
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

=== 拉格朗日放松与 KKT 条件

引入 $lambda >= 0$ 并定义
$
cal(J)_"fast" [f_arrow.b, f_arrow.t; lambda]
:=
T[f_arrow.b, f_arrow.t] + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$
在最优 $(f_arrow.b^star, f_arrow.t^star, lambda^star)$ 下的 KKT 条件：
$
lambda^star >= 0,
quad
R[f_arrow.b^star, f_arrow.t^star] <= R^"total",
quad
lambda^star (R[f_arrow.b^star, f_arrow.t^star] - R^"total") = 0.
$
因此，
- *预算不活跃：* $R < R^"total" arrow.r lambda^star = 0$，
- *预算活跃：* $R = R^"total" arrow.r lambda^star > 0$。

=== 深度逐点最优性（关键简化）

由于 $cal(J)_"fast"$ 不包含导数 $f'(z)$，固定 $lambda$ 下的最小化可在 $z$ 上逐点分离。
定义每个深度的被积函数：

*下潜。*
$
phi_arrow.b (z, f; lambda)
=
frac(1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p), v_arrow.b (z; f))
+ lambda beta f,
quad f in cal(F)_arrow.b (z).
$

*上浮。*
$
phi_arrow.t (z, f; lambda)
=
frac(1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p), v_arrow.t (z; f))
+ lambda beta f,
quad f in cal(F)_arrow.t (z).
$

则对每个深度 $z$，
$
f_arrow.b^lambda (z) in arg min_(f in cal(F)_arrow.b (z)) phi_arrow.b (z, f; lambda),
quad
f_arrow.t^lambda (z) in arg min_(f in cal(F)_arrow.t (z)) phi_arrow.t (z, f; lambda).
$
这提供对最优控制的解析刻画：每个深度上一维函数的最小化，必要时在可行边界饱和。

=== 内部驻点条件

当最小值落在内部（无主动饱和）时，满足 $partial_arrow.t phi = 0$。
令
$
g (f; lambda)
=
1 + lambda (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p),
quad
g'(f; lambda) = lambda alpha p frac(f^(p-1), F_"ref"^p).
$
且
$
 (dif v_arrow.b (z; f)) / (dif f) = frac(1, 2 k v_arrow.b),
quad
 (dif v_arrow.t (z; f)) / (dif f) = frac(1, 2 k v_arrow.t).
$
则驻点方程具有统一形式（下潜取 $v = v_arrow.b$，上浮取 $v = v_arrow.t$）：
$
lambda beta
+
frac(g'(f; lambda), v(z; f))
-
frac(g (f; lambda), 2 k v(z; f)^3)
= 0.
$
一般而言，这是 $f$ 的隐式方程（解析驻点条件），实际解为分段式：
- 若在 $cal(F)(z)$ 内的内部解由驻点方程给出，
- 否则夹制于主动边界（$f = F_"mech"^"max"$）或 $f v = P_"mech"^"max"$（以及／或平方根可行性边界）。

=== 确定 $lambda^star (D)$

定义由 $lambda$ 所诱发的耗氧：
$
R (lambda; D) := R[f_arrow.b^lambda, f_arrow.t^lambda].
$
则
$
lambda^star (D)
=
cases(
  0 & "if" R (0; D) <= R^"total",
  "solve " R (lambda; D) = R^"total" " for " lambda > 0 & "if" R (0; D) > R^"total",
)
$
在此表述下，增加 $lambda$ 会使耗氧成本加重，通常增加时间并降低耗氧，因此 $R (lambda; D)$ 在实务上具有足够的单调性，允许稳健的根搜寻（例如二分法）。

=== 最快前沿曲线的最终评估

一旦 $lambda^star (D)$ 确定，
$
T_"fast" (D)
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b^ (lambda^star)(z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t^ (lambda^star)(z))).
$
扫描 $D$ 即可得到模型预测的最快前沿 $T_"fast" (D)$。
不存在可行上浮解的深度即为预测深度上限。

== 最慢前沿推导 <sec:appendix-slow-frontier>

对每个深度 $D$，理论最慢前沿定义为在相同力学、上限与氧气预算下的最大时间问题：
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

如同最快前沿推导，我们在深度域中工作，并以可行集合 $cal(F)_arrow.b (z)$ 与 $cal(F)_arrow.t (z)$ 逐点施加可行性。

引入氧气约束的乘子 $lambda >= 0$，并考虑等价的最小化问题
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

定义最慢前沿的拉格朗日式
$
cal(J)_"slow" [f_arrow.b, f_arrow.t; lambda]
:=
(-T[f_arrow.b, f_arrow.t]) + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$

在深度表述中，相对于最快前沿唯一改变的是时间泛函前的符号。
因此，最快前沿推导中所有导致深度局部最优性的步骤都可直接套用，仅需替换
$
(+1) " in the time integrand" -> (-1).
$
