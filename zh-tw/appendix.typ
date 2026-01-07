#import "@preview/fancy-units:0.1.1": unit, qty

#import "../lib/macro.typ": O2, CO2, hfrac

#pagebreak()

= 符號表 <sec:glossary>

#table(
  columns: (auto, 1fr),
  inset: (x: 8pt, y: 4pt),
  align: (left, left),
  [*符號*], [*意義／典型值*],

  [$z$], [深度（向下為正）。],
  [$v$], [垂直速度，$v = dot(z)$.],
  [$dot(v)$], [垂直加速度。],
  [$t$], [時間。],
  [$m$], [潛水員 + 裝備質量。],
  [$g$], [重力加速度。],
  [$F_"mech" (t)$], [自我產生的推進力（向下為正）。],
  [$F_"mech"^"max"$], [最大瞬時推進力上限，$F_"mech"^"max" approx$ #qty[200][N].],
  [$B(z)$], [浮力，$B(z) = rho_w g V_"disp" (z)$.],
  [$V_"disp" (z)$], [總排水體積。],
  [$V_"const"$], [深度處近乎不可壓縮的排水體積（使用防寒衣深度上限時吸收其效應）。],
  [$V_"suit" (z)$], [防寒衣可壓縮體積（可選）。],
  // [$V_("suit", 0)$], [Suit volume at surface.],
  // [$V_("suit", oo)$], [Suit volume at infinite depth (deep limit).],
  // [$n_"suit"$], [Suit compression exponent (tunable).],
  [$V_"gas" (z)$], [深度處可壓縮氣體體積（波以耳壓縮）。],
  [$V_("gas", 0)$], [水面氣體體積（肺 + 相通氣腔）。],
  [$P(z)$], [靜水壓，$P(z) = P_0 + rho_w g z$.],
  [$P_0$], [水面壓力。],
  [$L_p$], [壓力長度，$L_p := hfrac(P_0, (rho_w g))$.],
  [$rho_w$], [海水密度。],

  [$F_"drag" (v)$], [流體阻力（與運動方向相反）。],
  [$k$], [阻力常數，$k := (hfrac(1, 2)) rho_w C_arrow.b A$.],
  [$C_arrow.b$], [阻力係數，$C_arrow.b approx$ 0.6--0.8.],
  [$A$], [有效正向截面積，$A approx$ 0.05--0.07 #unit[m^2].],
  // [$F_"drag"^prime (v)$], [Smoothed drag surrogate, $F_"drag"^prime (v) = k v sqrt(v^2 + epsilon^2)$.],
  // [$epsilon$], [Smoothing scale, $epsilon approx$ 0.02 #unit[m/s].],

  [$Delta F_oo$], [深處淨載重，$Delta F_oo := m g - rho_w g V_"const"$.],
  [$z_n$], [中性浮力深度，$m g = B(z_n)$.],
  [$z_"failure"$], [失效深度：肺壓縮迫使完全吐氣的深度（用於由 $V_"VC"$ 估計 $V_"TLC"$）。],
  [$v_oo$], [深處終端速度（若 $Delta F_oo > 0$），$v_oo := sqrt(hfrac(Delta F_oo, k))$.],
  [$Delta F(z)$], [隨深度變化的淨載重，$Delta F(z) := m g - B(z)$.],
  [$tilde(z)$], [無因次深度，$tilde(z) := hfrac(z, L_p)$.],
  [$tilde(z)_n$], [無因次中性深度，$tilde(z)_n := hfrac(z_n, L_p)$.],
  [$tilde(v)$], [無因次速度，$tilde(v) := hfrac(v, v_oo)$.],
  [$tilde(t)$], [無因次時間，$tilde(t) := hfrac(v_oo, L_p) t$.],
  [$tilde(F)_"mech" (tilde(t))$], [無因次推進力，$tilde(F)_"mech" := hfrac(F_"mech" (t), Delta F_oo)$.],
  [$lambda$], [無因次慣性參數，$lambda := hfrac(m, (k L_p))$.],

  [$dot(V)_#O2$], [可用氧儲量的耗竭速率。],
  [$dot(V)_(#O2,"rest")$], [閉氣時可用氧儲量的基礎耗竭速率，$dot(V)_(#O2,"rest")$ $approx$ 2.5--4 #unit[mL/s].],
  [$alpha$], [啟動／等長額外成本係數。],
  [$F_"ref"$], [啟動項的參考力。],
  [$T_"STA"$], [用於錨定基礎耗氧的靜態閉氣參考時間。],
  [$V_"VC"$], [用於氧氣預算校準的肺活量。],
  [$p$], [啟動指數。],
  [$beta$], [功率轉換為 #O2 的係數，$beta := hfrac(1, (eta e_#O2))$.],
  [$eta$], [總效率，$eta approx$ 0.05--0.095.],
  [$e_#O2$], [每 mL #O2 的能量，$e_#O2 approx$ 20.1 #unit[J/mL].],
  [$P_"mech" (t)$], [正機械功率，$P_"mech" (t) := (F_"mech" (t) dot.op v(t))_+$.],
  [$P_"mech"^"max"$], [最大瞬時機械功率上限，$P_"mech"^"max" approx$ #qty[120][W].],

  [$dot(V)_#CO2$], [瞬時 #CO2 生成率。],
  [$gamma$], [#CO2 與 #O2 的轉換因子，$gamma approx$ 0.85--0.95.],

  [$R_#O2 (T)$], [累積耗氧，$R_#O2 (T) := integral_0^T dot(V)_#O2 dif t$.],
  [$R_#CO2 (T)$], [累積 #CO2 生成，$R_#CO2 (T) := integral_0^T dot(V)_#CO2 dif t$.],
  [$R_#O2^"total"$], [可用氧儲量（預算）。],
  [$R_#CO2^"total"$], [有效 #CO2 耐受預算。],
  [$E_#O2$], [以 #O2 為基礎的努力程度，$E_#O2 := hfrac(R_#O2 (T), R_#O2^"total")$.],
  [$E_#CO2$], [以 #CO2 為基礎的努力程度，$E_#CO2 := hfrac(R_#CO2 (T), R_#CO2^"total")$.],
)

= 理論前沿 <sec:appendix-frontier-solvers>

本附錄推導在瞬時能力限制與氧氣預算下的模型預測最快與最慢前沿 $T_"fast" (D)$ 與 $T_"slow" (D)$。
我們在氧氣限制近似下進行（暫不考慮 #CO2），並使用本文已介紹的準穩態、重新參數化力學。

== 共通假設與記號

*深度與階段。*
深度 $z >= 0$ 向下為正，水面 $z = 0$。
一次下潛在最大深度 $D$ 有單一轉折點，先單調下潛再單調上浮。

*控制。*
令推進力大小 $f(t) := abs(F_"mech" (t)) >= 0$。
推進力方向與運動一致：下潛用 $F_"mech" (t) = +f$，上浮用 $F_"mech" (t) = -f$。

*準穩態力學（阻力主導）。*
定義被動漂移項
$
s(z) := v_oo^2 (z - z_n)/(L_p + z),
$
其中 $v_oo$ 為深處終端速度尺度，$z_n$ 為中性浮力深度，$L_p$ 為壓力長度。
在 $k$ 常數（推進力對阻力係數）下，準穩態平衡為
$
v abs(v) = s(z) + frac(F_"mech", k).
$
因此速度大小為
$
v_arrow.b (z; f) = sqrt(frac(f, k) + s(z)),
quad
v_arrow.t (z; f) = sqrt(frac(f, k) - s(z)).
$
可行性要求平方根內在相關深度範圍非負。

*瞬時限制（硬上限）。*
$
0 <= f <= F_"mech"^"max",
quad
f v_arrow.b (z; f) <= P_"mech"^"max" ("descent"),
quad
f v_arrow.t (z; f) <= P_"mech"^"max" ("ascent").
$
這些定義各深度上的可行集合 $cal(F)_arrow.b (z)$ 與 $cal(F)_arrow.t (z)$。

*耗氧模型（氧氣限制）。*
$
dot(V)_#O2
=
dot(V)_(#O2,"rest")
+ alpha (frac(f, F_"ref"))^p
+ beta P_"mech",
quad
P_"mech" = F_"mech" dot.op v.
$
定義有效氧氣預算為 $R^"total"$。

=== 時間的深度域表述

以深度參數化每個單調段。
因為 $dif t = (dif z) / v$，到達深度 $D$ 的下潛總時間為
$
T[f_arrow.b, f_arrow.t]
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b (z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t (z))).
$

=== 耗氧的深度域表述

將氧氣分為時間積分的基礎／力量項，與時間積分的功率項。

*基礎 + 力量項。*

在每一段，
$
integral (dot(V)_(#O2,"rest") + alpha (frac(f, F_"ref"))^p) dif t
=
integral frac(dot(V)_(#O2,"rest") + alpha (frac(f(z), F_"ref"))^p, v(z; f(z))) dif z.
$

*功率項恆等式。*

在推進力對齊且單調運動時，$P_"mech" = f abs(v)$ 且 $dif z = abs(v) dif t$，因此
$
integral beta P_"mech" dif t
=
beta integral f dif z.
$

*完整耗氧泛函。*

因此總耗氧為
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

== 最快前沿推導 <sec:appendix-fast-frontier>

=== 以受約束最小時間問題定義最快前沿

對每個深度 $D$，理論最快前沿為
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

=== 拉格朗日放鬆與 KKT 條件

引入 $lambda >= 0$ 並定義
$
cal(J)_"fast" [f_arrow.b, f_arrow.t; lambda]
:=
T[f_arrow.b, f_arrow.t] + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$
在最優 $(f_arrow.b^star, f_arrow.t^star, lambda^star)$ 下的 KKT 條件：
$
lambda^star >= 0,
quad
R[f_arrow.b^star, f_arrow.t^star] <= R^"total",
quad
lambda^star (R[f_arrow.b^star, f_arrow.t^star] - R^"total") = 0.
$
因此，
- *預算不活躍：* $R < R^"total" arrow.r lambda^star = 0$，
- *預算活躍：* $R = R^"total" arrow.r lambda^star > 0$。

=== 深度逐點最優性（關鍵簡化）

由於 $cal(J)_"fast"$ 不包含導數 $f'(z)$，固定 $lambda$ 下的最小化可在 $z$ 上逐點分離。
定義每個深度的被積函數：

*下潛。*
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

則對每個深度 $z$，
$
f_arrow.b^lambda (z) in arg min_(f in cal(F)_arrow.b (z)) phi_arrow.b (z, f; lambda),
quad
f_arrow.t^lambda (z) in arg min_(f in cal(F)_arrow.t (z)) phi_arrow.t (z, f; lambda).
$
這提供對最優控制的解析刻畫：每個深度上一維函數的最小化，必要時在可行邊界飽和。

=== 內部駐點條件

當最小值落在內部（無主動飽和）時，滿足 $partial_arrow.t phi = 0$。
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
則駐點方程具有統一形式（下潛取 $v = v_arrow.b$，上浮取 $v = v_arrow.t$）：
$
lambda beta
+
frac(g'(f; lambda), v(z; f))
-
frac(g (f; lambda), 2 k v(z; f)^3)
= 0.
$
一般而言，這是 $f$ 的隱式方程（解析駐點條件），實際解為分段式：
- 若在 $cal(F)(z)$ 內的內部解由駐點方程給出，
- 否則夾制於主動邊界（$f = F_"mech"^"max"$）或 $f v = P_"mech"^"max"$（以及／或平方根可行性邊界）。

=== 確定 $lambda^star (D)$

定義由 $lambda$ 所誘發的耗氧：
$
R (lambda; D) := R[f_arrow.b^lambda, f_arrow.t^lambda].
$
則
$
lambda^star (D)
=
cases(
  0 & "if" R (0; D) <= R^"total",
  "solve " R (lambda; D) = R^"total" " for " lambda > 0 & "if" R (0; D) > R^"total",
)
$
在此表述下，增加 $lambda$ 會使耗氧成本加重，通常增加時間並降低耗氧，因此 $R (lambda; D)$ 在實務上具有足夠的單調性，允許穩健的根搜尋（例如二分法）。

=== 最快前沿曲線的最終評估

一旦 $lambda^star (D)$ 確定，
$
T_"fast" (D)
=
integral_0^D frac(dif z, v_arrow.b (z; f_arrow.b^ (lambda^star)(z)))
+ integral_0^D frac(dif z, v_arrow.t (z; f_arrow.t^ (lambda^star)(z))).
$
掃描 $D$ 即可得到模型預測的最快前沿 $T_"fast" (D)$。
不存在可行上浮解的深度即為預測深度上限。

== 最慢前沿推導 <sec:appendix-slow-frontier>

對每個深度 $D$，理論最慢前沿定義為在相同力學、上限與氧氣預算下的最大時間問題：
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

如同最快前沿推導，我們在深度域中工作，並以可行集合 $cal(F)_arrow.b (z)$ 與 $cal(F)_arrow.t (z)$ 逐點施加可行性。

引入氧氣約束的乘子 $lambda >= 0$，並考慮等價的最小化問題
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

定義最慢前沿的拉格朗日式
$
cal(J)_"slow" [f_arrow.b, f_arrow.t; lambda]
:=
(-T[f_arrow.b, f_arrow.t]) + lambda (R[f_arrow.b, f_arrow.t] - R^"total").
$

在深度表述中，相對於最快前沿唯一改變的是時間泛函前的符號。
因此，最快前沿推導中所有導致深度局部最優性的步驟都可直接套用，僅需替換
$
(+1) " in the time integrand" -> (-1).
$
