#import "lib/tmlr.typ": tmlr

#show: tmlr.with(
  title: [自由潜水中的表现前沿（Performance Frontier）：探索性分析],
  authors: (
    (
      (
        name: "Tzu-Ming (Harry) Hsu",
        affl: "independent",
        email: "stmharry@alum.mit.edu",
      ),
    ),
    (
      independent: (
        institution: "独立研究者",
        location: "台北",
        country: "台湾",
      ),
    ),
  ),
  keywords: (
    "自由潜水（Freediving）",
    "闭气下潜（Breath-hold diving）",
    "表现前沿（Performance Frontier)", 
    "最佳控制（Optimal control)", 
    "流体力学与阻力（Hydrodynamics and drag)", 
    "代谢成本建模（Metabolic cost modeling)", 
    "氧气预算标准化（Oxygen budget normalization)", 
    "训练处方与诊断（Training prescription and diagnostics)",
  ),
  abstract: include "zh-cn/abstract.typ",
  bibliography: bibliography("lib/main.bib"),
  appendix: include "zh-cn/appendix.typ",
  accepted: none,
  aux: (
    font-family: (
      serif: "Noto Sans CJK TC", 
      sans: "Noto Sans CJK TC", 
    ),
  ),
)

#show: doc => {
  set heading(supplement: "章节")
  set figure(supplement: "图")
  set math.equation(supplement: "方程")
  show ref: set text(fill: blue)

  // intercept above template's show-set
  show ref: it => {
    let el = it.element

    if el != none and el.func() == math.equation {
      let number = numbering("1", ..counter(math.equation).at(el.location()))

      [#text(fill: blue, [#el.supplement#number])]
    }
    else {
      it
    }
  }

  doc
}

#include "zh-cn/body.typ"
