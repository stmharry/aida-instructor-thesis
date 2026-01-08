#import "lib/tmlr.typ": tmlr

#show: tmlr.with(
  title: [自由潛水中的表現前沿（Performance Frontier）：探索性分析],
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
        institution: "獨立研究者",
        location: "臺北",
        country: "臺灣",
      ),
    ),
  ),
  keywords: (
    "自由潛水（Freediving）",
    "閉氣下潛（Breath-hold diving）",
    "表現前沿（Performance Frontier)", 
    "最佳控制（Optimal control)", 
    "流體力學與阻力（Hydrodynamics and drag)", 
    "代謝成本建模（Metabolic cost modeling)", 
    "氧氣預算標準化（Oxygen budget normalization)", 
    "訓練處方與診斷（Training prescription and diagnostics)",
  ),
  abstract: include "zh-tw/abstract.typ",
  bibliography: bibliography("lib/main.bib"),
  appendix: include "zh-tw/appendix.typ",
  accepted: none,
  aux: (
    font-family: (
      serif: "Noto Sans CJK TC", 
      sans: "Noto Sans CJK TC", 
    ),
  ),
)

#show: doc => {
  set heading(supplement: "章節")
  set figure(supplement: "圖")
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

#include "zh-tw/body.typ"
