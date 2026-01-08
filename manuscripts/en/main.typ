#import "../lib/tmlr.typ": tmlr

#show: tmlr.with(
  title: [Performance Frontier in Freediving: An Exploratory Analysis],
  authors: (
    (
      (
        name: "Tzu-Ming Harry Hsu",
        affl: "independent",
        email: "stmharry@alum.mit.edu",
      ),
      (
        name: "Josh Chang",
        affl: "cat-fish",
        email: "joshfreediving@gmail.com",
      ),
    ),
    (
      independent: (
        institution: "Independent Researcher",
        location: "Taipei",
        country: "Taiwan",
      ),
      cat-fish: (
        institution: "Cat-Fish Ocean Life",
        location: "Hengchun",
        country: "Taiwan",
      ),
    ),
  ),
  keywords: (
    "Freediving",
    "Breath-hold diving",
    "Performance frontier", 
    "Optimal control", 
    "Hydrodynamics and drag", 
    "Metabolic cost modeling", 
    "Oxygen budget normalization", 
    "Training prescription and diagnostics",
  ),
  abstract: include "abstract.typ",
  bibliography: bibliography("../lib/main.bib"),
  appendix: include "appendix.typ",
  accepted: none,
)

#show: doc => {
  show ref: set text(fill: blue)

  doc
}

#include "body.typ"
