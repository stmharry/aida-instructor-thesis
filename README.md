# AIDA Instructor Thesis (Typst)

Research manuscript and experimental analysis for freediving. The main paper is
written in Typst and compiled to PDF, with figures generated from Python scripts
and CSV data.

## Contents

**Original**

- `main.en.typ`: Primary manuscript source (Typst).
- `TRANSLATION.zh-tw.md`: Translation notes and the Markdown source used to translate `main.zh-tw.typ`.
- `data/`: Input data and generated CSV outputs for experiments.
- `scripts/`: Python scripts for analysis and plotting.

**Derivatives**

- `main.zh-tw.typ`: Translated manuscript source (Typst), not the native drafting file.
- `images/`: Generated figures (PDF/PNG).

**Project**

- `Makefile`: Build and plotting recipes.
- `pyproject.toml` / `uv.lock`: Python environment definition.

## Requirements

- [Typst](https://formulae.brew.sh/formula/typst) (`typst`) for compiling the manuscript.
- [Python 3.13+](https://formulae.brew.sh/formula/python@3.13) for analysis and plotting.
- [uv](https://formulae.brew.sh/formula/uv) for managing the Python environment.
- [ImageMagick](https://formulae.brew.sh/formula/imagemagick) (`magick`) for PDF → PNG conversions (optional).

## Setup

Create a Python environment and install dependencies from `pyproject.toml`.
Choose a workflow you prefer (uv, pip, or another tool).

Using uv:

```sh
uv venv && source .venv/bin/activate
uv sync
```

## Build the manuscript

```sh
make
```

## Notes

- Some plots generate CSV outputs into `data/` before plotting (see `Makefile`).
- Plot targets include `make plot.pdf`, `make plot.png`, `make plot-frontier.default.pdf`, `make plot-frontier.default.png`, `make plot-frontier-zx.pdf`, and `make plot-frontier-zx.png`.
