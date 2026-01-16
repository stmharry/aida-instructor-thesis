TYPST ?= typst
MAGICK ?= magick -density 300
CODEX ?= codex
UV ?= uv
PYTHON ?= python

STATIC_IMAGES = \
	images/depth-vs-dive.jpg \
	images/depth-vs-time.jpg \
	images/zx.a.jpg \
	images/zx.b.jpg \
	images/yzy.jpg

MANUSCRIPT_IMAGES = \
	$(STATIC_IMAGES) \
	$(PLOT_TARGETS) \
	$(PLOT_FRONTIER_DEFAULT_TARGETS) \
	$(PLOT_FRONTIER_ZX_TARGETS)

LOCALES = en zh-tw zh-cn
MANUSCRIPT_DIR = manuscripts
MANUSCRIPT_PDFS = \
	$(patsubst %, $(MANUSCRIPT_DIR)/%/main.pdf, $(LOCALES))

.PHONY: main
main: $(MANUSCRIPT_PDFS)

$(MANUSCRIPT_DIR)/%/main.pdf: $(MANUSCRIPT_DIR)/%/main.typ $(MANUSCRIPT_IMAGES)
	@$(TYPST) compile $< $@ --root . 

$(MANUSCRIPT_DIR)/zh-cn/main.typ: $(MANUSCRIPT_DIR)/zh-tw/main.typ
	@set -eu; \
	find $(MANUSCRIPT_DIR)/zh-tw -name '*.typ' | while read -r source; do \
	  target=$$(echo "$$source" | sed 's#/zh-tw/#/zh-cn/#'); \
	  mkdir -p "$$(dirname "$$target")"; \
	  opencc -c t2s.json -i "$$source" -o "$$target"; \
	done

$(MANUSCRIPT_DIR)/zh-tw/main.typ: $(MANUSCRIPT_DIR)/zh-tw/TRANSLATION.md
	@$(CODEX) exec $< 

%.png: %.pdf
	@$(MAGICK) $< $@

%.jpg: %.pdf
	@$(MAGICK) $< -quality 70 $@

#

PLOT_TARGETS = \
	images/example-profile.pdf \
	images/example-budget.pdf \
	images/td-diagram.pdf \
	images/dive-profiles.pdf

.PHONY: plot.pdf
plot.pdf: $(PLOT_TARGETS)

.PHONY: plot.png
plot.png: $(PLOT_TARGETS:.pdf=.png)

images/example-profile.pdf: scripts/plot-example-profile.py
	@$(PYTHON) $< --output $@ 

images/example-budget.pdf: scripts/plot-example-budget.py
	@$(PYTHON) $< --output $@

images/td-diagram.pdf: scripts/plot-td-diagram.py
	@$(PYTHON) $< --output $@

images/dive-profiles.pdf: scripts/plot-dive-profiles.py
	@$(PYTHON) $< --output $@

#

DIVES_DEFAULT ?= data/dives.csv
FRONTIERS_DEFAULT ?= data/frontiers.csv
PROFILES_DEFAULT ?= data/frontier-profiles.csv

PLOT_FRONTIER_DEFAULT_TARGETS = \
	images/frontiers.pdf \
	images/frontier-usage.pdf \
	images/frontier-profiles-slow.pdf \
	images/frontier-profiles-fast.pdf \
	images/frontiers-efforts.pdf

.PHONY: plot-frontier.default.pdf
plot-frontier.default.pdf: $(PLOT_FRONTIER_DEFAULT_TARGETS) 

.PHONY: ppython scripts/plot-frontiers.py --lang zh-twlot-frontier.default.png
plot-frontier.default.png: $(PLOT_FRONTIER_DEFAULT_TARGETS:.pdf=.png)

$(FRONTIERS_DEFAULT) $(PROFILES_DEFAULT): scripts/compute-frontiers.py
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_DEFAULT) \
	  --profiles-csv $(PROFILES_DEFAULT) \
	  --D-max 55.0 \
	  --Dstep 1.0 \
	  --v-infinity 0.8 \
	  --T-sta 240 \
	  --V-vc 5.0 \
	  --F-max 200 \
	  --P-max 200 \
	  --alpha 15.0 \
	  --beta 0.2

images/frontiers.pdf: scripts/plot-frontiers.py $(FRONTIERS_DEFAULT) $(DIVES_DEFAULT)
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_DEFAULT) \
	  --dives-csv $(DIVES_DEFAULT) \
	  --output $@ \
	  --T-sta 240.0

images/frontier-usage.pdf: scripts/plot-frontier-usage.py $(FRONTIERS_DEFAULT)
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_DEFAULT) \
	  --output $@ 

images/frontier-profiles-%.pdf: scripts/plot-frontier-profiles.py $(PROFILES_DEFAULT)
	@$(PYTHON) $< \
	  --profiles-csv $(PROFILES_DEFAULT) \
	  --output-template images/frontier-profiles-{frontier}.pdf

images/frontiers-efforts.pdf: scripts/plot-frontier-efforts.py $(FRONTIERS_DEFAULT) $(DIVES_DEFAULT)
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_DEFAULT) \
	  --dives-csv $(DIVES_DEFAULT) \
	  --output $@ 

#

DIVES_ZX ?= data/dives.zx.csv
FRONTIERS_ZX ?= data/frontiers.zx.csv
PROFILES_ZX ?= data/frontier-profiles.zx.csv

PLOT_FRONTIER_ZX_TARGETS = \
	images/frontiers.zx.pdf \
	images/frontier-usage.zx.pdf

.PHONY: plot-frontier.zx.pdf
plot-frontier.zx.pdf: $(PLOT_FRONTIER_ZX_TARGETS)

.PHONY: plot-frontier.zx.png
plot-frontier.zx.png: $(PLOT_FRONTIER_ZX_TARGETS:.pdf=.png)

$(FRONTIERS_ZX) $(PROFILES_ZX): scripts/compute-frontiers.py
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_ZX) \
	  --profiles-csv $(PROFILES_ZX) \
	  --D-min 5.0 \
	  --D-max 75.0 \
	  --Dstep 1.0 \
	  --Nz 140 \
	  --Mu 320 \
	  --slow-start 15.0 \
	  --shallow-depth 20.0 \
	  --efforts 1.0 \
	  --v-infinity 0.85 \
	  --T-sta 245 \
	  --V-vc 5.0 \
	  --F-max 200 \
	  --P-max 250 \
	  --alpha 10.0 \
	  --F-ref 100.0 \
	  --beta 0.06

images/frontiers.zx.pdf: scripts/plot-frontiers.py $(FRONTIERS_ZX) $(DIVES_ZX)
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_ZX) \
	  --dives-csv $(DIVES_ZX) \
	  --output $@ \
	  --T-sta 245.0

images/frontier-usage.zx.pdf: scripts/plot-frontier-usage.py $(FRONTIERS_ZX)
	@$(PYTHON) $< \
	  --frontiers-csv $(FRONTIERS_ZX) \
	  --output $@
