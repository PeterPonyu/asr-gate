# asr-gate manuscript Makefile.
#   make numbers  - recompute results/numbers.json from frozen result JSONs
#   make figures  - regenerate figures/fig{1..5}*.pdf from numbers.json
#   make bib      - copy the portfolio shared bibliography next to the paper
#   make paper    - numbers + figures + bib, then latexmk -pdf
#   make clean    - remove LaTeX aux files
#
# Every number/figure traces to a frozen result JSON; nothing is hand-typed.

SHARED_BIB ?= ../../../shared.bib

.PHONY: all numbers audit-ci figures fig-overview bib paper clean

# relmetrics lives in the asr-gate venv; the excess-AURC bootstrap CIs (Table 2)
# are computed there. Override VENV_PY if your interpreter differs.
VENV_PY ?= ../.venv/bin/python

all: paper

numbers:
	python3 compute_numbers.py

# Speaker-blocked bootstrap CIs on excess-AURC + non-median-split check (Table 2).
# Needs relmetrics (venv). Writes results/audit_ci.json.
audit-ci:
	$(VENV_PY) compute_audit_ci.py

figures: numbers fig-overview
	python3 figures/make_figures.py

# TikZ method-overview schematic (Fig. 1, hand-drawn, not numbers-derived).
fig-overview:
	$(MAKE) -C figures-src fig0_overview.pdf
	cp figures-src/fig0_overview.pdf figures/fig0_overview.pdf

bib:
	@if [ -f "$(SHARED_BIB)" ]; then \
		cp "$(SHARED_BIB)" shared.bib; \
		echo "Copied $(SHARED_BIB) -> shared.bib"; \
	else \
		echo "WARNING: $(SHARED_BIB) not found; creating empty shared.bib"; \
		touch shared.bib; \
	fi

paper: figures bib
	latexmk -pdf -interaction=nonstopmode paper.tex

clean:
	latexmk -C paper.tex 2>/dev/null || true
	rm -f paper.bbl paper.run.xml
