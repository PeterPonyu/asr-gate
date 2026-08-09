PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
FIGS := fig0_overview

.PHONY: all clean
all: $(addsuffix .pdf,$(FIGS))

fig0_overview.pdf: fig0_overview.tex
	$(PDFLATEX) fig0_overview.tex
	$(PDFLATEX) fig0_overview.tex

clean:
	rm -f *.aux *.log *.out
	rm -f $(addsuffix .pdf,$(FIGS))
