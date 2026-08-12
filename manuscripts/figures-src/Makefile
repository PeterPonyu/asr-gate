PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
FIGS := fig0_overview

.PHONY: all clean png
all: $(addsuffix .pdf,$(FIGS)) png

fig0_overview.pdf: fig0_overview.tex
	$(PDFLATEX) fig0_overview.tex
	$(PDFLATEX) fig0_overview.tex

png: fig0_overview.pdf
	pdftoppm -png -r 250 fig0_overview.pdf fig0_overview
	@if [ -f fig0_overview-1.png ]; then mv -f fig0_overview-1.png fig0_overview.png; fi

clean:
	rm -f *.aux *.log *.out
	rm -f $(addsuffix .pdf,$(FIGS)) fig0_overview.png fig0_overview-1.png
