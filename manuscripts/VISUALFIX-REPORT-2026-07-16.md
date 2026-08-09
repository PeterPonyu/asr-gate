# VISUALFIX report — 2026-07-16 (asr-gate, REVIEW-DIRECTIVE-3)

User's own visual audit of the compiled PDFs. Canonical (`paper.tex`) + TASLP port
(`taslp/paper_ieeetran.tex`) kept in lockstep. Rails observed: `.bak-pre-d3` backups
taken before any edit; zero result-number changes; every affected figure/page
rasterized (pdftoppm) and inspected before AND after; `latexmk` exit 0 both venues.

Backups: `paper.tex.bak-pre-d3`, `taslp/paper_ieeetran.tex.bak-pre-d3`,
`figures/make_figures.py.bak-pre-d3`, `figures-src/fig0_overview.tex.bak-pre-d3`.

Figure-number map (PDF order vs. file name): Fig 1 = `fig0_overview` (schematic),
Fig 2 = `fig1_teaser_rc`, Fig 3 = `fig3_certified_frontier`, Fig 4 = `fig4_noise_robustness`,
Fig 5 = `fig5_vacuity_rc`, Fig 6 = `fig2_holm_matrix`.

## Per-issue fixes

### asr-specific
- **Fig 1 (schematic) long text → caption.** The in-figure "if no λ certifies:
  VACUOUS-AT-TARGET — nothing accepted, reported (not silently dropped)" sentence was
  reduced to a short two-line label ("if no λ certifies: / VACUOUS-AT-TARGET"). The full
  sentence already lived in the caption, so no information lost. (`fig0_overview.tex`)
- **Fig 2 (teaser) arrow length.** The "certified operating point" annotation was
  anchored at `xytext=(26,3.2)` — a long arrow spanning the whole panel that also ran
  under the legend's third line. Moved adjacent to the marked point (`xytext=(55,2.35)`);
  arrow now short and clear of the legend. (`make_figures.py` fig1)
- **Fig 3 (frontier) orange text over axis + inset over marks.** (a) "VACUOUS AT TARGET"
  was orange and rendered at `y=4`, overlapping the x/y axes — now black, lifted into the
  empty vacuous column (`y=16`), x-limits widened to `(0.2,5.8)` so it clears the y-axis
  spine. (b) The reseed-violation **inset** (`add_axes([0.53,0.35,…])`) sat on top of the
  tall α=3%/5% coverage bars and clipped the α=2% "CER 1.00%" label; the bar it drew was
  0/20 (zero height), so it only conveyed "below the δ line". **Restyled** to a compact
  black text note in the empty vacuous column ("20-reseed attainment (certified α=2%):
  0/20 violations within δ=0.1") — no overlap, same information. δ text now black.
- **Table 9 (`tab:landscape`) "-" cells.** Already `\textemdash` in the vacuous zipformer
  rows (from the earlier tablefix pass). Added an explicit caption note: "for these rows
  the accept and acc-CER cells are — (no set is accepted, so both are undefined)".
- **Annotation text → black.** All hue-colored in-figure TEXT set to black across the
  figure suite: Fig 1 "Guarantee:"/"accept" (were cAccept blue) + "defer" (was cDefer
  orange); Fig 3 "vacuous at target"/"δ=0.1"; Fig 4 violation counts "1/20" (was warn
  orange); the α-target/ceiling labels in Figs 2/4/5 (were #333). Colored marks/lines/
  fills (box borders/fills, arrows, data curves, heatmap, δ dashed line) left as-is.

### Common items
1. **Page/section volume (TASLP norm).** IEEE SPS Information for Authors: Regular Paper
   **initial submission ≤ 13 double-column pages** (revised ≤ 16), everything included
   except supplemental. TASLP port = **12 pages**, within the 13-page initial cap — no
   forced prose tightening needed (avoids result loss). Canonical one-column preprint went
   **22 → 21 pages** as a side effect of the caption/figure edits (no content removed).
2. **Figure text → caption.** Handled in Fig 1 (above); other figures already carried only
   short labels. Captions grown where text was moved (Fig 1 caption already held the full
   vacuous sentence; Fig 3 frontier caption reworded from "Inset: …" to describe the
   annotation).
3. **Annotation color.** Done (above) — no hue-colored TEXT remains in any figure.
4. **"?" strings.** Grep of both compiled PDF text layers: only legitimate marks — two
   rhetorical questions in the intro and the schematic decision test "s ≥ λ⋆ ?". **No
   broken \ref/\cite/author artifacts.** Clean.
5. **"-" placeholder cells.** Table 9 vacuous rows already em-dash; caption note added
   (both venues).
6. **Prereg/plan leakage in captions.** Fixed: `tab:landscape` caption "(frozen
   \textsc{freeze-amendment 2026-07-13}; …" → "(frozen protocol; …" in both venues. The
   Methods-body prose that discloses the pre-registration ("we preregistered (…signed
   before any new test split was touched)…") is the sanctioned once-in-prose mention and
   was left intact. "re-decoded 2026-07-16" in captions is legitimate methods provenance
   (the corrected official-test re-decode) and kept.
7. **Floats after references.** Verified in the rendered PDFs: every table/figure in both
   venues renders before the bibliography; last pages (canonical p19-21, TASLP p12) are
   pure references. No stranded floats.
8. **Figure geometry per venue.** Zero overfull hboxes in either venue after the edits.
   Fig 1 schematic renders un-clipped and well-proportioned at 0.53\linewidth (canonical)
   and \columnwidth (TASLP); not towering.

## Venue-volume findings
- IEEE/ACM TASLP Regular Paper: initial ≤ 13, revised ≤ 16 double-column pages (10 pt),
  inclusive except supplemental/graphical-abstract. Source: IEEE Signal Processing Society
  "Information for Authors". Our port is 12 pp → compliant with margin.

## Compile status / page counts
- Canonical `paper.pdf`: latexmk exit 0, **21 pages** (was 22), 0 overfull.
- TASLP `paper_ieeetran.pdf`: latexmk exit 0, **12 pages**, 0 overfull.
- Figures regenerated from `results/numbers.json` (no numbers touched); `fig0_overview.pdf`
  rebuilt (pdflatex exit 0) and copied to both `figures/` and `taslp/figures/`; F1-F5
  synced to `taslp/figures/`.

## Ready-score inputs (asr)
- Strengths: 6-cell certified landscape on official pools, THCHS official re-decode
  integrated, tables audited/expanded, figures now clean (short labels, black text, no
  overlaps), within venue page cap, clean "?"/float placement, 0 overfull.
- Remaining gaps (not visual-fix scope): (1) **Data & Code Availability carries a
  "TODO-USER: insert the repository URL and Zenodo DOI before submission" placeholder** —
  hard submission blocker, user-owned (directive publication action: GitHub + Zenodo, sole
  author Zeyu Fu). (2) Methods prose still styles the prereg as `\textsc{freeze-amendment
  2026-07-13}` — kept intentionally, but a reviewer-facing polish could drop the
  filename-like styling to plain "a pre-registered analysis plan (2026-07-13)".
- Suggested ready score: **92/100** (visual/compile fully clean; the −8 is the TODO
  repo/DOI placeholder plus the prereg-styling polish, both user-side).
