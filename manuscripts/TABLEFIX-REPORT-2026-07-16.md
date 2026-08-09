# TABLEFIX report — 2026-07-16 (asr-gate, REVIEW-DIRECTIVE-2)

Scope: table row-count audit + remedy, figure typography consistency (effective font
size), and decorative-bold/italic sweep (figures + main text), for the canonical
(`paper.tex`) and TASLP (`taslp/paper_ieeetran.tex`) manuscripts, kept in lockstep.

Backups taken before any edit: `paper.tex.bak-pre-tablefix`,
`taslp/paper_ieeetran.tex.bak-pre-tablefix`, `figures/make_figures.py.bak-pre-tablefix`,
`figures-src/fig0_overview.tex.bak-pre-tablefix`.

**Revision note (post-clarification):** the user clarified mid-review that small tables
were placeholders and the *preferred* remedy is expansion from existing frozen records,
not merge/prose — merge only when no expansion data exists. This superseded an initial
pass that merged `tab:comparator`+`tab:baselines`. On re-checking each flagged table's
source file against that guidance: `tab:baselines`'s frozen record
(`baselines_2026-07-15/results.json`) turned out to carry a full 3-backbone × 3-corpus
grid that the paper's 2-row table had summarized away — a clean expansion case, so the
merge was undone and this table now stands alone, expanded. `tab:comparator`'s frozen
record (`M4-M6-RESULT.json`) has no other cells to expand (see below), so it reverted to
its original standalone form with a keep-justification. `tab:mondrian` was re-checked
against this same guidance (see its entry) and its verdict is unchanged.

## 1. Table audit — every table, row count, verdict

| Label | Data rows | Verdict | Why |
|---|---|---|---|
| `tab:data` | 7 | KEEP | ≥5 rows, no action needed |
| `tab:deviations` | 11 | KEEP | ≥5 rows, no action needed |
| `tab:mondrian` | 3 | **KEEP, justified** | Checked `results/mondrian_tercile_analysis.json` for a hidden categorical axis first (per the expansion-preferred guidance): it does carry a `per_reseed` block (20 reseeds × 3 terciles = 60 raw rows), but reseed index is a resampling nuisance parameter, not a category like backbone/corpus/class — the existing table already reports the correct honest summary (mean/min/max + violations + CI per tercile) of that repeated-measures data, so expanding to 60 raw rows would replace a well-organized summary with a data dump, not surface new categorical detail. The row count is otherwise fixed by the design's fixed tercile axis (short/medium/long); no 4th stratum exists without inventing one (gender is dropped per `tab:deviations` for lack of a label), and no adjacent table shares its column schema, so no merge partner exists either. Justification recorded inline as a `% TABLEFIX-2026-07-16 KEEP-JUSTIFIED` comment above the table in both files. |
| `tab:english` | 6 | KEEP | ≥5 rows, no action needed |
| `tab:holm` | 12 | KEEP | ≥5 rows, no action needed |
| `tab:cert` | 7 | KEEP | ≥5 rows, no action needed |
| `tab:comparator` | 4 | **KEEP, justified** | Checked `m4_m6_audit_2026-07-12/M4-M6-RESULT.json` for other cells to expand — it computes this speaker-blocked-robustness check on exactly one cell (Paraformer, Aishell-1 clean); $s_5$ needs a tune-side temperature fit and $s_6$ a learned regressor fit on Aishell dev, both one-off analyses never run elsewhere, so there is no per-corpus/per-backbone detail being summarized away. No merge partner remains either: the (now-expanded) `tab:baselines` reports a different statistic over a different axis, so a forced merge would leave most new rows blank. Kept as-is; justification recorded inline. |
| `tab:baselines` | 2 → **6 (expanded)** | **EXPANDED** using the frozen `baselines_2026-07-15/results.json` (see detail below) | The source file already computes 9 backbone × corpus cells; the paper's table had summarized this down to 2 (Paraformer/Belle, Aishell only) — a direct instance of the directive's "needlessly summarized away" case |
| `tab:landscape` | 9 | KEEP | ≥5 rows, no action needed |

Table count: **9 → 9** (unchanged; no merge, one table expanded).

### Expansion detail: `tab:baselines` (2 rows → 6 rows)

- `baselines_2026-07-15/results.json` computes the named-confidence-baseline audit on
  **9 cells** (Paraformer/Belle/zipformer × Aishell-1/THCHS-30/MagicData). The paper's
  original table showed only 2 of these (Paraformer, Belle — both Aishell-1 only). The
  6 newly-added values per row (5 populated cells + 1 disclosed-exclusion row) all trace
  directly to that frozen JSON — verified programmatically, cell by cell, against the
  table (`per_score.{s1,s2,class_prob_mean,class_prob_min,lengthnorm_pathprob,cem}.
  excess_aurc` and `macro_cer`); zero numbers invented or recomputed.
- **Rows added:** Paraformer/MagicData (shown as an explicit disclosed-exclusion row —
  41/4,094 = 1.0% of rows lack a decoded score, so the whole score family was dropped by
  the audit's non-null gate, matching the paper's existing "excluded, not imputed"
  convention), Belle/MagicData, zipformer/Aishell-1, zipformer/MagicData. The zipformer
  rows are new substantive evidence consistent with the paper's existing landscape
  narrative: its class-probability-derived scores (`s1`, class-prob-mean,
  lengthnorm-pathprob) go *negative* while `s2`/CEM stay positive — added one sentence of
  prose after the "Tsallis entropy" aside connecting this to `tab:landscape`'s zipformer
  `vac.` verdict (verdict-symmetric, not a new claim).
- **THCHS-30 cells deliberately excluded from the expansion**, despite existing in the
  same frozen file, for a data-integrity reason: this baseline run is dated 2026-07-15
  and its THCHS-30 numbers are on the **superseded** $n=1{,}339$ mislabeled-mirror-pool
  decode (macro-CER 3.75/6.62/81.80% for Paraformer/Belle/zipformer) — the exact
  contamination the paper discloses in `tab:deviations` and fixes elsewhere (the
  2026-07-16 official-pool redecode used by `tab:landscape`, which reports 4.07/6.99/
  82.15% for the same three cells). Reporting the stale numbers beside the corrected
  ones would silently reintroduce a disclosed-and-fixed error. The caption and the
  source comment both explain the omission and point to
  `baselines_2026-07-15/RESULTS.md` for the frozen (but superseded) full 9-cell record.
- Both venues' table grew from 7 to 9 columns (added `Corpus`); this required widening
  from `l r r r r r r` to `l l r r r r r r r` and tightening `\tabcolsep` (2pt canonical,
  3.5pt TASLP `table*`) to avoid an overfull hbox in the single-column canonical layout —
  confirmed clean (0 overfull) after the fix.

## 2. Figure typography — effective size (before → after)

Effective size = native-PDF-pt font size × (included width ÷ native PDF width). Native
PDF widths measured via `pdfinfo`; included widths from each venue's `\includegraphics`.
"Min-annot" = the smallest explicit font size used anywhere in that figure (the binding
constraint for the 7pt floor); "label" = the axis-label/title font (the typical largest
recurring text, used to gauge cross-figure consistency toward the ~8–9pt target).

| Figure | Venue | Native pt (bef→aft) | Included pt (bef→aft) | Label eff. (bef→aft) | Min-annot eff. (bef→aft) |
|---|---|---|---|---|---|
| fig0 (overview, TikZ) | canonical | 248.0→248.0 | 291.2→249.0 | 9.40→8.03 | 7.99→7.53 |
| fig0 (overview, TikZ) | TASLP | 248.0→248.0 | 252.0→252.0 | 8.13→8.13 | **6.91→7.62** |
| fig1 (teaser RC) | canonical | 233.6→233.6 | 291.2→234.9 | 11.22→9.05 | 8.11→7.54 |
| fig1 (teaser RC) | TASLP | 233.6→233.6 | 252.0→252.0 | 9.71→9.71 | **7.01→8.09** |
| fig2 (Holm matrix) | canonical | 337.0→306.4 | 291.2→305.3 | 6.91→7.97 | **6.05→7.97** |
| fig2 (Holm matrix) | TASLP | 337.0→306.4 | 252.0→309.6 | 5.98→8.08 | **5.23→8.08** |
| fig3 (frontier) | canonical | 235.6→235.6 | 281.9→239.6 | 10.77→9.15 | 7.18→7.63 |
| fig3 (frontier) | TASLP | 235.6→235.6 | 252.0→252.0 | 9.63→9.63 | **6.42→8.02** |
| fig4 (noise robustness) | canonical | 401.3→401.3 | 446.3→404.0 | 10.01→9.06 | **6.67→7.55** |
| fig4 (noise robustness) | TASLP | 401.3→401.3 | 443.8→443.8 | 9.95→9.95 | **6.64→8.29** |
| fig5 (vacuity) | canonical | 248.7→248.7 | 281.9→249.0 | 10.20→9.01 | 7.71→7.51 |
| fig5 (vacuity) | TASLP | 248.7→248.7 | 252.0→252.0 | 9.12→9.12 | **6.89→7.60** |

Bold = was below the 7pt floor before the fix; all cells are ≥7.5pt after. Label-size
spread also tightened from 5.98–11.22pt (before) to 7.97–9.71pt (after) across the family.

**What changed and why:**
- `figures/make_figures.py`: every explicit `fontsize=` below 7.5 was bumped to 7.5
  (fig1 target label/annotation, fig3 bar labels/inset, fig4 target label/violation
  counts, fig5 ceiling label/legend). Fig2's y-tick condition labels were shortened
  (`"Aishell +noise 5 dB"` → `"+5 dB"`, backbone-name-carries-corpus convention) to shrink
  its `bbox=tight`-inflated native width (337.0pt → 306.4pt, still the widest figure but
  no longer requiring destructive downscale), and its tick/cell/title fonts bumped
  7→8, 8→9.
- `figures-src/fig0_overview.tex`: the `\sub` macro (used for all box subtext) was bumped
  6.8pt→7.5pt — at TASLP's near-1:1 columnwidth scale, 6.8pt native was landing at 6.91pt
  effective, under the floor.
- `\includegraphics` widths were recalculated per figure per venue to land each figure's
  native-to-included scale near 1.0 (previously several figures were arbitrarily
  upscaled ~1.1–1.25× in the canonical single-column layout by reusing the same
  `0.6–0.62\linewidth` fraction regardless of native size). New canonical fractions:
  fig0 0.62→0.53, fig1 0.62→0.50, fig2 0.62→0.65, fig3 0.6→0.51, fig4 0.95→0.86,
  fig5 0.6→0.53.
- **fig2 was promoted `figure`→`figure*` in TASLP** (width `\columnwidth`→`0.6\textwidth`):
  its native content (6×2 matrix + colorbar + tick labels) cannot fit a single
  252pt column without forcing cell/tick text under 7pt effective; this is the only
  structural (non-cosmetic) figure change.

## 3. Decorative bold/italic sweep

**Inside figures:**
- `figures-src/fig0_overview.tex`: 6× `\textbf` (box titles), 1× `\bfseries`
  ("Guarantee:"), 3× `\itshape` (accept/defer arrow labels, vacuous-at-target callout) —
  **all removed** (0 remaining). Structural distinction (title vs. subtext size; accept
  vs. defer lane) is already carried by font-size hierarchy and the existing
  `cAccept`/`cDefer` colors, so the bold/italic was purely decorative.
- `figures/make_figures.py`: 2× `fontweight="bold"` (the `(a)`/`(b)` panel tags on
  fig4) — **kept**, judged structurally meaningful (standard multi-panel labeling
  convention, the figure-internal analog of the class-emitted bold `Figure N:` caption
  label the directive explicitly permits).
- No `fontstyle`/italic usage found in `make_figures.py` (matplotlib figures were
  already clean).

**Captions:** swept all 9 (pre-merge) `\caption{...}` blocks in the canonical file
programmatically for `\textbf`, `\bfseries`, `\emph`, `\textit`, `\itshape` — zero hits.
Clean before and after.

**Main text:** diffed the current `paper.tex` against `paper.tex.bak-pre-thchsofficial`
(the backup immediately preceding this session's other in-flight edits) restricted to
bold/italic macros — zero new `\textbf`/`\bfseries`/`\itshape`/`\textit` introduced by the
recent THCHS-official/Config-B/reference-weave additions. All existing `\textbf{...}`
uses are the paper's established paragraph-lead-in convention (e.g. `\textbf{Data.}`,
`\textbf{(C1) Certified selective triage.}`, the `\textbf{certified}` status flag in
`tab:cert`) or the title/Index-Terms/TODO-USER markers, not decorative emphasis; all
`\emph{...}` uses mark a genuine technical term-of-art or contrastive distinction (e.g.
`\emph{macro}-CER` vs `\emph{micro}-CER`, `\emph{wrong evaluation pool}`), consistent with
directive #1's established rules. No changes needed.

## 4. Compile status and page counts

| | Canonical (`paper.tex`) | TASLP (`taslp/paper_ieeetran.tex`) |
|---|---|---|
| `latexmk -pdf` exit code | 0 | 0 |
| Undefined references | 0 | 0 |
| Multiply-defined labels | 0 | 0 |
| Overfull/underfull hboxes from this edit | 0 (an initial 9-column `tab:baselines` draft overfilled the single-column canonical layout by 25.8pt; fixed by tightening `\tabcolsep` to 2pt — confirmed 0 overfull after) | 0 |
| Pre-existing underfull hboxes (unrelated) | 3, in `tab:data` | several, normal IEEEtran two-column justification artifacts |
| Page count | **22** (was 21 pre-session; +1 from the expanded `tab:baselines`) | **12 (unchanged)** |

TASLP page delta: **0 pages** (still 12pp) — no cost impact against the $220/page-from-p11
threshold. The canonical (non-charged) draft grew by 1 page from the table expansion.

## 5. Diff scope check

`diff` against both `.bak-pre-tablefix` files shows changes confined to exactly the
intended edit classes: (a) 6 `\includegraphics` width lines per venue, (b) the
`tab:mondrian` keep-justification comment, (c) the `tab:comparator` keep-justification
comment (table body itself is byte-identical to the original — verified programmatically,
same 8 excess-AURC/p-value numbers in the same order), (d) the `tab:baselines` expansion
from 2 to 6 rows (new numbers verified cell-by-cell against
`baselines_2026-07-15/results.json`, see §1), (e) the fig2 `figure`→`figure*` promotion
in TASLP only, (f) `\tabcolsep` tightening on the widened `tab:baselines` table. No other
lines touched; no result numbers altered anywhere outside the intentionally expanded
table.

## Files touched
- `paper.tex`, `taslp/paper_ieeetran.tex` (table expansion, two keep-justification
  comments, includegraphics widths, fig2 float promotion in TASLP)
- `figures/make_figures.py` (font-size floor bump, fig2 label shortening)
- `figures-src/fig0_overview.tex` (sub-font bump, bold/italic removal)
- `figures/fig{0..5}*.pdf`, `taslp/figures/fig{0..5}*.pdf` (regenerated, synced)

## 6. Figure CONTENT audit (directive §3b, 2026-07-16 second pass)

Backup taken before edit: `figures/make_figures.py.bak-pre-figcontent` (comment-only
edit; no plotting logic touched). Every figure's data source was cross-checked against
`results/numbers.json`'s full key set to confirm no richer breakdown was being withheld.

| Figure | Source data available | Data actually plotted | Verdict |
|---|---|---|---|
| `fig0_overview` (schematic) | n/a — symbolic TikZ pipeline diagram, no result data | n/a | **N/A** — not a results figure; §3b content-sparseness does not apply. Already cleaned of decorative bold/italic under directive #1/§3 (see §3 above). |
| `fig1_teaser_rc` (`fig:teaser`) | `risk_coverage` carries 6 series (paraformer_clean + musan5/15/25db + whisper_clean + whisper_thchs30) | 1 series (paraformer_clean) + 1 marked operating point | **KEEP, justified** — this is the paper's introductory teaser, whose sole job is to explain the risk–coverage mechanism (gate vs. oracle vs. random-deferral line) before any backbone/noise/cross-corpus comparison exists in the narrative. The other 5 `risk_coverage` series are not withheld: musan5/15/25db are the x-axis of `fig4` (noise arm) and whisper_clean/whisper_thchs30 are the comparison series of `fig5` (backbone vacuity). Overlaying them here would duplicate those figures and dilute the one-curve pedagogical read. Justification recorded inline (`FIGCONTENT-2026-07-16 KEEP-JUSTIFIED`) above `fig1()` in `figures/make_figures.py`. |
| `fig2_holm_matrix` (`fig:holm`) | `holm_realized.rows` = 12 rows (2 backbones × up to 4 conditions × 2 scores) | All 12 rows (full matrix: 6 backbone/condition rows × 2 score columns) | **Already fully enriched** — no placeholder gap; every realized audit cell is on the plot. No change. |
| `fig3_certified_frontier` (`fig:frontier`) | `main_alpha_frontier` has exactly 4 alpha keys (0.01, 0.02, 0.03, 0.05) | All 4 alphas, plus a violation-rate inset from `main_attainment` | **Already fully enriched** — the full alpha grid is plotted; no held-back cells. No change. |
| `fig4_noise_robustness` (`fig:noise`) | `expansion_attainment` (musan5/15/25db) + `main_attainment` (clean) — the complete Paraformer noise arm | All 4 conditions (clean, 25dB, 15dB, 5dB), 2 panels (accepted-CER with error bars, coverage) | **Already fully enriched** — every noise condition in the frozen expansion is plotted; `expansion_attainment`'s 4th key (`thchs30_whisper`) belongs to a different backbone/axis and correctly appears in `fig5` instead, not here. No change. |
| `fig5_vacuity_rc` (`fig:vacuity`) | `risk_coverage` carries 6 series (see fig1 row) | 3 series (paraformer_clean, whisper_clean, whisper_thchs30) | **KEEP, justified** — this panel's specific claim is backbone-driven vacuity at clean condition (certifiable Paraformer vs. vacuous Whisper on two Mandarin corpora); the 3 omitted series are Paraformer's noise arm (a different axis, noise level not backbone) already shown in `fig4`. Adding them would overlay two unrelated claims on one axis. Justification recorded inline (`FIGCONTENT-2026-07-16 KEEP-JUSTIFIED`) above `fig5()` in `figures/make_figures.py`. |

**Decorative-element sweep (§3b directness check, all 6 figures):** no ornamental
frames, gradient fills, or drop shadows in any figure source (`make_figures.py` uses
plain matplotlib patches/lines only; `fig0_overview.tex` uses flat TikZ fills, no
shading). No `ax.grid()` calls anywhere — no gridlines of any kind are drawn, so no
no-value gridlines exist to remove. Legends: `fig1` (3 entries: gate/oracle/random,
all load-bearing) and `fig5` (3 entries: 1 per plotted backbone/corpus, all
load-bearing) both use `frameon=False`; no redundant legend duplicates a caption
statement. Axis space: `fig3`'s inset and `fig4`'s dual-panel layout use the reserved
space for additional real data (violation rate; coverage-cost panel), not empty
margin. No changes required.

**Rebuild verification:** `python3 figures/make_figures.py` re-run after the
comment-only edit; `fig1_teaser_rc.pdf` and `fig5_vacuity_rc.pdf` are byte-identical
in size to the pre-edit files and differ from the previous copies only in the
embedded PDF `/CreationDate` (confirmed via regex extraction) — i.e. zero plotted-content
change, as expected from a comment-only source edit. Fresh copies synced to
`taslp/figures/`. Both `latexmk -pdf` builds re-run clean after resync:

| | Canonical (`paper.tex`) | TASLP (`taslp/paper_ieeetran.tex`) |
|---|---|---|
| `latexmk -pdf` exit code | 0 | 0 |
| Undefined references | 0 | 0 |
| Page count | 22 (unchanged) | 12 (unchanged) |

## Files touched (§3b pass)
- `figures/make_figures.py` (2 keep-justification comments above `fig1()`/`fig5()`;
  no plotting-logic change)
- `figures/fig1_teaser_rc.pdf`, `figures/fig5_vacuity_rc.pdf`,
  `taslp/figures/fig1_teaser_rc.pdf`, `taslp/figures/fig5_vacuity_rc.pdf`
  (re-rendered, content-identical, timestamp-only diff; resynced canonical↔TASLP)
