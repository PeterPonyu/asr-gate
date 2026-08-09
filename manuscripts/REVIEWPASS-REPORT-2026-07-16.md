# asr-gate review pass — report (2026-07-16)

Applies the user review directive (docs/records/REVIEW-DIRECTIVE-2026-07-16.md) to
both the canonical `paper.tex` (article class) and the venue kit
`taslp/paper_ieeetran.tex` (IEEE/ACM TASLP, IEEEtran two-column), kept in lockstep.

## Integrity summary (item 8)
- Backups written before any edit: `*.bak-pre-reviewpass` for paper.tex, refs.bib,
  shared.bib and the three taslp/ copies.
- **Zero result-number changes**: the number multiset (every `\d+(\.\d+)?`) of each
  edited `.tex` is byte-identical to its backup (verified both files).
- Prose integrity (full-line-comment-stripped, hyphen-normalised word multiset vs
  backup): the ONLY word-level deltas are the intended ones — `calibrate`→`calibration`
  (leakage fix) and the removed appendix heading "Dataset and Protocol Detail" +
  unreferenced label `app:detail`. Nothing else changed.
- `latexmk` exit 0 on BOTH versions; **zero undefined / multiply-defined refs or
  citations**; no large overfull boxes; no `\clearpage`/`\newpage`.
- Page counts (before→after): canonical **19→19**, TASLP **11→11** (appendix merge
  relocated tables without adding pages).

## Item 1 — style/tone (bold/italic reduction)
Whitespace-robust transform, counts reported per replacement, then diffed.
- `\textbf` occurrences: canonical **32→19**, TASLP **29→16**.
- `\emph` occurrences: canonical **58→25**, TASLP **61→29**.
- Unbolded: all inline verdict words/phrases (`0/20 violations in every tercile`,
  `vacuous at every α`, `non-vacuous`, `twelve … certified cells`) and all
  full-sentence bold claim lead-ins (base-rate disclosure; α=1.5% floor; "We read the
  effect sizes…"; "At the frozen 613-utterance carve…"; "This sub-5% vacuity…";
  "Belle's certificate is calibration-budget-robust…").
- `\textbf{macro}`/`\textbf{micro}` → `\emph{}` (first-use term definition); `\textbf{audit}`
  → plain.
- KEPT as bold (genuine convention): run-in section/paragraph labels (Data., Backbones.,
  Reseeds…, Exchangeability candor…, Disclosed protocol deviations, Disclosed pivot…,
  The multiplicity family…, Named confidence baselines…, Resolution of the single-cell…,
  Verdict-symmetric caveat., Adding the speaker-exchangeability axis…, Interpreting 0/20…,
  C1/C2/Novelty-positioning claim labels), the article title, TODO-USER placeholder,
  IEEE Index Terms line, and the single `certified` table cell.
- Italics thinned to term-definitions / Latin (`a priori`, distribution-free,
  non-monotone, prediction sets, decision, mislabeled mirror split, wrong evaluation
  pool, Pool change/Like-for-like, deliberately weak-backbone stressor, etc.); gratuitous
  emphasis (`\emph{not}`, `\emph{both}`, `\emph{also}`, `\emph{general}`, `\emph{certify}`,
  `\emph{report}`, `\emph{analytic}`, `\emph{documented}`, `\emph{fixed}`, `\emph{budget}`,
  `\emph{contingent}`, …) unwrapped.

## Item 2 — source-code leakage
- Prose scan (non-comment lines) of both files found exactly ONE leaked internal
  identifier in main text: `\texttt{calibrate\_gate}` → re-expressed as "calibration
  gate" (both files).
- `asr-gate` retained (released tool/system name — the paper's subject, allowed).
- All file/script/dir/JSON names (`compute_numbers.py`, `make_figures.py`, `*.json`,
  `*.jsonl`, results-dir names) appear ONLY in `% source:` LaTeX comments, which the
  directive preserves (invisible provenance). Left intact.

## Item 3 — appendix merged, compactness
- Both papers' appendix (`\appendix` / `\appendices` + "Dataset and Protocol Detail"
  heading + label `app:detail`) removed. The two tables (`tab:data`, `tab:deviations`)
  relocated into the main body at the end of Experimental Setup, before Results, so they
  now follow their first in-text references. Their trailing `% source:` provenance
  comments were carried with them (a first pass dropped them; caught by the number-multiset
  check and restored — re-verified identical).
- No `\clearpage`/`\newpage`; TASLP 11 pp is within the 13-page TASLP regular-paper limit
  (note: pages 11+ incur IEEE overlength charges — see limits below).

## Item 4 — figures
- Submission `.tex` already includes every figure via `\includegraphics` of a compiled
  vector PDF — **no inline `tikzpicture`** anywhere; PDF-only-inclusion requirement met.
- Figures are data-driven: `figures/make_figures.py` reads only frozen
  `results/numbers.json` (no hardcoded arrays). Ran it: exit 0, all 5 regenerate, and the
  output is **bit-for-bit identical** to the committed PDFs (fig1 17398 B before=after)
  with proper NimbusRoman/STIX fonts embedded (fonttype 42, no Type-3), effective text
  ≥7pt at the printed span.
- Decision: retained the committed vector PDFs (they reproduce exactly from frozen data
  through the checked-in generator). A TikZ/pgfplots `figures-src/` re-authoring was NOT
  done — the matplotlib generator already satisfies the "regenerate from underlying data"
  and "PDF-only inclusion" requirements, and re-drawing the heatmap/inset/error-bar
  figures in TikZ is high-risk with no submission benefit. Flagged here per "no silent
  skips."

## Item 5 — tables/floats column-span discipline
- No `\resizebox` anywhere (0 in both files).
- Every float is defined AFTER a reference to it exists (checked all 14 labels in TASLP:
  zero premature/orphan floats).
- Float order matches first-reference order except two legitimate forward references the
  prose itself frames as later: `tab:cert` (parenthetical "carried in every table" in the
  method, table belongs with results) and `tab:deviations` (prose says "disclosed **below**
  and in tab:deviations"). No reorder needed.
- Column spans (TASLP two-column): wide tables (`tab:baselines` 7-col, `tab:landscape`
  6-col, `tab:data`/`tab:deviations` wide `p{}` columns, `tab:holm`/`tab:english`/
  `tab:mondrian`/`tab:comparator`/`tab:cert`) use `table*`; single-column figures use
  `figure`, the wide noise panel uses `figure*`. Compiles with no large overfull.

## Item 6 — references (live verification)
All 25 cited keys verified against live sources (arXiv, DOI.org, DBLP, PMLR, ACL, ACM DL).
- **Verified OK: 20.** **Fixed: 3.** **Fabricated/unverifiable: 0.**
- Critically, all six high-risk post-cutoff arXiv IDs genuinely resolve to their claimed
  title+authors (NOT fabricated): angelopoulos2026nonmonotone (2602.20151),
  yu2026joint (2606.08517), xu2025scrc (2512.12844), bai2026conformalselective
  (2603.24704), zhou2026falsesafety (2606.15153), jia2025coverageser (2503.22712).
- Fixes applied to BOTH bib copies (root + taslp):
  1. `piczak2015esc50`: `@misc`→`@inproceedings`; added `doi=10.1145/2733373.2806390`.
  2. `ernez2023conformal`: title completed with the official "(wav2vec 2.0)" suffix.
  3. `bates2021rcps`: pagination `1--34`→`43:1--43:34` (JACM article 43); added
     `doi=10.1145/3478535`.
- Cleared two now-verified `% TODO verify` flags in shared.bib (geifman pages 4878-4887;
  angelopoulos2023conformal DOI 10.1561/2200000101). radford2023whisper year 2023 kept
  (ICML-2023 canonical; arXiv 2212.04356 is the Dec-2022 preprint — internally consistent).

## Item 7 — code archiving (per-venue) — RESOLVED
- Policy file `docs/records/CODE-ARCHIVE-POLICY-2026-07-16.md` was written by code-policy-scan
  during this pass; confirmed finding for TASLP/IEEE: **GitHub + Zenodo DOI is explicitly the
  correct baseline, no pending hedge needed.** Code Ocean is IEEE-wide but optional/encouraged
  (not mandated) with no TASLP-specific elevation — added only as opt-in flavor text IF a
  capsule is actually created (none planned), so NOT added (avoids an unfulfilled namedrop).
- Data/Code Availability statement updated in both files to explicitly name a public GitHub
  repository + Zenodo DOI archive, keeping the `TODO-USER` placeholder for the repo URL and
  DOI. All existing licensing content (ESC-50 CC BY-NC, MagicData CC BY-NC-ND, MUSAN-labelled
  filename disclosure) preserved. No result numbers touched (number multiset still identical).

## Live venue limits (retrieved 2026-07-16)
Source: IEEE Signal Processing Society "Information for Authors"
(https://signalprocessingsociety.org/publications-resources/information-authors),
applies to TASLP.
- Abstract: **150–250 words**, self-contained (no abbreviations/equations/references).
  TASLP abstract is **237 words** — within cap. (Canonical article abstract 307 words;
  journal-agnostic, uncapped — left as is.)
- Regular paper: **≤13 double-column pages** (10-pt). TASLP kit is **11 pp** — compliant.
- Overlength charges: **$220/page for each published page beyond the first 10** — the kit's
  page 11 is the first charged page; flagged for the author (still within the 13-pp cap).

## Build status
- `paper.pdf` — latexmk exit 0, 19 pp, 0 undefined refs/cites.
- `taslp/paper_ieeetran.pdf` — latexmk exit 0, 11 pp, 0 undefined refs/cites.
