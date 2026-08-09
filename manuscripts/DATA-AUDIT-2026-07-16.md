# asr-gate manuscript data audit — 2026-07-16

Read-only audit of `paper.tex` (canonical, 1045 lines) and `taslp/paper_ieeetran.tex`
(IEEEtran port, 990 lines), plus the frozen result JSONs their `% source` comments point to.
Two dimensions: (1) cross-paper/cross-location number consistency; (2) assumption-based claims
exceeding the frozen records. Nothing in the manuscripts was edited.

**Verdict:** 2 CRITICAL, 6 MINOR. The two papers are otherwise numerically faithful to the
frozen artifacts — the overwhelming majority of numbers reproduce bit-for-bit. Both papers
share both CRITICAL findings identically (the IEEEtran file is a faithful port, so every
finding below lives in both).

Findings are ranked by severity. CRITICAL = a number that contradicts another location/table
or a factual/integrity claim the records do not support; MINOR = rounding, loose wording, or an
under-supported interpretive aside.

---

## Coverage evidence (what was checked)

**Numbers traced to source and confirmed correct (dimension 1):**
- `main_attainment` (numbers.json): acceptance 85.7/96.2/90.7%, accepted macro-CER
  1.00/1.53/1.21%, 0 violations — matches abstract, §Results, `tab:cert`, `fig:teaser/frontier`. ✓
- `accepted_micro_2026-07-13.json`: accepted micro-CER 1.02/1.54/1.23% — matches `tab:cert`, body. ✓
- `tab:cert` full-set macro-CER for all 6 rows recomputed from `expansion/audit/audit_*.json`
  (1.98 / 3.80 / 2.30 / 2.04 / 28.06 / 9.93%) — all match. ✓
- `tab:holm` (12 rows): every excess-AURC and 95% CI matches `numbers.json holm_realized.rows`
  and `audit_ci.json` to the printed digit (0.0118…0.0513; CIs e.g. [0.0101,0.0137]). ✓
- `tab:comparator` (s1/s2/s5/s6 = 0.0118/0.0137/0.0118/0.0125; p_Holm 0.0015/0.0005) —
  matches `m4_m6_audit_2026-07-12/M4-M6-RESULT.json`. ✓
- `tab:baselines` (Paraformer + Belle rows, 12 values) — matches `baselines_2026-07-15/results.json`. ✓
- `tab:english` (6 cells: accept 97.99–100%, CER 0.61–3.90%, KS 0.176/0.280/0.247;
  λ*=−0.0959/−0.1989/−0.1498) — matches `english_arm_fixed_2026-07-12/ATTAINMENT.json` +
  `english_arm2_2026-07-13/ATTAINMENT2.json` and the frozen gate JSONs. ✓
- `tab:mondrian` (dur0/1/2 acceptance, acc-CER, min/max, reseed-0 CIs — 24 values) — matches
  `mondrian_tercile_analysis.json` exactly. ✓
- `tab:landscape` cell values (tightest α, accept, acc-CER, full-set CER, all 9 rows) — every
  value reproduces from `numbers.json landscape.attainment` and `landscape/audit/audit_*.json`
  (+ `capped_magicdata_2026-07-15/` for the MagicData rows). ✓ — **but see CRITICAL-2 for the
  THCHS-30 pool those THCHS rows were computed on.**
- Noise, speaker-partition, and calibration-sweep numbers (1/20 @ 5 dB, 86.7–95.0% partition,
  crossover budgets 5000/3567/3000/1000, Belle power curve 2000/14326, 49.9%/2.51%) — all match
  their cited `alpha015_/speaker_partition_/mandarin_calsweep_/calsize_power_` results.json. ✓
- Whisper THCHS-30 audit cell (`tab:holm`, `tab:cert`) confirmed run on the **corrected**
  n=2,495 file (`thchs30_whisper_scored_fixed_2026-07-12.jsonl`, 2495 lines verified). ✓

**Claims read for over-assertion (dimension 2):** ~40 declarative sentences in Method, Results,
Discussion, Limitations, and all captions. The paper is unusually disciplined — most causal/
scope language is explicitly hedged and tied to a measurement (e.g. the 0/20-vs-δ paragraph, the
exchangeability-candor paragraph, the verdict-symmetric caveats). Six residual over-reaches below.

---

## CRITICAL

### C1. "twelve … certified cells across two backbones × three corpora" — arithmetic self-contradiction (both papers)

- `paper.tex:926-927`; `taslp/paper_ieeetran.tex:880-881`.
- Text: *"The frontier now spans **twelve** non-vacuous, 0/20-violation certified cells across
  **two backbones × three corpora**, not the single α=2%/Paraformer/Aishell cell."*
- Two backbones × three corpora = **6**, not twelve. `tab:landscape` confirms exactly **six**
  non-vacuous certified rows (Paraformer×{Aishell,THCHS,MagicData} + Belle×{Aishell,THCHS,
  MagicData}); the three zipformer rows are `vac.`. The Limitations sentence
  (`paper.tex:1014-1016` / `taslp:966-968`) independently describes the same set as "Paraformer
  and Belle both certify … on THCHS-30 and MagicData" (4 transfer + 2 in-corpus = 6),
  corroborating six.
- Almost certainly a carry-over of the audit family size m=12 (`tab:holm`) into the *certified*
  count. The number "twelve" is contradicted by its own "two × three" in the same clause and by
  the table it summarizes.
- **Fix:** change "twelve" → "six" in both files. (This is the headline M1-resolution sentence,
  so the wrong count directly overstates the central generalization claim.)

### C2. THCHS-30 evaluated on TWO different pools; the landscape's THCHS cells use the pool the paper itself condemns as contaminated (both papers)

This is one issue with three coupled defects (number inconsistency + integrity-claim contradiction).

**(a) Direct number contradiction — same "3-backbone THCHS" role listed as two different n.**
- `tab:data` row (`paper.tex:351` / `taslp:323`): THCHS-30 test, role *"cross-corpus transfer
  (**3 backbones**)"*, Utterances = **2,495**, note *"full official test (D-prefix spk)"*.
- `tab:landscape` caption (`paper.tex:898` / `taslp:854`): *"THCHS-30 (**n=1,339**)"* — the same
  three landscape backbones.
- Verified against source: `landscape/decode_thchs30_{paraformer,belle,zipformer}_test.jsonl`
  are **1,339 lines each**, and `landscape/audit/audit_thchs30_*.json` all carry `n=1339`; the
  landscape-table CER values (3.75 / 6.62 / 81.80%) reproduce **only** from n=1,339. So the
  3-backbone landscape ran on 1,339, while `tab:data` asserts 2,495 for that exact role. The two
  tables disagree on the size of the same evaluation set.

**(b) Integrity: the landscape THCHS cells are the exact wrong-pool the paper says it discarded.**
- The setup (`paper.tex:312-324` / `taslp:287-298`) and `tab:deviations` (`paper.tex:379-380` /
  `taslp:348-349`) state the n=1,339 THCHS decode was a *"mislabeled mirror split … 82.7%
  train-speaker audio (1,107 of 1,339), only 232 genuine test utts,"* i.e. contaminated, and that
  it was *"corrected to the official test"* (n=2,495). The Whisper THCHS audit cell was indeed
  moved to n=2,495. **But the landscape's THCHS cells for all three backbones were not** — they
  remain on the 1,339 contaminated mirror pool. Those cells feed the headline claims that the
  certificate "breaks the single-cell reading" and that *"Paraformer and Belle both certify
  non-vacuously as transfer cells on THCHS-30"* (`paper.tex:1014-1016` / `taslp:966-968`).
- Consequence: the THCHS legs of the landscape (2 of the 6 non-vacuous certified cells, plus the
  3 vacuous zipformer THCHS rows) rest on a pool that is 82.7% train-speaker audio by the paper's
  own accounting — so the THCHS full-set CERs (Para 3.75%, Belle 6.62%, zip 81.80%) and their
  certified frontiers are **not** the official-test numbers a reader is led to expect by `tab:data`.

**(c) The integrity claim is contradicted for these cells.**
- `paper.tex:323-324` / `taslp:297-298`: *"We disclose the wrong-pool contamination and report
  the **corrected** test-split number rather than silently swapping it."* This holds for the
  Whisper cell but is false for the landscape THCHS cells, which still carry the uncorrected
  number. The landscape caption discloses "n=1,339" but does **not** flag it as the contaminated
  mirror pool, does not explain why it differs from the 2,495 used for Whisper/`tab:data`, and
  does not carry the wrong-pool caveat.
- **Scope note (to avoid overclaiming):** the core *contamination rebuttal* (M2) rests on Belle's
  **Aishell-1** certificate (α=5%, 93.3%, 4.20%, on n=7,176 — verified), not on THCHS, so that
  argument is not undermined. The damage is confined to the THCHS transfer cells and to `tab:data`'s
  size claim.
- **Fix options:** either (i) re-decode/re-audit the THCHS landscape cells on the corrected
  n=2,495 official test and update `tab:landscape` + text, or (ii) if the 1,339-pool landscape is
  kept, correct `tab:data` to n=1,339 for the 3-backbone role, relabel the landscape THCHS cells as
  the contaminated-mirror pool with the same caveat applied elsewhere, and soften the "corrected …
  rather than silently swapping" claim so it does not imply the landscape was corrected too.
  Option (i) is strongly preferred for integrity.
- (Informational, not a manuscript defect: `numbers.json holm_realized._fix_note` still describes
  the old THCHS decode as *"tail-truncated"*, the mechanism the paper deliberately corrected to
  "wrong pool" — a stale description inside a frozen record, worth reconciling for provenance.)

---

## MINOR

### M1. "roughly halved" / "halved error at 91% coverage" overstates the reduction
- `paper.tex:403-404` (*"roughly halved error rate on the accepted ~91% of traffic"*) and the
  Discussion `paper.tex:977` / `taslp:931` (*"halved error at 91% coverage"*); `taslp:372`.
- Mean accepted macro-CER 1.21% vs full-set 1.98% is a **~39% reduction**, not 50%. (Only the
  minimum-reseed 1.00% approaches "halved.") "Roughly" hedges the Results instance, but the
  Discussion states "halved" flatly.
- **Reword:** "cuts the error rate by roughly 40% (1.98% → 1.21% mean) on the accepted ~91%."

### M2. "where the weak backbone leaves more headroom to exploit" — mechanism asserted as fact
- `paper.tex:690` / `taslp:650-651`, explaining the 0.0513 Whisper-clean excess-AURC.
- The excess-AURC magnitude is measured; the *reason* ("weak backbone leaves more headroom") is
  an unmeasured causal attribution stated parenthetically as fact.
- **Reword:** "(Whisper Aishell clean s1, the largest in the family)" or "…consistent with a
  weaker backbone having more separable error mass," flagged as interpretation.

### M3. "the distribution shift the domain fingerprint is meant to flag is largest" at 5 dB — asserted, not shown
- `paper.tex:564-566` / `taslp:527-529`.
- The KS domain-fingerprint distances are reported for the English test-other cells (0.176/0.280/
  0.247) but **not** for the 5/15/25 dB noise conditions, so "largest [shift] at 5 dB" is a
  plausible inference (most noise) stated as a measured fact.
- **Reword:** "…begins to strain only at the most severe 5 dB condition (the largest nominal SNR
  drop)," or report the noise-cell KS distances.

### M4. "matching the 1/√n_cal bound slack" — quantitative theory-match asserted from 4 points
- `paper.tex:467` / `taslp:435`.
- Four crossover budgets (α 1.5/1.7/1.9/5% → 5000/3567/3000/1000) are shown to be monotone; the
  claim that they *match* the 1/√n_cal rate is a stronger quantitative statement not demonstrated
  (no fit/residual). Monotone ≠ matches a specific power law.
- **Reword:** "…monotone in the target, qualitatively consistent with the 1/√n_cal bound slack."

### M5. "few papers actually test [this precondition]" — unsupported generalization about the literature
- `paper.tex:981-982` / `taslp:935`.
- A claim about the field's practice with no citation or survey behind it.
- **Reword:** drop "few papers actually test," or cite the point (Traub et al. is already in the
  bibliography for the selective-classification debunking lens).

### M6. "difficulty gradient the per-stratum design anticipates" — mild causal attribution
- `paper.tex:524-529` / `taslp:491-496`.
- dur2 (long) having higher accepted-set CER is observed; labeling duration as the *cause*
  ("difficulty gradient") is a light interpretation. Well hedged ("mild, expected") and low-stakes,
  but it attributes a mechanism to a correlation. Acceptable as-is; noted for completeness.

---

## Dimensions reported clean

- **Cross-paper (paper.tex ↔ taslp) numeric parity:** every shared number is identical; the port
  changed only class/format machinery and prose compression, not values. The single genuine
  divergence is a `% source` *comment* (paper.tex:358 cites the scored file's line count for
  THCHS n=2,495; taslp:330 cites `THCHS-REDECODE-DELTA.json n_new`) — a provenance-comment
  mismatch, not a body/table number, and immaterial to the rendered PDF.
- **Abstract ↔ body ↔ table traceability:** all abstract numbers (0/20, 85.7–96.2%, 1.00–1.53%,
  1.98%, α=1.9%, α=5% Belle 0/20, 0.012–0.051 excess-AURC) trace to a body occurrence and a source
  JSON. No abstract number is orphaned or contradicted.
- **Rounding:** consistent throughout (28.06→28.1%, 9.93/9.95% micro correction documented,
  0.01368 vs 0.01370 s2 distinctness footnoted). No rounding errors found.

*End of audit. No manuscript files were modified.*
