# English-arm calibration-size sweep (PART A / red-team F1 kill)

**Date:** 2026-07-13 · **Runner:** `run_calsweep.py` · **Output:** `results.json`

## What this answers

The frozen English arm (`english_arm_fixed_2026-07-12`,
`english_arm2_2026-07-13`) certifies non-vacuously **only at α=5%** and is
**vacuous at α∈{1,2,3}%** on all three backbones, on a **613-utterance
calibration carve** (the conformalize/cal pool size, `n_cal=613`). The paper's
original sentence framed that sub-5% vacuity as *"a property of the conservative
distribution-free bound … not an artifact of a weak backbone"* — reading as an
**intrinsic** property of the bound. Red-team finding **F1** falsified that
framing: the LTT empirical-Bernstein bound slack shrinks with `n_cal`, so the
**same test-clean data certifies tighter targets given more calibration budget**.
This is a *calibration-budget* property, not an intrinsic-bound one.

## Method (post-hoc, disclosed)

Exact frozen G1 machinery — `asr_gate.ltt.ltt_certify`, EB p-value,
Bonferroni-over-grid, `δ=0.1`, `n_grid=200`, `min_accept_frac=0.1`,
`g1-score=s1`, loss=CER — re-run over `n_cal ∈ {613, 1000, 1500, 2000}` at
`α ∈ {0.02, 0.03, 0.05}`, per backbone. To isolate the calibration-budget axis,
the **held-out eval set is fixed** (last speakers in a seeded permutation, ≥500
utts, speaker-disjoint from every calibration carve, mirroring the frozen
speaker-disjoint cal/eval protocol) and only the calibration size is varied by
subsampling the complementary speaker pool to exactly `n_cal`. No bound
machinery changed. `n_cal=613` reproduces the frozen carve size as the anchor.
Per backbone: 2,620 test-clean utts / 40 speakers → eval n=581 (fixed),
cal candidate pool = 2,039 utts.

## Headline result — smallest `n_cal` that certifies non-vacuously (seed 0)

| Backbone | α=2% | α=3% | α=5% |
|---|---|---|---|
| Whisper large-v3 | — (>2000) | **1500** | 613 |
| wav2vec2 base | — (>2000) | **1500** | 613 |
| wav2vec2 large (LV60k) | **1500** | **1000** | 613 |

- **Anchor reproduces the frozen result:** at `n_cal=613` every backbone is
  vacuous at α=2% and α=3% and certifies only at α=5% — matching the frozen
  `ATTAINMENT.json` / `ATTAINMENT2.json`.
- **α=3% becomes certifiable with more calibration data:** at `n_cal≈1500` all
  three backbones certify α=3% non-vacuously; the strongest backbone
  (wav2vec2-large) already certifies α=3% at `n_cal=1000`.
- **α=2% is reachable but backbone-dependent:** only the strongest backbone
  (wav2vec2-large) certifies α=2% within the available test-clean pool, at
  `n_cal=1500`, accepting 96.7% of eval-clean at accepted-set macro-CER 0.42%
  (a genuinely *binding* certificate — target 2%, achieved 0.42%). Whisper and
  wav2vec2-base do not reach α=2% even at `n_cal=2000`, the ceiling of the
  test-clean calibration pool.

## Seed robustness (seeds 0–4)

Crossover `n_cal` is stable across 5 re-draws of the eval/cal partition:

| Backbone | α=2% | α=3% |
|---|---|---|
| Whisper | >2000 (all 5) | 1500 (all 5) |
| wav2vec2 base | >2000 (all 5) | 1500 (all 5) |
| wav2vec2 large | 1500 (all 5) | 1000 (all 5) |

(α=5% certifies by `n_cal=613` in every seed except one Whisper draw at 1000.)

## Takeaway for the manuscript

The sub-5% vacuity at `n_cal=613` is a **calibration-budget** phenomenon, not an
intrinsic property of the distribution-free bound. Reframed as practitioner
guidance ("what calibration data buys"): with ≈1,500 speaker-disjoint
calibration utterances the same LibriSpeech data certifies α=3% on all three
strong English backbones, and the strongest backbone certifies α=2%. The
original intrinsic-bound sentence is **withdrawn** and replaced with this
budget-framed statement.
