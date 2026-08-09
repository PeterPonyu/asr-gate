# Belle-whisper (B3') calibration-size power curve — Aishell-1 (0 GPU)

**Date:** 2026-07-15 · **Runner:** `run_belle_calsweep.py` · **Output:** `results.json`

## What this answers

The pulled landscape (`FREEZE-AMENDMENT-2026-07-13`, `landscape_pulled_2026-07-15/`)
certifies **Belle-whisper-large-v3-zh (B3′)** on the official Aishell-1 test only at
**α∈{5,10}%** (vacuous at α∈{1.5,2,3}%) at the frozen calibration budget
(`n_cal≈3,567` after the duration-tercile Mondrian fit/conformalize split, 20
speaker-level reseeds). Belle is the paper's **contamination-independence cell**: it is
the *only* fully-independent backbone (different architecture, org, and documented
test-excluded training data from Paraformer) that certifies non-vacuously on Aishell-1
test at all, so how robust that certificate is to calibration budget matters for the M2
answer. The English (`english_calsweep_2026-07-13`) and Mandarin/Paraformer
(`mandarin_calsweep_2026-07-13`) arms both showed sub-target vacuity is a
**calibration-budget** artifact, not an intrinsic bound property. This asks the same
question for Belle, reported as a **power curve** — fraction of `R=20` reseeds
certifying non-vacuously per `(n_cal, α)` cell — rather than a single-seed crossover.

## Method (post-hoc, disclosed) — mirrors `mandarin_calsweep_2026-07-13` exactly

- **EVAL (fixed):** `landscape_pulled_2026-07-15/backbone_belle/aishell_test_scored.jsonl`
  — the pulled, byte-verified frozen box artifact, official Aishell-1 test, **n=7,176**
  s1/CER-bearing utts, never touched by calibration.
- **CAL POOL:** `landscape_pulled_2026-07-15/backbone_belle/dev_canonical.jsonl`
  (Aishell-1 dev, 14,329 utts / **14,326** ref-bearing), scored **once** with the frozen
  pipeline (`score_table` + `compute_cer_batch`, `numeral_policy=keep`) — the same
  pipeline that produced `aishell_test_scored.jsonl`. Speaker-disjoint from test by
  construction.
- **Consistency guard (passes exactly):** re-scoring `aishell_test_scored_canonical.jsonl`
  with this runner's pipeline reproduces the frozen `aishell_test_scored.jsonl` s1/CER to
  `max|Δs1| = 0.0`, `max|ΔCER| = 0.0` over 7,176 utts.
- **SWEEP:** grid `n_cal ∈ {250, 500, 1000, 1250, 1500, 1750, 2000, 3000, 4000, 6000,
  8000, 10000, 12000, 14326(full)}` — the requested `{250,500,1k,2k,4k,full}` anchors
  plus interpolation points added post-hoc (disclosed) to pin the α=5% crossover
  (which fell strictly between 1000 and 2000 on the anchor grid alone) and trace the
  α=3% frontier between 4000 and the full pool. For each `n_cal`, **R=20 reseeds**
  (seeded utterance shuffles 0–19 of the dev pool, truncated to `n_cal`), certified with
  the **exact frozen G1 machinery** — `asr_gate.ltt.ltt_certify`, EB p-value,
  Bonferroni-over-grid, `δ=0.1`, `n_grid=200`, `min_accept_frac=0.1`, `g1-score=s1`,
  `loss=CER` — identical config to `english_calsweep_2026-07-13` /
  `mandarin_calsweep_2026-07-13`'s PRIMARY view. **α-grid** = the frozen landscape grid
  `{0.015, 0.02, 0.03, 0.05, 0.10}`. No bound machinery changed.
- These 20 budget-subsample reseeds are **distinct** from the frozen landscape's 20
  speaker-level calibration-carve reseeds (which resample the cal/tune split at a fixed
  budget); here the budget itself is the swept variable.

**Base rate:** Belle's full-set Aishell-1 test macro-CER is **5.12%**. Targets below
this (α ≤ 5%) genuinely bind (accept-all is inadmissible); α=10% does not bind.

## Headline — power-curve crossovers (min `n_cal` with cert. non-vacuous in ≥90% of 20 reseeds)

| α | binds (α < 5.12%)? | min `n_cal` for ≥90% of 20 reseeds | at that `n_cal`: mean eval accept / accepted macro-CER |
|---|---|---|---|
| 1.5% | yes | **never reached** (0/20 at every tested `n_cal`, up to and incl. the full 14,326-utt pool) | — |
| 2.0% | yes | **never reached** (0/20 at every tested `n_cal`, incl. full pool) | — |
| 3.0% | yes | **14,326 (full pool only)** — 15% (3/20) at `n_cal=12,000`, jumping to 100% (20/20) only at the full pool | 49.9% @ **2.51%** |
| **5.0%** | **yes (binding)** | **2,000** | **80.8% @ 3.38%** |
| 10.0% | no | 500 | 99.4% @ 4.96% |

**Belle-whisper certifies α=5% in ≥90% of reseeds at `n_cal=2,000`** (65% of reseeds
already certify at `n_cal=1,750`; 0/20 at `n_cal=1,500` and below). Zero eval-side
violations (accepted macro-CER exceeding α) occur in **any** certifying reseed at
**any** `(n_cal, α)` cell — every certificate reported here is genuinely binding.

## The headline nuance: α=5% is a budget effect, but α≤2% is not (for Belle)

Unlike the Paraformer/English arms — where more calibration budget kept pushing the
certifiable frontier tighter, all the way past the base rate — **Belle's frontier stalls
around α=3% even at the full 14,326-utt dev pool, and never reaches α≤2% at any budget
tested**:

- α=5% (above the 5.12% base rate... no, 5% < 5.12%, so it also binds) tightens
  smoothly and saturates fast: 0/20 at `n_cal≤1,500`, 65% at 1,750, 100% by 2,000, and
  stays 100% out to the full pool (accepted-set macro-CER *rises* toward the base rate
  as `n_cal` grows — from 3.38% at 2,000 to 4.95% at the full pool — because a larger
  calibration sample lets the bound certify a more permissive, higher-acceptance
  threshold at the same α).
- α=3% is reachable but **budget-hungry and fragile**: 0/20 through `n_cal=10,000`,
  only 15% (3/20) at `n_cal=12,000`, and 100% (20/20) only at the full 14,326-utt pool
  — accepting just 49.9% of the fixed test at 2.51% accepted-set macro-CER. This is
  the *entire* available Aishell-1 dev split; there is no more calibration budget to
  test with this data.
- **α=2% and α=1.5% never certify non-vacuously in any of the 20 reseeds at any tested
  `n_cal`, including the full dev pool.** This is a genuine contrast with the
  English/Mandarin arms (where the analogous budget sweep kept unlocking tighter
  targets): for Belle, the LTT bound's calibration-budget slack is not the only
  binding constraint at these tight targets — `s1`'s separation power on this backbone
  (full-set CER 5.12%, versus Paraformer's 1.98%) appears to run out before the bound
  does. This is consistent with, and adds evidence for, the Limitations section's
  existing statement that the sub-3% regime is Paraformer-only.

## Takeaway for the manuscript

Belle's reported α≥5% floor (the frozen-budget landscape result) **is** a
calibration-budget effect and is cheap to fix: **`n_cal=2,000`** (about 4× smaller than
the frozen ~3,567-utt gate-calibration budget already in use) reaches ≥90%-of-reseeds
non-vacuous certification at α=5%, with **zero violations** anywhere on the grid. But
unlike the Paraformer/English arms, throwing the *entire* available dev pool (14,326
utts, ~4× the frozen budget) at Belle only marginally reaches α=3% (100% of reseeds,
but a low 49.9% acceptance) and **never** reaches α≤2% — so Belle's contamination-answer
certificate is budget-robust at its reported α=5% operating point, but its ceiling on
tighter targets looks like a backbone/score-quality limit, not a calibration-budget one.

## zipformer (B4) — not swept

Per the pulled landscape digest (`landscape_pulled_2026-07-15/LANDSCAPE-DIGEST.md`),
zipformer-multi-zh exposes **unusable posteriors** on Aishell-1: full-set CER **45.88%**
and `s1` excess-AURC **negative** (−0.0158, *worse than random abstention*), vacuous at
every `(corpus, α)` cell in the frozen landscape. More calibration budget cannot fix an
uninformative/anti-informative score — the LTT bound only tightens around whatever risk
the *accepted set* actually has, and a negative-excess-AURC score does not separate
correct from incorrect hypotheses at all. Skipped per instructions, not computed.

## Reproduce

```
cd calsize_power_2026-07-15 && python3 run_belle_calsweep.py
```

Deterministic (fixed seeds 0–19 per `n_cal`); re-running reproduces `results.json`
exactly (verified: identical console summary across two runs during this analysis).
