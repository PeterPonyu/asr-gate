# Mandarin calibration-pool sweep (the free odds-mover, 0 GPU)

**Date:** 2026-07-13 · **Runner:** `run_mandarin_calsweep.py` · **Output:** `results.json`

## What this answers

The frozen Mandarin main run certifies **α=2%** on a dev calibration carve of
`n_cal≈3,567`, but `alpha015_2026-07-13` showed the binding sub-base-rate target
**α=1.5% is 0/20 vacuous** at that same frozen budget. The paper's own English
calsweep (`english_calsweep_2026-07-13`) established that such sub-target vacuity
is a **calibration-BUDGET** artifact (LTT/EB bound slack ~ 1/√n_cal), not an
intrinsic property of the distribution-free bound. This runner asks the Mandarin
analogue: **does a larger speaker-disjoint dev calibration pool push α=1.5% into
a non-vacuous certificate on the FIXED official Aishell-1 test set?** The answer
is **yes** — and it tightens the certified frontier for ~0 compute.

## Method (post-hoc, disclosed) — the legitimate frozen protocol

Unlike the English sweep (which drew cal and eval from a single split), this uses
the **actual frozen Mandarin protocol** — calibrate on **dev**, certify on the
**fixed official test** — so there is **no test-partition calibration** caveat:

- **EVAL (fixed):** official Aishell-1 test, `main_results_2026-07-09/test_scored.jsonl`
  — the **frozen** artifact, 7,172 s1-bearing utts / 20 speakers, never touched by
  calibration; speaker-disjoint from dev by construction.
- **CAL POOL:** Aishell-1 dev, `pilot_results_2026-07-09/dev_canonical.jsonl`,
  **14,325** ref-bearing s1-bearing utts / 40 speakers, scored **once** with the
  frozen pipeline (`score_table` + `compute_cer_batch`, `numeral_policy=keep`).
- **SWEEP:** for each `n_cal ∈ {613,1000,1500,2000,3000,3567,5000,7000,10000,14000}`
  take the first `n_cal` utts of a seeded shuffle of the dev pool and certify.

**Two certification views**, both reported:
1. **PRIMARY** — flat `asr_gate.ltt.ltt_certify` (EB p-value, Bonferroni-over-grid,
   δ=0.1, `n_grid=200`, `min_accept_frac=0.1`, score=s1, loss=CER) — mirrors
   `english_calsweep` **exactly**, isolating the pure budget axis (`n_cal` = the
   number of calibration examples the bound sees).
2. **CONFIRMATION** — the frozen **Mandarin certificate machinery**
   (`gate.calibrate_gate`, `strata=[duration_tercile]`, `fit_frac=0.5`, the config
   the main run / `alpha015` / `speaker_partition` use) on the **full dev pool**,
   applied to the fixed test — shows the effect survives under the *actual reported
   certificate* (Mondrian + fit/conformalize split), not only the flat bound.

**Consistency guard (passes exactly).** Re-scoring `test_canonical.jsonl` with this
runner's pipeline reproduces the frozen `test_scored.jsonl` s1/CER to
`max|Δs1| = 0.0`, `max|ΔCER| = 0.0` over 7,172 utts — proving the swept cal-pool
scores are on the identical frozen scale as the eval scores.

Base-rate note: the full-set macro-CER over the 7,172 s1-bearing eval utts is
**1.95%** (the 4 utts with degraded token-logp / null s1 cannot be thresholded and
are excluded); the all-7,176-utt figure is 1.98% (`numbers.json`). Both α=1.5% and
α=1.9% still **bind** against 1.95%.

## Headline — the pool → α frontier (smallest `n_cal` certifying non-vacuously, seed 0)

| α | binds (α<1.95%)? | crossover `n_cal` | at crossover: eval acceptance / accepted macro-CER |
|---|---|---|---|
| **1.5%** | **yes** | **5,000** | 83.2% @ **0.91%** |
| 1.6% | yes | 5,000 | — |
| 1.7% | yes | 3,567 | — |
| 1.8% | yes | 3,000 | — |
| 1.9% | yes | 3,000 | — |
| 2.0% | no | 3,000 | 87.9% @ 1.06% |
| 3.0% | no | 1,500 | — |
| 5.0% | no | 1,000 | — |

**The frontier is monotone and binding:** tighter targets need more calibration
budget, exactly the 1/√n_cal prediction. The binding target **α=1.5%** is vacuous
through `n_cal=3,567` (reproducing `alpha015`'s frozen-budget verdict) and turns
**non-vacuous at `n_cal=5,000`**, accepting **83.2%** of the fixed test at **0.91%**
accepted macro-CER (a genuinely *binding* certificate — target 1.5%, achieved
0.91%, **0 violations**). By the full dev pool (`n_cal≈14k`) α=1.5% accepts 93.0%
at 1.32%.

## Seed robustness (seeds 0–4)

Crossover `n_cal` is stable across 5 re-draws of the dev-pool shuffle:

| α | crossover `n_cal` across seeds 0–4 |
|---|---|
| 1.5% | 5000, 5000, 5000, 5000, 7000 |
| 1.7% | 3567, 5000, 3567, 5000, 5000 |
| 1.9% | 3000, 3000, 3000, 3000, 3000 |
| 2.0% | 3000 (all 5) |
| 5.0% | 1000, 1000, 1000, 613, 1000 |

α=1.5% certifies by `n_cal=5,000` in 4 of 5 seeds (7,000 in the fifth) — not a
single-draw artifact.

## Confirmation under the frozen certificate machinery (full dev pool, duration-tercile Mondrian)

`calibrate_gate` (strata=`duration_tercile`, `fit_frac=0.5`, the exact main-run
config) on the full dev pool (realized `n_cal≈7,184` after the speaker-aware
fit/conformalize split), applied to the fixed test:

| α | certified | eval acceptance | accepted macro-CER | violation |
|---|---|---|---|---|
| **1.5%** | **yes** | 90.6% | **1.20%** | **no** |
| 1.7% | yes | 94.1% | 1.39% | no |
| 1.9% | yes | 97.6% | 1.64% | no |
| 2.0% | yes | 98.7% | 1.75% | no |

The budget effect **survives under the actual reported certificate**: with the
full dev pool, the duration-tercile-Mondrian α=1.5% certificate certifies at
90.6% acceptance / 1.20% accepted macro-CER, **0 violations** — versus `alpha015`'s
frozen ~3,567-carve result of **0/20 certified** at α=1.5%.

## Takeaway for the manuscript

The sub-2% vacuity on the Mandarin arm is a **calibration-budget** phenomenon,
not an intrinsic property of the bound — the same conclusion the English sweep
reached, now on Paraformer/Aishell-1 and against the **fixed official test set**.
Reframed as practitioner guidance: **with ≈5,000 speaker-disjoint dev calibration
utterances, the certified frontier tightens from α=2% to a binding α=1.5%** on the
same test data, and under the full frozen certificate the α=1.5% certificate
accepts 90.6% at 1.20% accepted macro-CER. This directly answers red-team **M1**
(the "one α, one cell" attack) on the *budget* axis — the certified operating band
below the base rate is real, mapped, and reproducible for **0 GPU-hours** — and
composes with the backbone/corpus landscape (FREEZE-AMENDMENT-2026-07-13) that
answers M1/M2 on the *scope* axis.
