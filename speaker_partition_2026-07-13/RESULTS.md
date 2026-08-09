# Speaker-partition validation (PART B / red-team M2 repair)

**Date:** 2026-07-13 · **Runner:** `run_speaker_partition.py` · **Output:** `results.json`

## What this answers

The frozen main run's 20 reseeds (`main_results_2026-07-09/reseed_*`) resample
only the **dev cal/tune carve** and apply every gate to the **same fixed
7,176-utt / 20-speaker official test set**. That check exercises
*calibration-draw* randomness but holds the **eval speaker population fixed** —
it is structurally blind to speaker-level variability in the evaluated
population (red-team **M2**, reject-capable).

This adds the missing axis: it repartitions the **full Aishell-1 speaker set**
(dev 40 speakers + official test 20 speakers = **60 speakers**, all scored) into
a calibration cohort and a **disjoint** eval cohort, R=20 times, reruns
calibrate+apply per partition, and asks whether the α=2% certificate still holds
when the **eval speakers themselves change**.

## Method (post-hoc, disclosed)

Frozen config throughout: α=2%, δ=0.1, `strata=duration_tercile`,
`g1-score=s1`, LTT/EB/Bonferroni, `numeral_policy=keep`, Paraformer-zh (B2).
Pool = `dev_canonical` (40 spk) ∪ `test_canonical` (20 spk), scored once with
the frozen `score_table`+`compute_cer_batch` pipeline. Per seed r∈{0..19}:
shuffle the 60 speakers → first 20 = **cal cohort** (`calibrate_gate` auto-splits
by speaker into fit ≈10 spk + conformalize ≈10 spk → n_cal ≈ 3,567, the frozen
scale), next 20 = **eval cohort** (≈7,000 utts, the frozen test scale),
speaker-disjoint. Apply the gate to the eval cohort, join ACCEPT decisions back
to true per-utterance CER, flag a **violation** iff accepted-set macro-CER > α.
This necessarily calibrates on partitions of the official test split, so it is a
**validity check of the speaker-exchangeability axis, not a headline result**;
the main-run numbers still calibrate on dev only.

## Result — the M2 attack is answered

| Quantity | Value |
|---|---|
| Certified partitions | **20 / 20** (0 vacuous) |
| **Violations** (accepted macro-CER > α=2%) | **0 / 20** |
| Acceptance range | **86.7% – 95.0%** |
| Accepted macro-CER range | **1.00% – 1.38%** |
| Accepted micro-CER range | 1.01% – 1.40% |
| n_cal range | 3,537 – 3,622 |
| eval n range | 7,110 – 7,226 |

Across 20 partitions with **genuinely different eval speakers**, the α=2%
certificate holds every time (**0/20 violations**), at acceptance and
accepted-CER ranges essentially indistinguishable from the frozen
calibration-reseed check (85.7–96.2% acceptance, accepted-CER 1.00–1.53%). The
speaker-exchangeability axis, which the reseed check could not probe, does not
break the guarantee.

## Takeaway for the manuscript

Two distinct validity checks, now separated in the prose:
- the **reseed check** (frozen) = *calibration-randomness* only (resamples the
  dev cal/tune carve against a fixed test set);
- the **speaker-partition check** (this, post-hoc, disclosed) = the
  *speaker-exchangeability* axis (repartitions cal vs eval speakers), 0/20
  violations at α=2%.
