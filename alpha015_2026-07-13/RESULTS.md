# α=1.5% binding-regime point (PART C / red-team M1 repair)

**Date:** 2026-07-13 · **Runner:** `run_alpha015.py` · **Output:** `results.json`

## What this answers

Red-team **M1**: on clean Aishell-1 the full-set Paraformer macro-CER is
**1.98%**, *below* the 2% target, so "0/20 violations at α=2%" can be read as
"accept-all trivially meets a budget that already exceeds the error" (grid-floor
saturation). This maps the **binding regime** — targets *below* the 1.98% base
rate, where accept-all cannot satisfy the budget and the certificate must do
real work — by replicating the exact frozen main protocol (each reseed's dev cal
carve → apply to the fixed official test set) across all 20 reseeds, changing
only α.

## Base-rate disclosure (the M1 framing)

Full-set clean macro-CER = **1.98%** < the 2% target. So on clean data accept-all
already meets α=2%; the certificate's *information* is **(i)** the conditional
accepted-set quality (at α=2% it certifies a **restricted** 85.7–96.2% set at
accepted-CER 1.00–1.53%, i.e. it does **not** accept all — it withholds the
low-confidence tail) and **(ii)** the binding regimes below the base rate,
mapped here.

## Binding-band sweep (20 reseeds, frozen n_cal≈3567)

| α | binds (α<1.98%)? | certified | mean acceptance | mean accepted macro-CER |
|---|---|---|---|---|
| **1.5%** | yes | **0/20** (vacuous) | — | — |
| 1.6% | yes | 4/20 | 79.4% | 0.85% |
| 1.7% | yes | 8/20 | 84.1% | 0.95% |
| 1.8% | yes | 16/20 | 85.8% | 1.01% |
| **1.9%** | **yes** | **20/20** | **87.8%** | **1.09%** |
| 2.0% | no | 20/20 | 90.7% | 1.21% |

**Zero violations across the entire band** (accepted macro-CER 0.85–1.21%, always
well under the respective α).

## The two headline points for M1

1. **α=1.5% is honestly vacuous (0/20).** Below the base rate, at the frozen
   calibration budget the certificate *refuses* to certify — a verdict-symmetric,
   honest failure, not a false positive. (PART A shows this vacuity is a
   calibration-*budget* property: larger n_cal would certify tighter targets.)
2. **α=1.9% is the binding point that certifies (20/20).** It sits *below* the
   1.98% base rate, so accept-all would **fail** the budget, yet the certificate
   certifies a restricted **87.8%**-acceptance set at **1.09%** accepted macro-CER.
   This is the certificate doing genuine, non-trivial work — directly rebutting
   the "accept-almost-everything trivially satisfies" reading of M1.

## Additional disclosure to add (verified)

- **δ=0.1 violation budget.** The frozen config is δ=0.1 (verified in every
  reseed `gate.json`). The 5 dB noise cell's **1/20** violation is therefore
  *within* the δ=0.1 violation budget (1/20 = 0.05 ≤ 0.1) — the guarantee permits
  up to a δ fraction of failing calibration draws, so 1/20 is consistent with the
  certificate holding, not a breach. State this explicitly where the 5 dB result
  is discussed.
