# G2 — per-duration-tercile Mondrian result (CPU-only, cached artifacts) — 2026-07-11

**Verdict: STRENGTHENS.** The G1 selective-risk certificate holds *within every duration
tercile* — 0/20 violations in all three terciles across all 20 reseeds, with every tercile's
speaker-blocked bootstrap CI comfortably under the α = 2 % target. There is a mild, expected
long-utterance difficulty gradient (disclosed below), but it stays inside the certified budget
and is exactly what the Mondrian per-tercile thresholds are there to absorb.

Source: `compute_mondrian_tercile.py` → `results/mondrian_tercile_analysis.json`. Inputs
(read-only): `main_results_2026-07-09/reseed_{0..19}/{applied_test.json, gate.json}` +
`test_scored.jsonl`. Backbone Paraformer, Aishell-1 official test, α = 0.02, δ = 0.1. CIs:
`relmetrics.blocked_bootstrap`, block = speaker, 2 000 resamples, seed 0, percentile — the same
methodology as the paper's existing CIs.

## Table (α = 2 %, 20 speaker-level reseeds)

| Duration tercile | Acceptance % (mean [min, max]) | Accepted-set macro-CER % (mean [min, max]) | Violations | Reseed-0 speaker-blocked 95 % CI on accepted-set macro-CER |
|---|---|---|---|---|
| dur0 / short  | 91.8 [86.8, 96.3] | 1.12 [0.92, 1.40] | **0/20** | [0.77, 1.09] %  (n = 1 757, 20 blocks) |
| dur1 / medium | 91.4 [87.0, 96.2] | 1.06 [0.86, 1.33] | **0/20** | [0.72, 0.97] %  (n = 1 870, 20 blocks) |
| dur2 / long   | 89.8 [84.1, 96.2] | 1.36 [1.16, 1.75] | **0/20** | [1.01, 1.32] %  (n = 2 524, 20 blocks) |

**Cross-check (validates the computation against the frozen paper number):** the reference
reseed-0 *overall* accepted-set macro-CER recomputed here is **1.0005 %** (CI [0.90, 1.11]),
identical to the frozen `attainment_table.json` reseed-0 value (0.01000455) and the paper's
headline 1.00 %; the three tercile accepted counts sum to 6 151 = the overall reseed-0 accept
count. Duration-tercile edges are frozen on each reseed's calibration split and applied to test,
so test tercile sizes are unequal and reseed-dependent (correct Mondrian behavior), which is why
the long tercile carries more test utterances (~3 000) than short/medium.

## Interpretation (one paragraph)

The marginal certificate the paper reports is not hiding a stratum where it fails: conditioning
on duration — the one input-only axis available at gate time — the accepted-set macro-CER stays
under the 2 % target in **every** tercile and **every** reseed (0/20 throughout), and the
speaker-blocked CIs place even the worst tercile's upper bound at 1.32 %, well below target. The
only heterogeneity is a mild and unsurprising gradient: long utterances (dur2) are harder, so
their accepted-set CER runs ~0.3 pp higher (1.36 % vs 1.06–1.12 %) and their acceptance is ~2 pp
lower (89.8 % vs 91.4–91.8 %). This is precisely what the deployed Mondrian thresholds are
designed to handle — the gate already sets a *tighter* score threshold for the long tercile
(reseed-0 gate.json: dur2 = 0.111 vs dur0 = 0.150), trading a little coverage on long utterances
to keep their error rate certified. So the result reads as **conditional-coverage evidence that
strengthens the paper**, with the long-utterance gradient disclosed honestly rather than a
finding that per-stratum control breaks.

## Honest note: strengthens vs heterogeneity-to-disclose

- **Strengthens:** yes — per-stratum (conditional) risk control holds uniformly (0/20 in all
  three terciles), upgrading the paper's marginal 0/20 to a per-duration-stratum 0/20 with CIs.
- **Heterogeneity to disclose:** a mild, expected long-utterance difficulty gradient (dur2
  accepted-set CER ~0.3 pp above short/medium, acceptance ~2 pp lower). It stays within the
  certified budget (no tercile violates α, all CIs < 1.4 %) and is absorbed by the Mondrian
  thresholds. It should be reported plainly if this result is integrated, not smoothed over — but
  it is a difficulty gradient, not a control failure.

## Integration note (not done here, per instruction)

Not integrated into `paper.tex`. If approved, the natural home is a short paragraph + this table
in §4 (a "conditional coverage across duration terciles" subsection), and it reinstates the
Mondrian arm currently trimmed to future work in §2.3. The numbers regenerate via
`../.venv/bin/python compute_mondrian_tercile.py` (CPU only; needs the relmetrics venv for the
blocked bootstrap). Suite unaffected: `pytest tests/` = 168 passed after this addition.
