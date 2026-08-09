# asr-gate post-analysis — main run (Aishell-1 official test), 2026-07-09

Frozen config: α=0.02 (primary), δ=0.1, LTT procedure=bonferroni, p_value=eb, single
backbone B2=Paraformer (see `../FREEZE-NOTE-2026-07-09.md`). Everything below is computed
CPU-locally from cached artifacts; no new decodes. All pre-existing result files under
`main_results_2026-07-09/` and `pilot_results_2026-07-09/` are read-only inputs; every file
in this directory is new output from this post-analysis pass.

## 1. Attainment table (20 reseeds, α=0.02, test macro-CER of the accepted set)

Per-reseed speaker-level dev cal/tune re-carve (seeds 0–19); each gate applied once to the
same frozen `test_scored.jsonl` (7,176 utterances). Accepted-set macro-CER = mean CER over
utterances with `action == "ACCEPT"` in `applied_test.json`, joined to `test_scored.jsonl`
by `utt_id`. Source: `attainment_table.json`.

| reseed | λ* | acc. fraction | acc.-set macro-CER | n_accept | n_defer | n_ood_refuse | violation (CER>0.02) |
|---|---|---|---|---|---|---|---|
| 0  | -0.1087 | 0.8572 | 0.0100 | 6151 | 1022 | 3 | No |
| 1  | -0.1139 | 0.8775 | 0.0105 | 6297 |  876 | 3 | No |
| 2  | -0.1148 | 0.8800 | 0.0107 | 6315 |  858 | 3 | No |
| 3  | -0.1278 | 0.9186 | 0.0125 | 6592 |  581 | 3 | No |
| 4  | -0.1147 | 0.8800 | 0.0107 | 6315 |  858 | 3 | No |
| 5  | -0.1339 | 0.9337 | 0.0134 | 6700 |  473 | 3 | No |
| 6  | -0.1221 | 0.9043 | 0.0120 | 6489 |  684 | 3 | No |
| 7  | -0.1274 | 0.9182 | 0.0125 | 6589 |  584 | 3 | No |
| 8  | -0.1154 | 0.8825 | 0.0110 | 6333 |  840 | 3 | No |
| 9  | -0.1321 | 0.9299 | 0.0132 | 6673 |  500 | 3 | No |
| 10 | -0.1233 | 0.9072 | 0.0121 | 6510 |  663 | 3 | No |
| 11 | -0.1130 | 0.8747 | 0.0104 | 6277 |  896 | 3 | No |
| 12 | -0.1310 | 0.9268 | 0.0130 | 6651 |  522 | 3 | No |
| 13 | -0.1532 | 0.9622 | 0.0153 | 6905 |  268 | 3 | No |
| 14 | -0.1253 | 0.9126 | 0.0123 | 6549 |  624 | 3 | No |
| 15 | -0.1308 | 0.9264 | 0.0130 | 6648 |  525 | 3 | No |
| 16 | -0.1356 | 0.9362 | 0.0135 | 6718 |  455 | 3 | No |
| 17 | -0.1253 | 0.9125 | 0.0123 | 6548 |  625 | 3 | No |
| 18 | -0.1117 | 0.8697 | 0.0103 | 6241 |  932 | 3 | No |
| 19 | -0.1332 | 0.9326 | 0.0134 | 6692 |  481 | 3 | No |

**Summary:** acc. fraction range **[85.72%, 96.22%]** (mean 90.71%); acc.-set macro-CER
range **[0.01000, 0.01533]** (mean 0.01211), all comfortably under the α=0.02 target.
**Violations: 0/20** — the G1 certificate held on every one of the 20 independent
speaker-level re-calibrations against the single frozen test set. `n_ood_refuse=3` on
every reseed (OOD-REFUSE is computed from `hyp_text` alone, independent of the gate).

## 2. Fixed audit (excess-AURC / matched-abstention / Holm), test, n_perm=2000, seed=0

`asr_gate/audit.py` previously dropped a score family entirely if ANY row had a null score
(`all(...)` roster check) — on test, s1/s2 have 4/7,176 null rows (degraded token-logp
extraction tail), so `holm_family_size` silently collapsed to 0 and `audit_test.json`
reported no audit at all. Fixed to the tool's exclude-and-count convention (mirrors
`gate.py`'s `excluded_missing_s1`): per score, null rows are excluded-and-counted (kept
if ≤1% missing), and a score missing for 100% of rows is skipped with an explicit
`skipped_reason` rather than silently dropped. Output: `audit_test_fixed.json`.

| score | n (post-exclusion) | n_excluded | excess-AURC | p (perm.) | p (Holm) | reject (Holm) |
|---|---|---|---|---|---|---|
| s1 | 7,172 | 4 | 0.0118 | 0.0005 | 0.0010 | **True** |
| s2 | 7,172 | 4 | 0.0137 | 0.0005 | 0.0010 | **True** |

Skipped (all-null, permitted degraded mode — B2 exposes no N-best margin / full token
posteriors): s3 (7,176/7,176 missing), s4 (7,176/7,176 missing), s5 (7,176/7,176 missing —
this standalone audit run never fits the tune-side temperature scale, so s5 is absent
entirely; this is expected, not a bug).

`holm_family_size = 2` (never hardcoded). Both s1 and s2 beat honest random deferral with
excess-AURC > 0 and reject the Holm-adjusted null at α=0.05 (p_holm=0.0010, the permutation
floor at n_perm=2000). macro-CER=0.0198, micro-CER=0.0194 (speaker-blocked CI
[0.0172, 0.0217], n_blocks=20).

## 3. α-frontier (secondary recalibrations, `pilot_results_2026-07-09/cal20_scored.jsonl`, seed=0)

Calibrated with the frozen settings (δ=0.1, procedure=bonferroni, p_value=eb, g1_score=s1,
strata=duration_tercile, auto speaker-split fit_frac=0.5) at each α, then applied once to
the frozen test set. α=0.02 row reuses the already-computed, read-only
`reseed_0/gate.json` + `applied_test.json` (identical inputs/settings to the secondary
runs — verified to match `pilot_results_2026-07-09/recalibration_eb_bonferroni.json`'s
`alpha_0.02_new_bonferroni_eb` entry exactly); α=0.01/0.03/0.05 are new
`gate_alpha{01,03,05}.json` + `applied_test_alpha{01,03,05}.json` in this directory.

| α | certified? | λ* | cal. acc. fraction | test acc. fraction | test accepted-set macro-CER |
|---|---|---|---|---|---|
| 0.01 | **No — vacuous at target** | n/a | 0.0% | 0.0% | n/a (0 accepted) |
| 0.02 (primary) | Yes | -0.1087 | 85.98% | 85.72% | 0.0100 |
| 0.03 | Yes | -0.3079 | 100.0% | 99.80% | 0.0191 |
| 0.05 | Yes | -0.3079 | 100.0% | 99.80% | 0.0191 |

α=0.01 confirms the freeze note's prediction: `certified=False`, `lambda_star=None`,
0/7,173 accepted (dev macro-CER ≈1.63% already exceeds the 1% target, so no λ in the grid
can certify it — reported vacuous-at-target, not a tool failure). α=0.03 and α=0.05
certify at the *identical* λ* (-0.3079): that λ is the most permissive candidate in the
200-point LTT grid (accepts 100% of cal), and dev macro-CER is already low enough that
this floor candidate satisfies both α=3% and α=5% budgets with margin — so raising α
further beyond 3% buys nothing more (a grid-floor saturation effect, not a bug).

## 4. Data-integrity counts through the pipeline

| Stage | Count |
|---|---|
| Test utterances (official Aishell-1 test) | 7,176 / 7,176 transcribed, 0 empty (per freeze note) |
| Test s1/s2 null (degraded token-logp tail) | 4 / 7,176 (0.056%) — excluded-and-counted per score family in the fixed audit |
| Test s3/s4/s5 null | 7,176 / 7,176 (100%, permitted degraded mode: B2 exposes no N-best margin, no full token posteriors, and this standalone audit never fits s5's temperature) |
| Cal (pilot cal20, secondary-α recalibrations) fit-pool missing s1 | 1 / 3,575 → excluded, `n_fit=3,574`, `n_cal=3,567` (identical for α=0.01/0.02/0.03/0.05 — same cal pool, same seed) |
| Reseeds (0–19) attainment | 20/20 certified, 0/20 violations vs α=0.02 |
| Zero-frame / untranscribed wavs (dev+train+test) | 17 zero-frame (2 dev / 15 train / 0 test), 5 untranscribed dev (2 overlap with zero-frame, 3 excluded-and-counted from cal/tune) — per freeze note, unchanged by this pass |

## 5. Limitations

- **Single backbone (B2 = Paraformer only).** B1 (WeNet) had no installable wheel for the
  box's Python; B3 (Whisper) stayed exploratory. The Holm family (m=2: s1, s2) reflects
  the roster actually present, not the design's headline m=10 (5 scores × 2 backbones).
  *Strengthens with:* landing the B3 Whisper decode arm — it would double the audit family
  (adds a second backbone's s1/s2, plus Whisper's `avg_logprob`-derived s1 gives an
  independent cross-architecture confidence signal) and let the Holm correction operate
  over a family that actually spans backbones, not just scores.
- **s3/s4 permanently degraded on this backbone.** N-best margin (s3) and full-posterior
  entropy (s4) are 100%-null for every test row — not a missingness bug, but this backbone
  literally never exposes the artifacts they need, so 2 of the design's 5 primary scores
  are structurally untestable here. *Strengthens with:* FunASR's ModelScope Conformer-
  Aishell as a true second backbone (same toolkit family, different architecture) is more
  likely to expose N-best beams than chasing WeNet's broken wheel, restoring s3 to the
  audit family.
- **Single, clean, read-speech dataset (Aishell-1, no noise/SNR strata).** The certificate
  and audit are both validated only against clean studio Mandarin read speech; robustness
  under acoustic stress is untested. *Strengthens with:* the design's D2 MUSAN noise arm —
  reinstating SNR strata would both stress-test the G1/G2 gate under distribution shift the
  KS-warning is meant to catch, and give the Mondrian stratification a genuine second axis
  beyond duration.
- **Audit family currently small (m=2).** With only s1/s2 testable, the Holm correction is
  nearly toothless (m=2 barely penalizes vs. uncorrected). *Strengthens with:* either of the
  above two backbone/score expansions, which would make the multiplicity correction do
  real work and make the "beats honest random deferral" claim cover more of the design's
  intended score family.

## Reproduction

```
# Task 2: fixed audit on test
python -m asr_gate.cli audit --instances ../test_scored.jsonl \
  --n-perm 2000 --seed 0 --alpha 0.05 --out audit_test_fixed.json

# Task 3: secondary-alpha recalibrations (pilot cal20) + apply to test
python -m asr_gate.cli calibrate --instances ../../pilot_results_2026-07-09/cal20_scored.jsonl \
  --alpha {0.01,0.03,0.05} --delta 0.1 --seed 0 \
  --ltt-procedure bonferroni --ltt-pvalue eb --out gate_alpha{01,03,05}.json
# apply_gate(gate_alphaXX, test_scored.jsonl) -> applied_test_alphaXX.json (see script)

# Task 4: attainment table across reseed_0..19 -> attainment_table.json
```
