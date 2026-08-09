# asr-gate

**ASR dual-gate reliability research theme** (public theme repo).
Contains package code, frozen experiment statistics needed to rebuild the
manuscript, figure SSOT, and venue kits (canonical + IEEE TASLP).

Frozen bulk decode trees / corpus staging stay local-only (see `.gitignore`);
public archive: Zenodo DOI [10.5281/zenodo.21392289](https://doi.org/10.5281/zenodo.21392289).

Portfolio layout: sibling of `reliability-commons/`; commons path
`tools/asr-gate` is a symlink here. Audit kits via `papers/asr-*`.

A conformal transcription-triage gate + evaluation pipeline for Mandarin
ASR on Aishell-1. Given decode artifacts (N-best hypotheses with
log-probabilities), `asr-gate` routes each utterance to
`{ACCEPT, DEFER, OOD-REFUSE}` with a **certified bound on the CER of the
auto-accepted set** (Learn-then-Test), plus an excess-AURC audit asking
whether field-standard ASR confidence scores beat honest random deferral
at all. A thin wrapper over `relmetrics` (from `reliability-commons`).

## Quickstart

```bash
# From ml-reliability-research/asr-gate (or via reliability-commons/tools/asr-gate symlink):
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../reliability-commons   # relmetrics
pip install -e .                        # asr-gate itself
pip install -e '.[test]'                # + pytest

python -m pytest
```

## Quickstart

```bash
# From reliability-commons/tools/asr-gate:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../../                 # relmetrics (editable, from reliability-commons root)
pip install -e .                      # asr-gate itself
pip install -e '.[test]'              # + pytest

python -m pytest                      # 55 tests, ~10s, no network/GPU
```

```bash
# Canonical utterance JSONL in, at every step (utt_id, speaker_id,
# duration_s, hyp_text, nbest[{text, logp, token_logps}], [ref_text],
# [gender], [region]).
asr-gate ingest    --hyps decode.jsonl --format funasr --refs refs.txt --out canonical.jsonl
asr-gate score     --instances canonical.jsonl --out scored.jsonl
asr-gate calibrate --instances cal.jsonl --alpha 0.02 --delta 0.1 \
                    --strata duration_tercile,gender --out gate.json
asr-gate apply     --gate gate.json --instances new.jsonl --out applied.json
asr-gate audit     --instances test.jsonl --n-perm 2000 --alpha 0.05 --out audit.json
asr-gate report    --audit audit.json --gate gate.json -o report.md
```

Full box-side pilot: `orchestration/run_pilot.sh` (see below).

## Honest-uncertainty rules (design §2.5)

| Rule | Exit state | Meaning |
|---|---|---|
| Stratum has < `min_stratum_n` (default 200) calibration utterances | `DEFER` | not enough calibration data in this Mondrian stratum for a valid per-stratum bound; unconditional defer |
| Hypothesis is > 20% non-CJK/non-digit characters | `OOD-REFUSE` | a validity statement (the certificate's domain is read Mandarin), not an uncertainty estimate -- a distinct exit state from `DEFER` |
| Incoming batch's score distribution KS-departs from the calibration fingerprint beyond a preregistered distance | warning only (batch-level, in `apply`'s output) | never triggers silent recalibration |
| No `ref_text` on any row | refusal (raises) on `calibrate`/`audit` | these need ground truth; `apply` is ref-free by design |

## Design notes / what each module does

- `io.py` -- canonical utterance schema (JSONL) + format adapters
  (`funasr`, `whisper`, `wenet`, `custom-schema`); adapters are the only
  format-aware code in the tool.
- `cer.py` -- character-level CER after a pinned Mandarin normalizer
  (NFKC + punctuation strip + a digit-substitution numeral policy),
  clipped to `[0, 1]` with a clip counter. **Macro** CER (mean of
  per-utterance CER) is the certified statistic; **micro** CER (total
  edits / total ref chars) is reported alongside with a speaker-blocked
  bootstrap CI and is *never* certified -- these are kept deliberately
  distinct throughout (`macro_cer`/`micro_cer`, never just "CER").
- `scores.py` -- s1 (mean token logp), s2 (min token p), s3 (N-best
  margin, `None` in 1-best-only degraded mode), s4 (negative mean token
  entropy, needs full per-token posteriors -- `None` in the common case
  they're absent), s5 (temperature-scaled s1, temperature fit on TUNE via
  NLL), s6 (exploratory CER-regressor **stub** -- see deviations below).
- `ltt.py` -- the one genuinely new module: a Learn-then-Test
  selective-risk certificate using the critic-corrected bounded-mean
  reformulation of the design doc (§2.3), not a naive HB test on the
  selective-risk ratio. Read its module docstring before touching the
  statistics.
- `gate.py` -- G1 (LTT) + G2 (Mondrian split-conformal upper bound on a
  tune-fit `score -> CER` map, conformalized on cal), Mondrian strata
  (duration terciles frozen at calibration + optional gender), and the
  honest-uncertainty refusal rules.
- `audit.py` -- excess-AURC vs `relmetrics.aurc.random_aurc` at matched
  abstention (`relmetrics.nulls.matched_abstention_null`), Holm across
  the *actually-present* score/backbone roster (never hardcoded to 10),
  speaker-blocked micro-CER bootstrap CI.
- `cli.py` -- `ingest / score / calibrate / apply / audit / report`.

## Deviations from the design doc's literal wording (and why)

1. **`calibrate` gains an optional `--tune` flag.** The design's §2.1 CLI
   listing shows only `--instances cal.parquet`. §2.3's fit/calibrate
   leakage rule (s5's temperature, G2's residual map: fit on tune, never
   cal) can't be expressed with one input file, so `--tune` was added.
   Omitting it auto-splits `--instances` by SPEAKER (never utterance)
   into an internal fit/conformalize pair; passing it lets you supply an
   explicit, separately-decoded tune cohort (the real pilot's use case).
   `gate.calibrate_gate` raises if `tune_instances` is the same object as
   `cal_instances` or shares any speaker with it.
2. **Input tables are JSONL throughout, not `.parquet`.** `io.py`'s own
   module docstring (part of the target spec) says "canonical utterance
   table: JSONL in/out"; §2.1's example paths show `.parquet`. JSONL was
   kept as the single, authoritative format (matches `io.py`'s spec,
   keeps the dependency footprint at just numpy/pandas/relmetrics/scipy,
   and is what every test drives).
3. **s6 is a ridge-regularized linear regressor (closed-form
   `numpy.linalg.solve`), not a GBRT.** The design calls for a
   "gradient-boosted" CER regressor, but the pinned dependency set is
   numpy + pandas + relmetrics (no scikit-learn); the target layout's own
   phrase "s6 stub" signals this is an intentional placeholder. Flagged
   exploratory throughout (never enters the Holm family), matching the
   design's own BH-only treatment of s6.
4. **s5's temperature-scaling mechanics are a concrete instantiation, not
   a literal spec.** No raw logits exist in the canonical schema (only
   log-probabilities), so temperature scaling is defined as fitting a
   scalar `T` minimizing the NLL of a binary "clean" (`CER == 0`) label
   under `sigmoid(s1 / T)`, then `s5 = s1 / T`. Documented in
   `scores.py`'s module docstring.
5. **G2's `score -> CER` map is a plain univariate OLS fit** (fit on
   tune, `relmetrics.conformal.SplitConformal`/`MondrianConformal` on the
   residuals, conformalized on cal), not tied to s6's multi-feature ridge
   stub -- kept simple and decoupled from s6's exploratory-only status.
6. **`ltt.build_lambda_grid`'s default `min_accept_frac` is 0.1, matching
   the design's own K4 vacuity threshold** ("no score family accepts
   >= 10% of test utterances" is reported VACUOUS-AT-TARGET). This
   matters for the *fixed-sequence testing procedure's power*: the
   bounded-mean LTT statistic is evaluated over ALL calibration points at
   every candidate lambda (zero-imputed for rejected points, which is
   what keeps its bound fixed across lambda), so its signal is diluted
   proportional to the accepted fraction -- a top-of-grid lambda that
   accepts only a handful of points is essentially unfalsifiable, and
   because fixed-sequence testing halts at the first non-rejection,
   wasting that first test on a hopeless near-empty acceptance set would
   make the whole procedure vacuous even when informative lambdas exist
   further down the grid. Full reasoning + literature pointers in
   `ltt.py`'s docstrings. Practical upshot: this LTT construction needs
   calibration sets in the thousands (not hundreds) to have real power
   at typical alpha/effect-size regimes -- confirm calibration-set size
   before trusting a VACUOUS result as a real finding rather than an
   underpowered one.
7. **`orchestration/split_cal_tune.py`** is one small helper file beyond
   the originally sketched orchestration layout, needed so
   `run_pilot.sh`/`run_main.sh` can do dev's speaker-disjoint cal/tune
   carve without duplicating that logic as an inline `python3 -c` blob in
   two shell scripts. Thin CLI wrapper over `gate.split_by_speaker`.
8. **Format adapters (`io.adapt_funasr`/`adapt_whisper`/`adapt_wenet`)
   assume documented-but-unverified raw shapes** (no network access to
   probe real FunASR/Whisper/WeNet outputs during this build). Each
   adapter's docstring states its exact assumed input shape and degrades
   (`None` fields) rather than guessing on an unrecognized one. Verify
   against real decode output at Phase 0 before trusting the adapter
   path; `orchestration/decode_paraformer.py` sidesteps this entirely by
   emitting the canonical schema directly from FunASR's Python API.
9. **`audit.run_audit` hard-refuses (raises) with no `ref_text`**, unlike
   `ope-audit`'s degrade-to-disagreement-only mode -- this matches design
   §2.5's literal "audit/calibrate refuse without refs" wording for this
   tool specifically.

## Box-side pilot (design §7 Phase 1)

Runs on the AutoDL box (funasr + a GPU for decode; everything downstream
is CPU-only). Every step is idempotent (skips if its output already
exists).

```bash
export DATA_ROOT=/root/autodl-pub/Aishell   # VERIFY against the actual box layout first (Phase 0)
export RESULTS_DIR=./pilot_results
export DEVICE=cuda
export ALPHA=0.02   # one of {0.01, 0.02, 0.03} per design §1
export DELTA=0.1
export SEED=0

pip install funasr           # box-side only; NOT an asr-gate dependency
pip install -e .              # this package (numpy/pandas/relmetrics/scipy only)

./orchestration/run_pilot.sh
```

Produces (under `$RESULTS_DIR`): `decode_dev.jsonl`, `dev_canonical.jsonl`,
`cal20.jsonl`/`tune20.jsonl` (speaker-disjoint 20/20 dev carve), `gate.json`
(G1 lambda*, G2 per-stratum thresholds, provenance stamp),
`applied_tune20.json` (per-utterance ACCEPT/DEFER/OOD-REFUSE + certificate
echo), `audit_tune20.json` (excess-AURC/Holm table), `pilot_report.md`.

K1-K3 (design §4) must still be checked by hand against those result
JSONs -- the script does not auto-adjudicate kill criteria.

Main run (post-freeze only): `orchestration/run_main.sh`, guarded by
`REQUIRES_FREEZE=confirmed` -- refuses to run otherwise. Structure-only
skeleton (decode-once + 20 calibration reseeds against a frozen config);
extend it with B1 (WeNet)/B3 (Whisper) decode entry points mirroring
`decode_paraformer.py`'s canonical-JSONL-out contract once those are
written.

## Testing

```bash
python -m pytest -v
```

55 tests, synthetic data throughout (no network, no GPU):
`test_cer.py`, `test_scores.py`, `test_ltt.py` (includes the LTT
violation-rate correctness check over 200 calibration resamples),
`test_gate.py`, `test_audit.py`, `test_cli_e2e.py` (full
ingest→score→calibrate→apply→audit→report pipeline via subprocess, plus
`decode_paraformer.py --help` with no `funasr` installed).
