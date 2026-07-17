# asr-gate expansion amendment — 2026-07-09 (amends apps-design/03 + FREEZE-NOTE-2026-07-09)

**Authorization:** user booted the asr box in GPU mode for the expansion run (2026-07-09
evening) after reviewing the expansion plan; this document is the preregistration amendment
`run_expansion.sh`'s guard requires.

## Expanded roster (fixed here)
| Axis | Values | Notes |
|---|---|---|
| Backbones | B2 paraformer-zh (frozen main-run baseline) + **B3 whisper-large-v3** (HF, fp16 safetensors; fp32/bin/tf/flax variants deleted from cache — disclosed) + **B1' conformer-aishell** (ModelScope `iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch`) | replaces the design's unavailable WeNet B1; each decoder smoke-tested against real output before the grid (gate in the session chain) |
| Corpora | Aishell-1 (openslr-33, phase-0-audited) + **THCHS-30 test split** (materialized from the `urarik/thchs30` parquet mirror; openslr direct unusably slow box-side — disclosed substitution of the delivery route, content identical) | cross-corpus arm: calibrate on Aishell dev, certify transfer on THCHS test |
| Noise | clean + {5, 15, 25} dB SNR additive mixing on Aishell test | **noise source = ESC-50 environmental corpus (`ashraq/esc50`), a DISCLOSED substitution for the design's MUSAN**: openslr-MUSAN is unreachable at usable speed from the box and no raw MUSAN mirror exists on HF. The mixer is source-agnostic; SNR stratification and the Mondrian SNR axis are unchanged in construction. |
| Certificates | α=0.02 primary (0.01/0.03/0.05 frontier), δ=0.1, LTT procedure=bonferroni, p_value=eb | unchanged from FREEZE-NOTE-2026-07-09 |
| Reseeds | 20 per calibration cell | unchanged |

## Holm-family recomputation rule (preregistered)
The audit family m is COMPUTED from the realized roster after decode gates pass — one test per
(auditable score s_k, backbone, corpus/noise condition) actually present with ≥99% non-null
values — mirroring the main run's disclosed-roster precedent (m fell 10→2 there). The exact m
is recorded in the audit result JSON and reported with the same denominator everywhere. No
value of m is asserted in advance here, per the roster-derived rule the design's critic pass
established.

## Data-integrity carryforward
THCHS materialization count and transcript coverage are gated in-session (≥1,400 wav+trn pairs
— the parquet mirror's test split materialized 1,497 in the partial no-card run; the full
count is measured at card-day materialization and recorded); ESC-50 clip count gated >1,500;
every decode content-gated on n_written and non-null-score fraction.

### Addendum (2026-07-09, no-card staging pass 2 — MEASURED counts, recorded per the rule above)
The mirror's complete test parquet (`data/test-00000-of-00001.parquet`, the repo's only
test-named file) contains **1,339 unique test utterances**; all 1,339 materialized with
transcripts (100% coverage, 0 missing from data/). The provisional ≥1,400 gate figure was an
overestimate from the earlier PARTIAL materialization (1,497 counted mid-download across
mixed row-groups, before split structure was recoverable). Per this section's own
measure-and-record rule, the realized test roster is fixed at **n=1,339** and the in-session
gate is set to ≥1,300 (THCHS_MIN_PAIRS default updated accordingly). Split membership was
rebuilt from the mirror's own test parquet file listing (names extracted from the audio.path
column; symlink dir `data_thchs30/test/` → `../data/`), so membership is exactly the mirror's
test split — no re-carving. THCHS remains apply/audit-only (calibration stays on Aishell dev),
so this count affects certification-arm sample size, not calibration validity. Measured
detail: `/root/nocard_stage2.log` on the asr box; ESC-50 measured clip count = 2,000 (>1,500
gate passes); noise corpus resides at `/root/autodl-tmp/noise_corpus` (ESC50_DIR default
aligned to the materialized location).
