# asr-gate design freeze — 2026-07-09

**Authorization:** user go-ahead 2026-07-09 ("The asr box was ready, you can launch now"),
following review of the pilot findings and the EB+Bonferroni recalibration evidence
(`pilot_results_2026-07-09/recalibration_eb_bonferroni.json`).

## Frozen configuration (main run, Aishell-1 official test — touched exactly once)

| Item | Frozen value | Basis |
|---|---|---|
| Backbone roster | **B2 only** (FunASR `paraformer-zh` / SeaCo-Paraformer, HF-mirror checkpoint) | B1 (WeNet) has no installable wheel for the box's Python; B3 (Whisper) stays exploratory. **Disclosed deviation** from the design's 2-backbone m=10 family: the Holm family is computed from the roster actually present (pilot: s1,s2 × 1 backbone = 2 tests); B1 joins later via the identical frozen machinery if/when its decode entry point lands. |
| G1 procedure | **LTT, Bonferroni-over-grid, empirical-Bernstein p-values** (`procedure=bonferroni`, `p_value=eb`) | Pilot: fixed-sequence+HB certified nothing at α=2% despite accepted-region CER 0.95% (power failure, trace on record); EB+Bonferroni on the same real data: α=2% certified at 86% acceptance, α=3% at 100%. |
| α targets | **primary 0.02**; secondary 0.03, 0.05 recalibrated post-hoc from the same cached artifacts (CPU-only, no new test decodes); 0.01 expected VACUOUS-AT-TARGET (dev macro-CER 1.63% exceeds it) and reported as such per K4 | pilot dev numbers |
| δ | 0.1 | design default |
| Reseeds | 20 speaker-level dev cal/tune re-carves (seeds 0–19); dev decoded once (pilot artifact), test decoded once | design §3.2 |
| Mondrian strata | **duration_tercile only** | decode records carry `gender=null` (speaker.info join not implemented); a gender axis would degenerate to all-`gender_unseen`. **Disclosed descope**; gender Mondrian is future work (metadata verified available: 186 M / 214 F). |
| Normalizer | `asr-gate-cer-normalizer-v1` (pinned; same binary for cal and test) | pilot K1 |
| Certificate estimand | macro-CER of the accepted set (per-utterance mean); micro-CER reported with speaker-blocked bootstrap CI, never certified | design §2.3 |

## Data-integrity facts carried into the paper
17 zero-frame wavs in the extracted openslr-33 Aishell-1 (2 dev / 15 train / 0 test) — skipped
and sidecar-logged at decode; 5 untranscribed dev wavs (2 of them the empty ones; 3 decodable,
excluded-and-counted from cal/tune); 1/7,142 cal utterances with degraded token-logp extraction
(excluded-and-counted in gate.json); test split: 7,176/7,176 transcribed, 0 empty.

## K6 scoop note
Header-instructed re-scan (conformal + ASR/CER/selective transcription, 2025–26) was last run at
design time (2026-07-08, no closing hit); re-run at submission per K6.
