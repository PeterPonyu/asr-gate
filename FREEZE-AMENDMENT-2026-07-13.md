# asr-gate landscape freeze amendment — 2026-07-13 (amends FREEZE-NOTE-2026-07-09 + EXPANSION-AMENDMENT-2026-07-09)

**Status: SIGNED — user sign-off recorded 2026-07-13 (explicit in-session direction: "i sign off
the asr freeze amendment"). The frozen matrix, α-grid, δ, calibration-pooling rule, and
test-touched-once rules below are now BINDING for the landscape box run.** Original preregistration
text follows unchanged. This document preregisters the resubmission-motivated
*ASR-landscape expansion* (backbone × corpus matrix) BEFORE any test split of any new corpus is
touched and before any new backbone decodes the existing corpora's test splits. It is written now,
during zero-cost local prep, so the matrix and the target α-grid are frozen *ahead* of the compute
— the paper's whole credibility rests on its disclosed-deviation model, and adding backbones/corpora
after a red-team is a forking-paths hazard (RED-TEAM-REPORT.md M5) unless the choices are fixed in
advance. **No box has been booted, no test data touched, no `boxkit_api.py`/ssh/spend incurred by
this amendment.** It takes effect only when the user authorizes the landscape box run (the same
discipline as EXPANSION-AMENDMENT-2026-07-09, whose authorization was the user booting the box after
reviewing the plan). Until then, the frozen configuration in force remains FREEZE-NOTE-2026-07-09 +
EXPANSION-AMENDMENT-2026-07-09.

**Motivation (from the red-team lever, not from the data).** RED-TEAM-REPORT.md leaves two
individually reject-capable *scope* levers (COMPUTE-PLAN-2026-07-13.md §0): **M1** — the non-trivial
certificate collapses to one cell (α=2%, Paraformer, Aishell-1, clean); **M2** — Paraformer's
undisclosed training data threatens that one cell with a contamination reading. Both dissolve by the
same move: add strong certified backbones with **documented, open training data** across **different
architectures**, made non-vacuous on **≥2 more credential-free corpora**. This amendment freezes
exactly which backbones, which corpora, which α-grid, and the calibration-pooling rule, so that the
subsequent box run has no post-hoc degrees of freedom.

---

## 1. Frozen backbone roster (matrix rows)

The roster is fixed to **three load-bearing backbones spanning three architectures**, all with
**exposable per-token posteriors** (the make-or-break criterion — s1/s2 and the certificate are
dead without per-emitted-token log-probs) and **credential-free** weights. Two **stretch** backbones
are named but **gated** (§5); they are NOT part of the frozen load-bearing roster and cannot become
load-bearing without satisfying the gate below.

| ID | Backbone (exact id) | Arch. | Posterior path | Training-data provenance | Frozen role |
|---|---|---|---|---|---|
| **B2** | FunASR `paraformer-zh` (SeaCo-Paraformer-large; ModelScope `iic/speech_paraformer-large...`) | NAR CIF | Path C: `_seaco_decode_with_ASF` hook (verified, funasr 1.3.14) | **undisclosed** (the M2 liability being answered) | KEEP (existing certified) |
| **B3'** | `BELLE-2/Belle-whisper-large-v3-zh` (non-punct v3) | AED | Path A: `compute_transition_scores` + BPE→char (reuses `decode_whisper.py` VERBATIM, `--model-name … --language zh`) | **documented**: Aishell-1/2 train + WenetSpeech + HKUST (test excluded) | ADD |
| **B4** | `zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2` (offline zipformer transducer) | RNN-T | Path D: `OfflineRecognizerResult.ys_probs` (new `decode_sherpa_onnx.py`) | **open**: multiple public Mandarin corpora, provably excluding Aishell-1 test — the red-team's *named* M2 fix | ADD |
| B5 (audit-only) | `openai/whisper-large-v3` zero-shot (existing) | AED | Path A | n/a (vacuous stressor) | keep as audit-only |

**Architectural diversity is load-bearing to M2:** three organizations (Alibaba/FunASR, BELLE/Lianjia,
k2-fsa) trained on different data with different architectures; simultaneous Aishell-1-test
memorization by all three independently is not a credible alternative explanation, so a low
accepted-set CER shared across all three bounds and answers the contamination card. **B4's fully
open training data (Aishell-1 train + other public Mandarin, test excluded) is the single strongest
answer** and is exactly the remediation the red-team named.

**Posterior-exposure is the gating risk and is front-loaded (§5).** B2 (Path C) and B3' (Path A) are
proven paths. B4's `ys_probs` token→char alignment is documented but **must be verified on-box in the
`--limit 20 --probe` smoke pass BEFORE the full decode is trusted** (`decode_sherpa_onnx.py` ships a
synthetic-fixture unit test of the alignment logic, but the real `ys_probs` population under the
transducer decoding mode is confirmable only on a real model). If B4's smoke fails to populate
`token_logps`, B4 degrades to `skipped-degraded` and is reported as such (verdict-symmetric honesty,
exactly as the Conformer cell already is), and the roster falls to B2+B3'.

## 2. Frozen corpus set (matrix columns)

Fixed to **credential-free** Mandarin corpora with real test splits low enough in CER to certify
non-vacuously, plus the retained English arm. Corpora behind an application/agreement
(**AISHELL-2**, **WenetSpeech**, LDC/HKUST) are **explicitly excluded from the autonomous run** and
held as reviewer-response-only (COMPUTE-PLAN §3.2) — they cannot be fetched without institutional
credentials the author's on-file email (`fuzeyu99@126.com`, public) does not confer.

| Corpus | Source | Frozen test split | License | Role |
|---|---|---|---|---|
| **Aishell-1** | openslr-33 (autodl-pub mirror preferred) | official test: **7,176 utts / 20 spk** | Apache-2.0 | KEEP (certify all three backbones) |
| **THCHS-30** | openslr-18 | test subset D: **n=1,339** as materialized by the frozen mirror route (EXPANSION-AMENDMENT addendum), OR the full **2,495**-utt openslr test split if the direct openslr fetch succeeds on the landscape box — whichever is realized is **measured-and-recorded** at stage time, gate ≥1,300 | Apache-2.0 | re-decode with strong backbones (only vacuous Whisper ran) |
| **aidatatang_200zh** | openslr-62 | speaker-disjoint **capped ~4–5k-utt** subset of `corpus/test/` (subset rule below) | CC BY-NC-ND 4.0 | NEW (3rd Mandarin corpus) |
| **(opt) MagicData** | openslr-68 | speaker-disjoint **capped ~4–5k-utt** subset of the explicit test set (subset rule below) | CC BY-NC-ND 4.0 | OPTIONAL 4th corpus |
| **LibriSpeech** | openslr-12 | existing test-clean / test-other | CC BY 4.0 | KEEP (cross-lingual arm) |

**aidatatang / MagicData subset rule (frozen NOW, before test is touched).** To bound decode cost
and keep a clean calibration/eval split, the decoded test set for each of these two corpora is a
**speaker-disjoint cap**: order the corpus's own `test/` speakers by ascending speaker-id (string
sort, deterministic), accumulate whole speakers until the cumulative utterance count first reaches
**≥4,000**, and decode exactly those speakers' utterances (the last-added speaker is included whole —
no mid-speaker truncation). This yields a ~4–5k-utt test set that is (a) reproducible from the id
sort alone, (b) never split mid-speaker, and (c) fixed before any transcript is read. The
calibration pool for these corpora is drawn from their **train/dev** splits only (never the decoded
test speakers) — see §3.

**License handling (frozen).** aidatatang / MagicData are CC BY-NC-ND. Consistent with the ESC-50
precedent already in the paper, they are used as **data, not executed code**; no NC/ND-licensed
artifact is redistributed (decode artifacts are derived *metrics*, not the corpus); a one-line
license-appendix disclosure is added per corpus. If house license review objects to ND, THCHS-30
(Apache-2.0) + Aishell-1 alone already deliver the non-vacuous cross-corpus result, and the two
ND corpora are dropped without disturbing the frozen α-grid or the M1/M2 answers.

## 3. Frozen calibration-pooling rule

The certificate/audit machinery is corpus- and backbone-agnostic (`asr_gate` consumes canonical
decode JSONL, never the model); every cell is one decode JSONL through the same frozen CLI. The
calibration protocol is fixed as follows and applies uniformly to every (backbone × corpus) cell:

1. **Calibration source is never the decoded test split.** For each corpus, calibration draws from a
   **speaker-disjoint** pool of that corpus's own train/dev utterances (Aishell-1 dev; THCHS-30
   train/dev; aidatatang/MagicData train/dev). The decoded test speakers (§2) are eval-only. This
   preserves the frozen "calibrate on dev, certify on test" protocol
   (FREEZE-NOTE-2026-07-09).
2. **Calibration-budget axis is preregistered as a reported variable, not a tuned one.** Per
   COMPUTE-PLAN §1.3, each cell may **pool a larger speaker-disjoint calibration set** to push the
   certifiable α tighter; the reported result for each cell is the certificate at the **frozen
   primary calibration budget** (matching the existing main-run scale, n_cal≈3,567 where the corpus
   supplies it) **plus** the disclosed calibration-size sweep showing the pool→α frontier (the
   `mandarin_calsweep_2026-07-13/` methodology, mirroring `english_calsweep_2026-07-13/`). The sweep
   is over **cached scores (0 GPU-hours)** and is labelled post-hoc-and-disclosed, exactly as the
   English calsweep is. No single "best" pool size is cherry-picked as the headline; the full
   frontier is reported.
3. **Cross-corpus transfer cells** (calibrate on corpus A dev, certify on corpus B test) keep A's
   calibration speaker-disjoint from B's eval by construction (different corpora), and are reported
   as transfer cells, never conflated with in-corpus cells.

## 4. Frozen certificate configuration and α-grid

Unchanged from FREEZE-NOTE-2026-07-09 / EXPANSION-AMENDMENT-2026-07-09 except the α-grid is stated
explicitly for the landscape:

| Item | Frozen value |
|---|---|
| **α-grid** | **{0.02, 0.03, 0.05, 0.10}** primary landscape grid; **α=0.015** additionally reported as the sub-base-rate *binding* probe on backbones/corpora whose full-set CER exceeds it (mirrors `alpha015_2026-07-13/`). α=0.01 expected VACUOUS-AT-TARGET and reported as such. |
| δ | 0.1 |
| G1 procedure | LTT, Bonferroni-over-grid, empirical-Bernstein p-values (`procedure=bonferroni`, `p_value=eb`) |
| Mondrian strata | duration_tercile only (gender metadata still not joined; disclosed descope carried forward) |
| Certificate estimand | macro-CER of the accepted set; micro-CER reported with speaker-blocked bootstrap CI, never certified |
| Reseeds | 20 speaker-level calibration re-carves per cell (seeds 0–19) |
| Normalizer | `asr-gate-cer-normalizer-v1` (pinned; same binary for cal and test), `numeral_policy=keep` |

**Test-touched-once-per-corpus rule (frozen).** Each corpus's frozen test split (§2) is decoded and
scored **exactly once per backbone**; the α-grid frontier, the reseed loop, and the calibration-size
sweep are **all recomputed from those cached scores on CPU** with **no re-decode of any test split**.
No test split is decoded a second time under any α, any calibration pool, or any reseed. This is the
existing main-run discipline (dev decoded once, test decoded once) extended to every new cell.

**Holm-family `m` recomputation (frozen rule, no value asserted here).** The audit family `m` is
COMPUTED from the realized roster after the decode gates pass — one test per (auditable score s_k ×
backbone × corpus/noise condition) actually present with ≥99% non-null values — carried forward
verbatim from EXPANSION-AMENDMENT-2026-07-09's Holm rule. No value of `m` is asserted in advance; the
realized `m` is recorded in the audit result JSON and reported with the same denominator everywhere.

## 5. Frozen stretch-backbone gating rule

Two 4th-architecture stretch backbones are named but **gated**. They may become reported certified
cells **only if** their posterior hook verifies on-box in the first smoke pass; otherwise they are
recorded as `skipped-degraded` (honest, verdict-symmetric) and contribute nothing to the certificate.

| Stretch backbone | Exact id | Gate condition (frozen) | Fallback if gate fails |
|---|---|---|---|
| **SenseVoiceSmall** (CTC) | ModelScope `iic/SenseVoiceSmall` / HF `FunAudioLLM/SenseVoiceSmall` | a CTC-logit capture hook exposes per-frame log-softmax at deduped non-blank emissions aligned 1:1 to emitted chars, verified in the `--limit 20 --probe` smoke **within the first hour on-box** | `skipped-degraded`, reported as such |
| **FireRedASR-AED-L** | `FireRedTeam/FireRedASR-AED-L` | a decoder-logit hook exposes per-token logps aligned to chars, verified in the smoke pass; AND its clone-repo+conda install completes | text-only "reported-CER reference point" only (0.55% Aishell-1), NOT a load-bearing certified cell |

**No stretch backbone is on the critical path.** The M1/M2 answers stand on B2+B3'+B4 alone; the
stretch adds are diversity-only and are explicitly permitted to fail closed. WeNet and the ModelScope
Conformer are **not** revived (no installable wheel / verified to expose no posteriors → stays
`skipped-degraded`).

## 6. What this amendment does NOT change

- The existing frozen main-run and expansion numbers (Paraformer × Aishell-1 clean/noise, the
  cross-corpus Whisper × THCHS cell) are untouched and remain the headline; the landscape cells are
  additive.
- The free CPU/prose fixes the red-team also requires (M3 abstract honesty, M4 comparators, M5/M6
  stats reframes, **M7 the four missing 2025–26 citations**) are out of scope here and still owed —
  compute fixes scope, not framing (COMPUTE-PLAN §8).
- No manuscript prose is edited by this amendment; it is a preregistration document only.

## 7. Sign-off

- [ ] **User authorization** to boot the landscape box and execute the frozen matrix above
  (authorization = the user booting the box after reviewing this amendment, per the
  EXPANSION-AMENDMENT precedent). **Until this box is checked, no test split of any corpus in §2 is
  touched by any new backbone, and this document is a non-binding draft.**
- [ ] Exact model ids / licenses / checksums re-verified at stage time and recorded in
  `DATA_MANIFEST` (B3' `BELLE-2/Belle-whisper-large-v3-zh`; B4
  `zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2` with `ys_probs` confirmed populated;
  openslr-62/68 URLs + md5).
- [ ] Realized THCHS-30 / aidatatang / MagicData test counts measured-and-recorded (§2 gates).
- [ ] Realized Holm `m` recorded in the audit JSON (§4).

---

## D1. Dated disclosed deviation — SLR62 withdrawn upstream (2026-07-13, launch day)

At box-launch time (same day as sign-off, BEFORE any new-corpus test decode), openslr **SLR62
(aidatatang_200zh) was found WITHDRAWN from openslr.org** — `https://www.openslr.org/62/` returns
"Resource not found: 62" and the tgz 404s on every openslr host, verified independently from two
networks (workstation + the landscape box). The corpus is therefore no longer credential-free
obtainable, through no analysis choice of ours; no aidatatang data was ever staged or decoded.

**Substitution (deviation-minimal):** the third Mandarin corpus becomes **MagicData (SLR68)** —
which is already IN this amendment's frozen matrix as the optional fourth corpus (§1.1 column
"(opt) MagicData test", §3.1 row, same CC BY-NC-ND license class, same data-not-code handling,
same speaker-disjoint test cap rule, `MAGICDATA_TEST_CAP=4000`). `test_set.tar.gz` (2.2 GB)
verified live (HTTP 200, content-length 2,201,936,013) from both networks. No other frozen choice
(backbones, α-grid, δ, pooling, touched-once rule) changes. The realized matrix is
aishell × thchs30 × magicdata; aidatatang remains listed in §1.1 for the record with this
deviation note. Disclosed in the manuscript's deviation table like every prior amendment.
