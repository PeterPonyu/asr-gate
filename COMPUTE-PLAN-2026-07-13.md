# asr-gate — COMPUTE PLAN: from "a gate on one model/corpus" to a certified-triage PROTOCOL

**Author:** lever-asr (strategy + feasibility research, READ-ONLY + web). **Date:** 2026-07-13.
**Scope:** design and cost the compute experiments that qualitatively raise the paper from a
single-backbone/single-corpus certificate to a *certified-ASR-triage protocol validated across the
ASR landscape*, neutralizing the red-team's two reject-capable levers. **No box was booted; no
`boxkit_api.py`, no ssh, no spend.** This is a turnkey plan the USER executes on a box later.

Companion inputs read: `manuscripts/paper.tex`,
the four `orchestration/decode_*.py` scripts, `asr_gate/scores.py`,
`asr_gate/corpora.py`, `asr_gate/cli.py`, `../AUTODL-PUB-DATA-MAP-2026-07-10.md`.

---

## 0. The attack this plan defeats (why compute is the lever, not prose)

The red-team could not land a FATAL blow (18/18 numbers trace), but has **two individually
reject-capable levers**, both about *scope*, not correctness:

- **M1 — the certified contribution collapses to ONE substantive cell.** α=1% is vacuous;
  α=3%/5% saturate at accept-all (budget ≥ full-set error); Whisper is vacuous everywhere;
  Conformer is `skipped-degraded`. The entire non-trivial certificate is **α=2%, Paraformer,
  Aishell-1, clean** — "a workshop-sized result dressed as a journal certification framework."
- **M2 — contamination threatens that ONE cell.** Paraformer-large is an industrial checkpoint
  with **undisclosed training data**; the ~1% accepted-set CER headline could be Aishell-1-test
  memorization. Disclosure alone will not satisfy a hostile referee; the remediation the red-team
  *names* is: "add a backbone with **known training data that provably excludes Aishell-1 test**,
  even if weaker, and show a non-vacuous certificate there."

Both levers are dissolved by the **same** move: add strong certified backbones with **documented,
open training data** spanning **different architectures**, and make the certificate **non-vacuous
on ≥2 more corpora**. That converts "one cell" into a **frontier of certified cells across the ASR
landscape** and makes the guarantee backbone-, provenance-, and corpus-robust. This plan does not
touch the paper's other (free, CPU-only) fixes — M3 abstract-honesty, M4 real comparators (already
partly in via s5/s6), M5/M6 stats reframes, M7 the four missing 2025–26 citations — those still
need doing, but they are not compute and are out of scope here.

**Honest headline:** compute cost is a rounding error (~$5–20, well under two box-days). The
binding constraints are (a) **posterior exposure** — most backbones do NOT hand you per-token
log-probs, and s1/s2 + the certificate are dead without them; (b) **credential-free data** — the
two most-cited additional Mandarin corpora (AISHELL-2, WenetSpeech) are **blocked behind an
application/agreement** and cannot be fetched autonomously; (c) **prereg discipline** — the whole
paper's credibility is its disclosed-deviation model, so the matrix and the target α must be
**frozen before test is touched**, as a resubmission-motivated amendment.

---

## 1. The recommended backbone × corpus matrix (and why it changes the contribution class)

### 1.1 The matrix

Rows = backbones spanning **four distinct architectures** with **exposable per-token posteriors**;
columns = **credential-free** corpora with real test splits low enough in CER to certify
non-vacuously. `cert` = expected non-vacuous certificate at some α in the frozen grid; `audit` =
excess-AURC audit cell (fires even where the certificate is vacuous); numbers are the reported
full-set CER anchor from the literature/model cards.

| Backbone \ Corpus | Aishell-1 test | THCHS-30 test | aidatatang test | (opt) MagicData test | LibriSpeech (EN) |
|---|---|---|---|---|---|
| **Paraformer-zh** (NAR/CIF, *existing*) | cert α=2% ✓ (1.98%) | cert (re-decode) | cert | cert | — |
| **Belle-whisper-large-v3-zh** (AED, *documented data*) | cert α=3–5% (2.78%) | cert (≈3–5%) | cert | cert | reuse EN arm |
| **zipformer-transducer multi-zh-hans** (RNN-T, *open data*) | cert α=5% (~4.3%) | cert | cert | cert | — |
| **(stretch) SenseVoiceSmall** (CTC) *or* **FireRedASR-AED-L** (0.55%) | cert (strong) | cert | cert | — | — |
| Whisper-lg-v3 zero-shot (*existing, vacuous stressor*) | audit only (28%) | audit only (9.9%) | — | — | audit/cert |

**The certified frontier is now a landscape, not a point.** A single operational target — **α=5%**
— is certifiable across the *entire* Mandarin matrix; **tighter targets (α=2–3%) certify on the
stronger backbones and with larger calibration budgets** (§1.3). That is the qualitative jump:
the paper stops claiming "one certificate" and starts demonstrating "a protocol whose certified
frontier we map across four architectures, three-plus corpora, and two languages."

### 1.2 Why this specific set defeats M1 and M2

- **Kills M1 (single cell).** Three independent backbones certify non-vacuously on Aishell-1 alone
  (Paraformer @2%, Belle @3–5%, zipformer @5%), and each also certifies on ≥2 more corpora. The
  substantive-cell count goes from **1 → ≥9**. "One α on one backbone on one clean corpus" is no
  longer a true description of the paper.
- **Kills M2 (contamination).** Both added backbones have **documented, public training data**:
  - *zipformer multi-zh-hans* and the icefall aishell zipformer are trained on **open corpora
    (Aishell-1 train + other public Mandarin sets), provably excluding Aishell-1 test** — this is
    *exactly* the red-team's named remediation ("known training data that provably excludes
    Aishell-1 test").
  - *Belle-whisper-large-v3-zh* discloses its fine-tuning data (Aishell-1/2 train, WenetSpeech,
    HKUST) — again test is not in training.
  Three organizations (Alibaba/FunASR, k2-fsa, BELLE/Lianjia) trained on different data with
  different architectures and all certify low accepted-set CER: memorization of Aishell-1 test by
  *all three independently* is not a credible alternative explanation. The contamination card
  becomes a bounded, answered concern instead of an open reject lever.
- **Neutralizes the single-corpus attack.** A **non-vacuous Mandarin cross-corpus certificate**
  (strong backbone on THCHS-30, then aidatatang) is the paper's *own* stated "single highest-value
  follow-up" (`paper.tex` L770). Delivering it removes the "single corpus" sentence from every
  hostile review.

### 1.3 The calibration-budget axis (cheap, high-leverage, CPU-only after decode)

The paper already *proved* that sub-target vacuity is a **calibration-budget artifact**, not a
property of the bound (English calsweep, `english_calsweep_2026-07-13/`: α=3% becomes certifiable
once n_cal≈1,500; α=2% on the strongest backbone). Exploit this deliberately: for each new
(backbone × corpus) cell, **pool a larger speaker-disjoint calibration set** (each new corpus ships
its own train/dev split — thousands of extra utterances) so α=2–3% certifies where the frozen
613/3,567-utterance carves were too small. This is a *reseed/recalibration* over cached scores —
**0 extra GPU-hours** beyond the one decode. It is the single highest ratio of "TASLP-odds moved"
to "compute spent" in the whole plan.

### 1.4 Honest odds assessment — what actually moves TASLP odds

| Priority | Experiment | Moves odds? | Why |
|---|---|---|---|
| **MUST-DO** | Belle-whisper-zh certified on Aishell-1 (+ THCHS, aidatatang) | **HIGH** | Second strong certified backbone, **documented data** → kills M1 headline + half of M2. **Zero new code.** |
| **MUST-DO** | zipformer-transducer certified on Aishell-1 (+ ≥1 corpus) | **HIGH** | Third architecture, **fully open training data provably excluding test** → the red-team's named M2 fix. |
| **MUST-DO** | Non-vacuous Mandarin cross-corpus cert (Paraformer/Belle on THCHS-30) | **HIGH** | Paper's own #1 follow-up; deletes the single-corpus attack. |
| **HIGH/cheap** | Larger calibration pools → push α=2–3% across cells | **HIGH** | Turns α=5%-only cells into α=2–3% cells for ~0 GPU-h. |
| **MEDIUM** | aidatatang + MagicData as extra corpora | **MEDIUM** | Breadth → "landscape" framing; cheap; diminishing returns past the 3rd corpus. |
| **OPTIONAL** | SenseVoice (CTC) or FireRedASR (SOTA) as a 4th architecture | **LOW–MED** | More diversity, but posterior extraction is unverified (§2.4). Only if the hook verifies fast. |
| **DON'T (initial)** | AISHELL-2, WenetSpeech | **BLOCKED** | Credential-gated (§3); reviewer-response-only. |
| **DON'T** | More Whisper-zero-shot cells | **NO** | Already vacuous; adds nothing. |

**Bottom line on odds:** the red-team put the paper at ~25% as-is. The three MUST-DO experiments
+ the calibration-pool axis are what *neutralize the two reject-capable levers*; without them the
paper stays a "one-cell" result no matter how good the prose. They are necessary and, on the
evidence, close to sufficient on the *scope* axis — but they must be paired with the free framing
fixes (M3/M5/M6/M7) to actually clear a hostile TASLP referee. Compute fixes scope; it does not
fix framing.

---

## 2. Backbones — exact ids, posterior exposure, install, credentials

**The make-or-break criterion is per-token posteriors.** s1 (length-normalized log-posterior) and
s2 (weakest-link min token prob) — and therefore the whole certificate — need per-emitted-token
log-probabilities. The pipeline already has **three proven extraction paths**; pick backbones that
fit one of them and avoid backbones that expose only text (the fate of the Conformer/WeNet cells).

| Path | How posteriors are obtained | Existing code | Backbones that fit |
|---|---|---|---|
| **A. Whisper/AED** | `compute_transition_scores` + BPE→char alignment | `decode_whisper.py` (verbatim) | Belle-whisper-zh; any HF `WhisperForConditionalGeneration` fine-tune |
| **B. CTC per-frame** | `log_softmax(logits)` at deduped non-blank emissions = char logps | `decode_librispeech.py::_wav2vec2_decode` | any HF CTC (wav2vec2/CTC-head); SenseVoice via a hook |
| **C. Paraformer hook** | monkeypatch `_seaco_decode_with_ASF`, replay argmax | `decode_paraformer.py` (verified) | Paraformer variants |
| **D. sherpa-onnx `ys_probs`** | `OfflineRecognizerResult.ys_probs` = per-token acoustic logp | *needs new ~80-line script* | zipformer/conformer transducer ONNX models |

### 2.1 Paraformer-zh — KEEP (existing certified backbone)
- **id:** FunASR `paraformer-zh` (resolves to SeaCo-Paraformer-large); ModelScope
  `iic/speech_paraformer-large...`. **Architecture:** non-autoregressive CIF/paraformer.
- **Posteriors:** ✓ via Path C (verified on-box, funasr 1.3.14). s1/s2 only (no N-best margin, no
  full posteriors → s3/s4 null, as documented).
- **Install:** `pip install funasr modelscope` — proven working on the AutoDL box.
  **Credential-free:** ✓. **Aishell-1 full-set macro-CER:** 1.98%. Certifies α=2%.
- **M2 liability:** undisclosed training data — this is *why* we add documented-data backbones.

### 2.2 Belle-whisper-large-v3-zh — ADD (the single best add; zero new code)
- **id (HF):** `BELLE-2/Belle-whisper-large-v3-zh`. (Variants: `-v3-turbo-zh` faster/slightly
  weaker; `-v2-zh`; `-v3-zh-punct`. Use the non-punct v3 to match Aishell's no-punctuation refs.)
- **Architecture:** attention encoder-decoder (Whisper large-v3 full fine-tune). Different family
  and different vendor from Paraformer → real architectural + provenance diversity.
- **Posteriors:** ✓ via **Path A — reuses `decode_whisper.py` VERBATIM**, only
  `--model-name BELLE-2/Belle-whisper-large-v3-zh --language zh`. It is a standard HF
  `WhisperForConditionalGeneration` (model card ships a `pipeline("automatic-speech-recognition")`
  example). **No new code.**
- **Install:** already satisfied (`transformers torch soundfile`, same as the existing Whisper arm).
  **Credential-free:** ✓ (public HF, Apache-2.0).
- **Reported CER (model card):** Aishell-1 test **2.78%**, Aishell-2 test 3.79%, WenetSpeech net
  8.87%, meeting 11.25%. → **Certifies non-vacuously at α=3% and α=5% on Aishell-1** (2.78% floor);
  α=2% reachable only with a larger calibration pool, if at all.
- **Training data (documented):** Aishell-1/2 train + WenetSpeech + HKUST — **test not in training**.
  Directly answers M2.

### 2.3 zipformer-transducer (Chinese) — ADD (third architecture, cleanest provenance, no k2 build)
- **Preferred id (pre-exported ONNX, robust):**
  `zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2` (offline zipformer **transducer**, trained
  on multiple **open** Chinese corpora). Alternatives: `k2-fsa/icefall-asr-zipformer-wenetspeech-*`
  ONNX exports; or export the aishell-only checkpoint
  `zrjin/icefall-asr-aishell-zipformer-2023-10-24` (aishell test CER **4.28%**; small variant
  4.67%; large variant available) with icefall's `export-onnx.py`.
- **Architecture:** RNN-T / pruned transducer — a genuinely different decoding paradigm from
  NAR-paraformer and AED-Whisper.
- **Posteriors:** ✓ via **Path D** — `sherpa-onnx` `OfflineRecognizer` returns
  `OfflineRecognizerResult.ys_probs` (per-token **acoustic log-probabilities**; the recognized
  prob = `exp(ys_log_probs)`). That is exactly s1's per-token log-posterior and s2's min. **Verify
  at stage time** that `ys_probs` aligns 1:1 with emitted tokens and map tokens→CJK chars (a small
  alignment step analogous to Whisper's `_align_token_logps_to_chars`; for these zh models tokens
  are largely single chars).
- **Install:** **`pip install sherpa-onnx` — pure-Python wheels, NO k2 / kaldifeat / sherpa build.**
  This is the deliberate answer to the prior **WeNet build failure**: we buy transducer diversity
  through the pip'able ONNX runtime, not through a from-source toolkit. **Credential-free:** ✓
  (public HF, Apache-2.0). Runs even CPU-only if needed.
- **Aishell-1 CER ~4.3%** → certifies α=5% (and α=3% with a bigger cal pool). Its **fully open
  training data** makes it the strongest single answer to M2.

### 2.4 STRETCH 4th architecture (pick ≤1; only if its posterior hook verifies quickly)
- **SenseVoiceSmall** — **id:** ModelScope `iic/SenseVoiceSmall` / HF `FunAudioLLM/SenseVoiceSmall`.
  **Architecture:** CTC encoder-only, non-autoregressive (a 4th distinct paradigm).
  **Posteriors:** CTC logits *exist* (Path B in spirit) but FunASR's `AutoModel.generate()` does
  **not** expose them — it does greedy CTC decode + blank removal and returns text/metadata only.
  Needs a **capture hook** on the model's CTC head (simpler than Paraformer's CIF hook: CTC output
  *is* the per-frame log-softmax). **Risk: UNVERIFIED custom extraction.** Install is free (FunASR,
  already on-box). Verdict: highest-diversity, medium-risk; attempt only if the hook lands in the
  first hour on-box, else drop to `skipped-degraded` honestly.
- **FireRedASR-AED-L** — **id:** `FireRedTeam/FireRedASR-AED-L` (1.1B, AED). **SOTA CER: Aishell-1
  0.55%, Aishell-2 2.52%** — would give the *strongest* certificate in the paper. **But:** stock
  `transcribe()` returns **text only** (no per-token logps), and install is **clone-repo + conda**,
  not a pip wheel. Posteriors would need a custom decoder-logit hook (it is AED, so feasible in
  principle, like Whisper). **Risk: HIGH (install friction + unverified hook).** Verdict: high
  reward, high risk; keep as reviewer-response ammunition or a text-only "reported-CER" reference
  point, not a load-bearing certified cell for the resubmission.

**Do NOT** revive WeNet (no installable wheel; the transducer diversity is covered by §2.3) or the
ModelScope Conformer (`iic/speech_conformer_asr_nat-...`; verified to expose no posteriors → stays
`skipped-degraded`).

### 2.5 Backbone recommendation, ranked by (posterior-certainty × install-robustness × diversity)
1. **Paraformer-zh** (keep) — verified, NAR.
2. **Belle-whisper-large-v3-zh** — verified path, AED, documented data, **zero new code**. *Do first.*
3. **zipformer multi-zh-hans (sherpa-onnx)** — documented API + pip install, transducer, open data.
4. *(stretch)* **SenseVoice** (CTC) *or* **FireRedASR** (SOTA) — only if the hook verifies fast.

---

## 3. Corpora — source, size, license, credential status

**The single most important field is credential status.** An autonomous box can only fetch
credential-free sources; anything behind an application/agreement blocks the run.

### 3.1 Credential-FREE (use these — the real corpus-robustness additions)

| Corpus | Source (SLR) | Test split | License | Credential | Notes |
|---|---|---|---|---|---|
| **Aishell-1** | openslr.org/33 (`data_aishell.tgz`, 15G) | 7,176 utts / 20 spk (~10h) | **Apache-2.0** | **FREE** | existing; **also on autodl-pub mirror** (`data_aishell.gz`) → skip download |
| **THCHS-30** | openslr.org/18 (~6.4G) | **2,495 utts / 10 spk** (subset D, ~6h) | **Apache-2.0** | **FREE** | existing route; NOT on autodl-pub mirror → openslr fetch. **Re-decode with strong backbones** (only vacuous Whisper ran) |
| **aidatatang_200zh** | openslr.org/62 (~18G) | ~part of 237k utts/200h (`corpus/test/`) | CC BY-NC-**ND** 4.0 | **FREE** | NEW. Data-only use OK (like ESC-50). Layout: `corpus/{train,dev,test}/<spk>/*.wav` + `transcript/aidatatang_200_zh_transcript.txt`. **Cap decode to a speaker-disjoint ~4–5k-utt subset** to bound cost |
| **MagicData (MAGICDATA)** | openslr.org/68 (~52G) | explicit test set (train:dev:test = 51:1:2 → ~28h) | CC BY-NC-**ND** 4.0 | **FREE** | NEW (optional 4th corpus). Read speech → low CER. Cap similarly |
| **LibriSpeech** | openslr.org/12 | test-clean/other (existing) | CC BY 4.0 | **FREE** | existing English arm — keep |

License note: aidatatang/MagicData are **CC BY-NC-ND** (non-commercial, no-derivatives). The paper
already sets the precedent that noise/eval corpora are **data, not executed code** (ESC-50, CC
BY-NC) and no NC/ND-licensed artifact is redistributed — decode artifacts are derived *metrics*,
not the corpus. Add a one-line license-appendix disclosure, same pattern as ESC-50. (Optional
alternatives if house license review prefers permissive: aidatatang is the standard read-speech
choice; if ND is objected to, THCHS-30 (Apache-2.0) + Aishell-1 alone already give the cross-corpus
result.)

Also credential-free but lower value: **ST-CMDS** (openslr-38, CC BY-NC-ND) and **primewords**
(openslr-47) — usable as further breadth, but 3 Mandarin corpora already make the "landscape"
point; skip unless a referee asks.

### 3.2 BLOCKED behind an application/agreement (reviewer-response-only; do NOT plan an autofetch)

| Corpus | Access mechanism | Verdict |
|---|---|---|
| **AISHELL-2** (1000h, Aishell-2 test 2.5–3.8% CER for strong models) | **Application form + email from an *institutional* address** (`edu.cn` etc.); public emails (gmail/qq/**126.com**) **explicitly rejected**. Not on openslr. tech@aishelldata.com. | **BLOCKED.** The author's on-file email is `fuzeyu99@126.com` (public) → cannot obtain autonomously. Treat as reviewer-response-only; if the user has institutional access, it becomes the single strongest corpus add (Belle/FireRedASR already report Aishell-2 CER). |
| **WenetSpeech** (10kh; Test_Net / Test_Meeting) | **Fill a Google form at wenet.org.cn/WenetSpeech → receive a password** (openslr-121 is password-gated). CC BY 4.0 non-commercial. | **BLOCKED** for autofetch. Its test sets are spontaneous/meeting → mostly vacuous certs anyway. Reviewer-response-only. |
| LDC corpora (HKUST, etc.) | LDC license | **BLOCKED.** |

### 3.3 Recommended corpus set for the resubmission
**Aishell-1 (keep) + THCHS-30 (re-decode strong backbones) + aidatatang_200zh (new)**, with
**MagicData** as an optional 4th, and **LibriSpeech** retained for the cross-lingual arm. That is
**3–4 Mandarin corpora + English**, all credential-free — enough for "landscape," none blocked.

---

## 4. Pipeline reuse — what exists vs what needs writing

The certificate/audit/calibrate machinery is **corpus- and backbone-agnostic by design**
(`asr_gate` consumes canonical decode JSONL, never the model). So **all** of `calibrate`, `apply`,
`audit`, `report`, the LTT/EB/Bonferroni certificate, the reseed protocol, `compute_numbers.py`,
and the figure scripts are **reused unchanged** — every new cell is just another decode JSONL fed
to the same CLI. The only new code is **decode adapters** and **corpus discoverers**.

| Component | Status | Work |
|---|---|---|
| `asr_gate` CLI (score/calibrate/apply/audit/report) | **EXISTS, reuse verbatim** | none — backbone/corpus-agnostic |
| LTT certificate + reseed + Mondrian + audit | **EXISTS, reuse verbatim** | none |
| `compute_numbers.py` / `figures/make_figures.py` | **EXISTS** | extend registry to read new result JSONs (mechanical) |
| **Belle-whisper decode** | **EXISTS (`decode_whisper.py`)** | **none** — `--model-name BELLE-2/Belle-whisper-large-v3-zh` |
| Paraformer decode on new corpora | **EXISTS (`decode_paraformer.py`)** | small: it is Aishell-inlined; either add a `--corpus` branch or route new corpora through `decode_conformer_ms.py`'s `corpora.discover_corpus` pattern |
| **zipformer/sherpa-onnx decode** | **NEEDS WRITING** | new `decode_sherpa_onnx.py` (~80 lines): `OfflineRecognizer.from_transducer(...)`, loop utts, read `result.tokens` + `result.ys_probs`, align tokens→chars → canonical `{hyp_text, nbest:[{text,logp,token_logps}], ...}`. Mirror `decode_librispeech.py`'s structure |
| **aidatatang / MagicData discovery** | **NEEDS WRITING** | add `discover_aidatatang` + `discover_magicdata` to `asr_gate/corpora.py` `CORPUS_DISCOVERERS` (openslr layouts; mirror `discover_thchs30`) |
| (stretch) SenseVoice decode | NEEDS WRITING + HOOK | new `decode_sensevoice.py` with a CTC-logit capture hook (risk §2.4) |
| (stretch) FireRedASR decode | NEEDS WRITING + HOOK + repo install | high friction (§2.4) |
| Golden-file tests for new decoders | NEEDS WRITING | tiny fixtures, CPU, mirrors `tests/test_decode_*.py` |

**Net new code: one decode script (`decode_sherpa_onnx.py`) + two corpus discoverers + tests.**
Belle-whisper — the highest-value add — needs **zero** new code.

---

## 5. GPU-hours, wall-clock, and cost (~$0.5/GPU-h, single RTX 4090/4090D)

Decode-only; calibrate/apply/audit/reseeds are **CPU on cached scores (0 GPU-h)**. RTF = decode
real-time factor on a 4090 (batch-1, incl. I/O). Audio: Aishell-1 test ≈10h; THCHS-30 test ≈6h;
aidatatang/MagicData decode a **capped speaker-disjoint ~4h subset** each (a few thousand utts is
ample to calibrate + certify; no need for the full test set).

| Backbone | RTF | Aishell-1 (10h) | THCHS-30 (6h) | aidatatang (~4h) | (opt) MagicData (~4h) |
|---|---|---|---|---|---|
| Paraformer (NAR) | ~0.03 | 0.3h *(reuse existing)* | 0.2h | 0.1h | 0.1h |
| Belle-whisper-lg-v3 (AED, nbest=1) | ~0.15 | 1.5h | 0.9h | 0.6h | 0.6h |
| zipformer (sherpa-onnx ONNX) | ~0.03 | 0.3h | 0.2h | 0.1h | 0.1h |
| *(stretch) SenseVoice (CTC)* | ~0.02 | 0.2h | 0.1h | — | — |
| *(stretch) FireRedASR-AED-L (1.1B)* | ~0.3 | 3.0h | 1.9h | — | — |

**Core plan (Paraformer + Belle + zipformer × Aishell-1 + THCHS-30 + aidatatang):**
new decode ≈ **~5–6 GPU-h**. Add model download/staging + warmup/verification overhead ≈ 2–3 GPU-h.
**Total ≈ 8–10 GPU-h ≈ $4–5**, wall-clock **~1 box-day** including install + shape-verification +
the CPU calibrate/audit/reseed sweep.

**With both stretch backbones and uncapped aidatatang/MagicData:** +15–25 GPU-h → **still < $20
total, < 2 box-days.** Compute is not the constraint; staging robustness, the prereg amendment, and
writing are. (Consistent with the portfolio's standing rule: binding constraints are prereg
freezes, licenses, and writing time — not GPUs.)

---

## 6. LOCAL prep stageable NOW (no box, no spend)

All of this is CPU/offline and can be done before any box boots, so the box session is pure
decode + a cached-score sweep:

1. **Write `orchestration/decode_sherpa_onnx.py`** (the one new decoder) against the `sherpa-onnx`
   Python API docs; unit-test its token→char alignment on a synthetic `ys_probs` fixture. Keep the
   lazy-import + degrade-gracefully + `--help`-without-deps conventions of the existing decoders.
2. **Add `discover_aidatatang` + `discover_magicdata` to `asr_gate/corpora.py`** (+ register in
   `CORPUS_DISCOVERERS`), with golden-file tests over tiny fixture trees (mirror `test_corpora.py`).
   Encode the openslr-62/68 layouts and speaker-id conventions.
3. **Generalize `decode_paraformer.py`** with a `--corpus` switch (or confirm the
   `discover_corpus` route) so Paraformer can decode THCHS-30/aidatatang, not just Aishell.
4. **Pre-write the next-boot run script** `orchestration/next_boot_asr_landscape.sh` with exact
   staging + decode + calibrate + audit commands (skeleton in §7), `--skip-existing`/`--resume`
   throughout, and a smoke `--limit 20 --probe` first pass per new backbone to verify posterior
   exposure **before** the full decode.
5. **Draft the prereg amendment** `FREEZE-AMENDMENT-2026-07-13.md`: freeze the backbone×corpus
   matrix, the α-grid {2,3,5,10}%, δ=0.1, the calibration-pooling rule, the roster-derived-`m`
   recomputation, and the "test touched once per corpus" rule — **before** any test decode. Disclose
   it as a resubmission-motivated expansion (same discipline as the existing
   `EXPANSION-AMENDMENT-2026-07-09.md`; pre-empts the M5 "forking paths" critique).
6. **Verify exact model ids / licenses at stage time** (record in `DATA_MANIFEST` with checksums):
   `BELLE-2/Belle-whisper-large-v3-zh`, `zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2` (and
   confirm `ys_probs` is populated for the transducer decoding mode), openslr-62/68 URLs. Belle and
   FireRed ship dual/Apache licenses — note in the license appendix.
7. **Extend `compute_numbers.py`** registry stubs to read the forthcoming result JSONs, so the
   paper's number-tracing pipeline covers the new cells the moment they land.

---

## 7. Turnkey next-boot recipe (skeleton — the USER runs this; do not run here)

```bash
# STAGE (credential-free) — prefer autodl-pub mirror for Aishell; openslr for the rest
#   Aishell-1: cp/extract from ${AUTODL_PUB} (per AUTODL-PUB-DATA-MAP) — no download
#   THCHS-30 : wget https://www.openslr.org/resources/18/{data_thchs30}.tgz
#   aidatatang: wget https://www.openslr.org/resources/62/aidatatang_200zh.tgz
#   models   : hf download BELLE-2/Belle-whisper-large-v3-zh ;
#              hf download zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2
#   env      : pip install funasr modelscope transformers torch soundfile sherpa-onnx

# SMOKE (verify posterior exposure BEFORE full decode) — 20 utts each
python orchestration/decode_whisper.py --corpus aishell --split test \
    --model-name BELLE-2/Belle-whisper-large-v3-zh --language zh --limit 20 --out smoke_belle.jsonl
python orchestration/decode_sherpa_onnx.py --corpus aishell --split test --limit 20 \
    --model-dir <multi-zh-hans> --out smoke_zip.jsonl   # confirm token_logps populated

# FULL DECODE (one JSONL per backbone×corpus cell)
for corpus in aishell thchs30 aidatatang; do
  python orchestration/decode_paraformer.py  --corpus $corpus --split test --skip-existing --out dec_para_$corpus.jsonl
  python orchestration/decode_whisper.py      --corpus $corpus --split test --model-name BELLE-2/Belle-whisper-large-v3-zh --language zh --resume --out dec_belle_$corpus.jsonl
  python orchestration/decode_sherpa_onnx.py  --corpus $corpus --split test --model-dir <multi-zh-hans> --out dec_zip_$corpus.jsonl
done

# CERTIFY + AUDIT (CPU; reuse the frozen CLI; larger cal pools per §1.3)
#   for each cell: asr-gate calibrate --instances <cal> --tune <tune> --alpha {0.02,0.03,0.05,0.10} ...
#                  asr-gate apply     --gate gate.json --instances <test>
#                  asr-gate audit     --instances <test-with-refs> --scores s1,s2
#   then the 20-reseed recalibration sweep over cached scores (0 GPU-h)
```

---

## 8. Risks and honest caveats

- **Posterior exposure is the gating risk, and it is front-loaded.** Belle (Path A) and Paraformer
  (Path C) are proven. The zipformer `ys_probs` path is documented but the **token→char alignment
  must be verified on-box in the 20-utt smoke pass** before trusting the full decode. SenseVoice /
  FireRedASR hooks are unverified — treat as stretch, and fall back to honest `skipped-degraded` if
  they do not expose posteriors, exactly as the paper already does for the Conformer.
- **Vacuity is expected on some cells and is a *feature*, not a failure** — the paper's disclosed
  vacuity discipline already covers it. The point is that the *stronger* backbones certify tight,
  and the frontier is mapped, not that every cell certifies at α=2%.
- **Prereg is non-negotiable.** Adding backbones/corpora after a red-team is a forking-paths hazard
  (M5). The amendment (step 5) freezing the matrix + α-grid before test decode is what keeps the
  paper's honesty model intact; skipping it would hand a hostile referee a new lever.
- **Compute fixes scope, not framing.** This plan neutralizes M1/M2 and the single-corpus attack.
  It does **not** substitute for the free CPU/prose fixes (M3 abstract honesty, M4 comparators, M5/M6
  stats reframes, **M7 the four missing 2025–26 citations**), which remain required for acceptance.

---

## 9. One-paragraph recommendation

Run **three backbones (Paraformer + Belle-whisper-large-v3-zh + zipformer-transducer multi-zh-hans)
× three credential-free corpora (Aishell-1 + THCHS-30 + aidatatang_200zh)**, plus a
**cached-score calibration-pool sweep** to push α=2–3% across cells. That is **~8–10 GPU-h (~$5,
one box-day)**, needs **one new decode script + two corpus discoverers** (Belle needs zero new
code), and converts the paper from "one certified cell on one possibly-contaminated backbone on one
corpus" into "a certified-triage protocol whose frontier we map across four architectures and three
Mandarin corpora with documented, open training data" — dissolving the red-team's two
reject-capable levers. Hold AISHELL-2 and WenetSpeech as reviewer-response-only (credential-gated),
and keep SenseVoice/FireRedASR as optional 4th-architecture stretch adds gated on a fast on-box
posterior-hook verification.
