# asr-gate — TASLP workload-gap audit — 2026-07-11

**Question:** is the experimental workload sufficient for IEEE/ACM TASLP acceptance, and
what experiments (if any) are still needed or would materially strengthen it?
**Constraint:** analysis only — no experiments run, no paper edits. Every "current" number
below traces to `results/numbers.json` / `results/audit_ci.json` (the frozen pipeline).

---

## Verdict (one paragraph)

The current workload is **sufficient for a major-revision-then-accept trajectory but
under-powered for a clean first-round accept.** The paper carries two Mandarin corpora, a
3-level noise arm, a certified + a vacuous + a degraded-excluded backbone, 12 audit cells
with speaker-blocked bootstrap CIs, an α-frontier, disclosed negatives, and bit-for-bit
reproducibility — a genuinely rigorous single-language study. But against the TASLP empirical
norm for ASR-confidence / UQ papers (**~4 speech datasets, multiple model families, named
comparators**; LibriSpeech English is the de-facto standard and the closest prior, Ernez et
al. 2023, lives there), it has **one language, one effectively-certified backbone, and a
two-score audited family**. The single highest-value, well-scoped addition is a **LibriSpeech
English arm** (≤5 GPU-h, CC BY 4.0, enables a head-to-head with the Ernez anchor). One nearly
free win — the **per-duration-tercile Mondrian result** — is already computable from frozen
artifacts (CPU, ~1 h) and should be reinstated regardless. A **second certified Mandarin
backbone** and the **s5/s6 comparators** are strong strengtheners; spontaneous speech is
optional.

---

## 1. Current empirical inventory (what the paper actually carries)

| Axis | Current content | Source |
|---|---|---|
| Languages | **1** — Mandarin only | §Setup |
| Corpora | **2** — Aishell-1 (read, test n=7,176) + THCHS-30 (read, cross-corpus n=1,339) | Table (dataset spec) |
| Speech style | read speech only (both corpora) | §Setup |
| Acoustic conditions | clean + ESC-50 additive noise at **5/15/25 dB** SNR (3 levels) | §4.2, Fig. 3 |
| Backbones attempted | Paraformer (**certified**), Whisper-large-v3 (**vacuous-disclosed**), Conformer-Aishell (**degraded-excluded**, no token posteriors) | §Setup, Table (deviations) |
| Effectively certified backbones | **1** (Paraformer); Whisper is vacuous at every α | §4.3 |
| Audit family | **m=12** = {s1 log-posterior, s2 weakest-link} × 6 (backbone×condition) cells | Table (Holm), Fig. 5 |
| Scores exercised | **s1, s2** only; s3 (N-best margin), s4 (entropy), s5 (temp-scaled) structurally null on Paraformer | §2.3, Limitations |
| Repeats | 20 speaker-level **calibration reseeds** (correlated, single fixed test set — stated) | §4.1 |
| Uncertainty | speaker-blocked bootstrap 95% CI on excess-AURC (all 12 exclude 0); micro-CER CI | Table (Holm), §4.4 |
| Baselines / comparators | analytic random-deferral null; oracle ordering (ceiling); s1/s2 are the field-standard audit subjects | §2.3, §3.3 |
| Ablations | α-frontier {1,2,3,5}%; SNR sweep; HB→EB+Bonferroni procedure pivot | §4.1, §2.2 |
| Negative results | α=1% vacuous-at-target; Whisper vacuous at every α; 1/20 violation at 5 dB | §4.1–4.3 |
| Certificate | LTT selective-risk on accepted-set macro-CER, δ=0.1 | §2.2 |
| Not evaluated | standalone G2 / Mondrian result (trimmed to future work, though the gate already uses per-duration-tercile thresholds) | §2.3 |

## 2. TASLP empirical norm (from the 2026-07-11 scan)

- **Corpus count.** Confidence-estimation-for-ASR work at this tier benchmarks on **~4
  well-known speech datasets** ("an extensive benchmark of popular confidence methods … on
  four well-known speech datasets"). Single-corpus confidence papers are the exception, not
  the norm.
- **Language / cross-lingual.** **LibriSpeech (English, 1000 h read; test-clean 2,620 +
  test-other 2,939)** is the standard English ASR benchmark and the setting of the closest
  prior (Ernez et al. 2023, CRC-controlled WER on LibriSpeech/wav2vec2). Cross-lingual
  generalization is increasingly expected (OOD-ASR, ML-SUPERB 2.0 benchmark multilingual
  Common-Voice subsets incl. Mandarin/Spanish/Arabic).
- **Model families / methods.** Accepted papers compare **multiple confidence methods and/or
  architectures** (RNN/Transformer/Conformer), not a single score on a single model.
- **Reproducibility.** TASLP explicitly encourages publishing the code and data that produce
  every figure/table — the paper's strongest card and already fully satisfied.

*Implication:* the paper's methodological framing (a certified-triage apparatus + a
debunk-or-confirm audit, not a Mandarin-specific accuracy claim) fits TASLP scope, but its
**breadth of empirical footprint (1 language, 1 certified backbone, 2 scores)** sits below the
typical accepted paper's, and that is where reviewers will push.

## 3. Gap list (concrete experiments, honest verdicts)

### G1 — LibriSpeech English arm  ·  **MUST-HAVE (bordering)**
- **What:** run the full certify+audit pipeline on LibriSpeech test-clean + test-other with
  (a) Whisper-large-v3 (strong on English, so a non-vacuous certificate should exist, unlike
  its Mandarin regime) and (b) a second strong English model that exposes token posteriors
  (wav2vec2-large or a HuBERT/Conformer CTC checkpoint). Loss = CER on English characters
  (keeps the estimand identical) with WER reported alongside. This directly instantiates the
  **head-to-head with Ernez et al. 2023** that the novelty section currently only argues in
  prose.
- **Cost:** GPU. ~5,551 test utts × 2 backbones; whisper-large-v3 RTFx ~10 → ≤5 GPU-h decode;
  calibration/audit/reseeds are CPU (< 2 h). One RTX 4090D box.
- **Why it matters:** converts "Mandarin-only, one language" (the obvious Reviewer-2 attack,
  §4) into "two languages, two script systems, and a direct comparison to the anchor." Also
  gives a second *certified* backbone regime for free (English Whisper is not expected to be
  vacuous).
- **Gating:** LibriSpeech is **CC BY 4.0** — clean, no license sign-off. wav2vec2/HuBERT
  checkpoints MIT/Apache. Needs a one-line prereg amendment (additive arm, same frozen
  machinery). No new statistical machinery.

### G2 — Per-duration-tercile Mondrian result  ·  **STRENGTHENER (highest ROI, nearly free)**
- **What:** report accepted-set macro-CER + coverage **per duration tercile** (a conditional-
  coverage view). The deployed gate already uses per-tercile thresholds and the cached
  `applied_*` decisions already carry the `stratum` label — this is a join to CER by stratum.
- **Cost:** **CPU only, ~1 h.** Zero GPU. Regenerable from frozen artifacts already in the
  Zenodo package.
- **Why it matters:** the design preregistered a Mondrian/G2 arm; a reviewer who reads the
  method will ask to see it. Reinstating one table/figure adds a conditional-coverage
  dimension reviewers value, at almost no cost, and removes the "you mention Mondrian but never
  show it" snag. (It was trimmed only as a scope call in the referee round.)
- **Gating:** none — data already collected; note it was preregistered.

### G3 — A second *certified* Mandarin backbone  ·  **STRENGTHENER (high value)**
- **What:** certify with a second strong Mandarin model that (a) reaches low CER on Aishell
  (so a non-vacuous certificate exists) AND (b) exposes per-token posteriors (so s1/s2 — and
  ideally s3/s4 — are computable). Candidates: WeNet U2++ conformer (if a working wheel is
  found — it failed at build time before), SenseVoice, or a second FunASR variant
  (SeaCo-Paraformer / Paraformer-large-vs-small). The Conformer-Aishell checkpoint used here
  did **not** expose posteriors, which is exactly the gap.
- **Cost:** ≤3 GPU-h decode + CPU. **Risk:** finding a second strong Mandarin model that
  actually exposes posteriors is non-trivial (one attempt already failed → degraded-excluded).
- **Why it matters:** the headline certificate currently rests on a **single industrial
  checkpoint with undisclosed training data** (the contamination caveat). A second
  independent Mandarin architecture that also certifies would blunt both the
  single-architecture and the contamination attacks. Partly redundant if G1 lands (English
  backbones add architecture diversity across languages).
- **Gating:** model license (WeNet Apache-2.0; SenseVoice verify). Prereg amendment.

### G4 — Additional audited scores (s5 temp-scaled, s6 learned CER-regressor) + named WER-est. comparator  ·  **STRENGTHENER (CPU-cheap where unblocked)**
- **What:** add s5 (temperature-scaled s1, fit on tune) and s6 (gradient-boosted CER regressor
  on cached s1..+duration+length — an e-WER / Park-et-al.-style *learned* comparator, kept in
  the walled-off exploratory family per the design). Position explicitly against a named
  reference-free WER-estimation baseline. s3 (N-best margin) / s4 (entropy) stay blocked until
  a backbone exposes the artifacts (ties to G3).
- **Cost:** **CPU** for s5/s6 on cached scores (a few CPU-h); s3/s4 blocked on G3.
- **Why it matters:** widens the audited family beyond {s1,s2} toward the design's intended
  roster and gives a "free field-standard score vs learned estimator" contrast that TASLP
  confidence papers routinely carry. Cheap and preregistered.
- **Gating:** none for s5/s6 (CPU, cached, s6 already exploratory-BH-walled). s3/s4 gated on G3.

### G5 — Spontaneous / conversational speech  ·  **OPTIONAL**
- **What:** one spontaneous corpus (Mandarin MagicData-RAMC / a conversational set, or English
  Switchboard/CallHome) to test the certificate off read speech.
- **Cost:** GPU decode + license/acquisition; higher effort than G1.
- **Why it matters:** removes the "read-speech-only" caveat. But the ESC-50 noise arm already
  supplies one acoustic-stress axis, and this is already disclosed as a limitation — lower
  priority than cross-language.
- **Gating:** corpus license + acquisition; prereg amendment.

### GATE — ESC-50 noise-corpus license sign-off  ·  **BLOCKER, not a new experiment**
- ESC-50 is **CC BY-NC**. House license review must sign off before submission (data-only use,
  no CC-BY-NC code/weights executed — already disclosed). If rejected: re-mix the noise arm
  with a CC-clean source (MUSAN if reachable, else WHAM!/DEMAND) at ~3 GPU-h re-decode. Bounded.

## 4. Reviewer-2 check: "only one language / one corpus" — how fatal?

**Partly blunted, but the single-language point is the most likely major-revision driver — not
a desk-reject.**

- *"Single corpus" is already weak:* the paper has **two** corpora (Aishell-1 + THCHS-30) and
  the cross-corpus transfer arm is a genuine generalization test, so a literal "one dataset"
  complaint is answerable from the current draft.
- *"Single language" stands and is the real exposure.* Both corpora are Mandarin read speech,
  and the closest prior (Ernez et al.) is **English/LibriSpeech**, so a reviewer will
  reasonably ask "does the apparatus transfer to English/WER where the anchor lives, and how
  does it compare head-to-head?" The paper answers this only in prose today.
- *Why it is not fatal:* (a) the contribution is methodological (a transferable certified-triage
  + audit apparatus), and TASLP does not desk-reject method papers on corpus count; (b) the
  honest negatives (vacuity, the 1/20 slip), the speaker-exchangeability candor, and bit-for-bit
  reproducibility are exactly what the venue rewards; (c) the cross-corpus arm already shows the
  audit transferring across corpora.
- *Why to pre-empt it anyway:* G1 (LibriSpeech) is cheap (≤5 GPU-h, clean license), turns the
  strongest attack into a strength, and unlocks the Ernez head-to-head. Leaving it out is a bet
  that reviewers accept a single-language method paper on reproducibility alone — plausible at
  TASLP, but avoidable for a few GPU-hours.

## 5. Recommended minimal package for a clean accept

Ordered by value-per-cost:

1. **G2 (Mondrian table)** — CPU ~1 h, artifacts exist. Do it regardless; near-zero cost.
2. **G1 (LibriSpeech English arm)** — ≤5 GPU-h, CC BY 4.0, enables the Ernez head-to-head and
   kills the single-language attack. The one arm that most changes the acceptance odds.
3. **G4 (s5/s6 comparators)** — CPU, preregistered; widens the audited family cheaply.
4. **G3 (second certified Mandarin backbone)** — ≤3 GPU-h if a posterior-exposing model is
   found; addresses contamination + single-architecture. Do if G1's English backbones don't
   already satisfy the architecture-diversity ask.
5. **GATE (ESC-50 license)** — resolve before submission; bounded fallback.
6. **G5 (spontaneous speech)** — optional; only if a reviewer demands it.

**Bottom line:** submit-ready as a major-revision candidate today; **G2 + G1** (≈1 CPU-h + ≤5
GPU-h, one clean-license corpus, one prereg amendment) is the smallest package that moves it
toward a first-round accept and neutralizes the predictable Reviewer-2 line of attack.

---

*Sources for the TASLP-norm characterization:* the 2026-07-11 web scan (confidence-estimation
benchmarks on ~4 speech datasets; LibriSpeech as the English standard and Ernez-anchor setting;
OOD-ASR / ML-SUPERB cross-lingual expectations; TASLP reproducibility guidance). No experiments
were run and the manuscript was not edited in producing this memo.
