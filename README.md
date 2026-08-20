# Certified transcription triage

A distribution-free **accept/defer** gate for Mandarin ASR. The object is not a
new recognizer. Given frozen decode artifacts (N-best hypotheses with
log-probabilities), each utterance is accepted for automatic transcription or
deferred to a human. Learn-then-Test certifies that **accepted-set macro-CER**
is at most a chosen target α at confidence 1−δ, or reports
**vacuous-at-target** instead of silently accepting nothing.

Dataset and code release is forthcoming (no resolvable DOI yet).
Code license: MIT.

## What is certified

- **Accept:** the utterance joins the auto-accepted set. The certificate is a
  bound on that set’s macro-CER (mean of per-utterance character error), not
  on micro-CER and not on a new word-error leaderboard.
- **Defer:** the utterance is held for human review. Deferral is the intended
  remainder, not a recognizer failure.
- **Vacuous-at-target:** no threshold on the calibration grid certifies the
  requested (α, δ). Nothing is auto-accepted at that target. Vacuity is a
  published outcome.

Coverage means the auto-accepted fraction. The signature plot is coverage
versus accepted-set CER.

## Frozen results

On Aishell-1 with a Paraformer backbone at α = 2%, δ = 0.1:

| Quantity | Value |
| --- | --- |
| Correlated-reseed violations | 0/20 |
| Acceptance (coverage) | 85.7–96.2% (mean 90.7%) |
| Accepted-set macro-CER | 1.00–1.53% |
| Full-set macro-CER | 1.98% |
| Reference reseed 0 | 85.7% coverage, 1.00% accepted-set macro-CER |

α = 1% is vacuous-at-target. α = 3% and α = 5% saturate. The 0/20 count is
20 correlated reseeds of one fixed test set (resampled calibration/tune
carve), not 20 independent trials.

Field-standard confidence is audited against analytic random deferral. All
twelve roster cells show positive excess-AURC (0.012–0.051) with
speaker-blocked 95% intervals that exclude zero.

## Where the certificate holds — and where it does not

The certificate is backbone-contingent.

- **Paraformer / Aishell-1:** certifies at α = 2% (binding band α = 1.9%,
  87.8% mean acceptance, 1.09% accepted-set macro-CER).
- **Belle / Aishell-1:** certifies only at α = 5%.
- **Whisper Mandarin:** honestly vacuous at every tight target (Aishell-1
  full-set macro-CER 28.1%; THCHS-30 9.93%).
- **zipformer:** vacuous on every corpus in the landscape — no usable
  posteriors; nothing is imputed.

A clean-calibrated gate transferred to ESC-50 additive noise keeps 0/20
violations at 25 dB and 15 dB, and 1/20 at 5 dB (still within δ = 0.1) at
82.2% mean coverage.

The English wav2vec 2.0 large / LibriSpeech series is a calibration-budget
arm (non-vacuous at α = 5% on the frozen carve). It is not a Mandarin
landscape cell. Spontaneous and multi-domain Mandarin at WenetSpeech scale
is untested.

## Code and data

This repository holds the accept/defer-gate code and the frozen statistics
used to rebuild the figures. Bulk corpus audio is not redistributed: use the
official Aishell-1, THCHS-30, MagicData, LibriSpeech, and ESC-50 sources.
MagicData remains evaluation data under CC BY-NC-ND; the archive stores
derived per-utterance CER metrics, not audio or transcripts.

Author: Zeyu Fu · [ORCID 0009-0001-8329-0108](https://orcid.org/0009-0001-8329-0108)
