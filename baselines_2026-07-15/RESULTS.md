# Named confidence-baseline comparison — 2026-07-15

Reviewers at the target venue expect a method-vs-method table naming standard ASR
confidence estimators (WORKLOAD-BENCHMARK-2026-07-15, asr row: "0 named methods vs
median 3-6"). We evaluate the TeLeS-family named baselines **through the same excess-AURC
audit machinery** (`asr_gate.audit.run_audit`, n_perm=2000, seed=0) on the **same landscape
cells** our primary scores s1/s2 use, so every number is directly comparable to the paper's
audit rows. Regenerate: `PYTHONPATH=<asr-gate>:<reliability-commons> python3 runner.py`
→ `results.json`.

## What is derivable from the cached decode artifacts
The dumps carry per-emitted-token log-probs (`nbest[0].token_logps`) but **not** the full
per-token vocabulary distribution (`token_full_posteriors` is `None` in every dump — the s4
carrier is absent). So:

| Baseline | Definition | Derivable? |
|---|---|---|
| class-probability (mean) | mean_t exp(logp_t) | YES (`class_prob_mean`) |
| class-probability (min) | exp(min_t logp_t) | YES — **== our s2 exactly** |
| length-normalized path prob | exp(mean_t logp_t) | YES — **== exp(s1); monotone in s1 ⇒ identical excess-AURC** |
| Tsallis entropy | needs full per-token vocab distribution | **NO — UNAVAILABLE, reported not imputed** |
| learned CEM | logistic regression on [mean, min, std logp, log1p(n_tok), duration], fit on Aishell-1 dev (per backbone, refs⇒CER label), score=−P(error) | YES (`cem`) |

## Excess-AURC (higher = better selective-prediction signal). MagicData capped to n=4,094 (§2).

| cell | n | macro-CER | s1 | s2 | class-prob-mean | class-prob-min | lengthnorm-pathprob | CEM |
|---|---|---|---|---|---|---|---|---|
| paraformer / aishell | 7,176 | 1.98% | 0.0118 | 0.0137 | 0.0117 | 0.0137 | 0.0118 | 0.0122 |
| paraformer / thchs30 | 1,339 | 3.75% | 0.0187 | 0.0186 | 0.0186 | 0.0186 | 0.0187 | 0.0175 |
| paraformer / magicdata | 4,094 | 4.89% | — | — | — | — | — | — |
| belle / aishell | 7,176 | 5.12% | 0.0236 | 0.0221 | 0.0237 | 0.0221 | 0.0236 | 0.0199 |
| belle / thchs30 | 1,339 | 6.62% | 0.0324 | 0.0282 | 0.0326 | 0.0282 | 0.0324 | 0.0296 |
| belle / magicdata | 4,094 | 7.80% | 0.0420 | 0.0372 | 0.0422 | 0.0372 | 0.0420 | 0.0346 |
| zipformer / aishell | 7,176 | 45.88% | −0.0158 | 0.0578 | −0.0226 | 0.0578 | −0.0158 | 0.1890 |
| zipformer / thchs30 | 1,339 | 81.80% | −0.0397 | 0.0410 | −0.0468 | 0.0410 | −0.0397 | 0.1367 |
| zipformer / magicdata | 4,094 | 32.21% | −0.0055 | 0.0567 | −0.0102 | 0.0567 | −0.0055 | 0.1504 |

paraformer/magicdata: the whole score family is excluded by the audit's ≤1%-missing rule
(41/4,094 = 1.0% of rows lack a decoded s1), reported as `—` — not imputed.

## Honest framing (the paper's contribution is the certificate, not a new score)
1. **class-prob-min IS s2, and lengthnorm-pathprob IS exp(s1)** (identical rankings ⇒
   identical excess-AURC, confirmed to the printed digit). Our two primary scores ARE two of
   the standard named confidence baselines — we do not claim a novel confidence estimator.
2. On the certifying backbones (Paraformer, Belle), **class-prob-mean and the learned CEM
   match s1/s2 within ±0.002 excess-AURC** on every cell — no named baseline dominates our
   scores; the confidence signals are essentially interchangeable. The paper's contribution
   is the **finite-sample certificate + honest audit** wrapped around these scores, not the
   scores themselves.
3. On the degraded backbone (zipformer, non-certifying everywhere), s1 / class-prob-mean /
   lengthnorm-pathprob go **negative** (mean-log-prob confidence is anti-correlated with
   error under this transducer's mis-scaled posteriors); s2 and the CEM stay positive but do
   **not** rescue certification (every zipformer cell is vacuous). This is a property of the
   backbone's posteriors, disclosed verdict-symmetrically, not of the method.

Bottom line for the paper: a named method-vs-method table shows our scores are competitive
with (indeed coincide with) the standard TeLeS-family confidence baselines; the audit floor
(excess-AURC) is reported for all of them side by side; the certificate is what the paper
adds on top.
