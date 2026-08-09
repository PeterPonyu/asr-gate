# RED-TEAM REPORT — asr-gate paper (TASLP target)

**Reviewer stance:** adversarial; goal is rejection. Prepared 2026-07-11 by an
attacker who did not author the paper. Targets: `manuscripts/paper.tex` and the
`manuscripts/taslp/paper_ieeetran.tex` port. No paper edits made.

**Bottom line up front.** I could not land a FATAL blow. Every load-bearing number
I traced (18 of them) matches the frozen JSONs bit-for-bit; the port carries no
content drift; the preregistration documents back up the "disclosed deviation"
claims (including the two I most expected to be post-hoc — the duration-tercile
Mondrian and the noise/cross-corpus arms — which are genuinely frozen in
`FREEZE-NOTE-2026-07-09.md` / `EXPANSION-AMENDMENT-2026-07-09.md`). The paper is
unusually candid. But candor is not contribution, and a hostile TASLP reviewer has
**seven MAJOR levers**, two of which (M1 thin certified contribution, M2 unresolved
contamination of the *only* substantive cell) are individually reject-capable at a
top journal. Ranked findings below.

---

## FATAL — none found

- **Number tracing (18 traced, 0 mismatch).** `0/20` violations, acceptance
  85.7–96.2% (mean 90.7%), accepted-CER 1.00–1.53% (mean 1.21%), full-set 1.98%,
  α=1% vacuous, α=3/5% at 99.8%, noise 0/0/1 violations at 25/15/5 dB, Whisper
  28.06%/15.77%, all 12 excess-AURC point estimates and CIs — every one matches
  `numbers.json` / `holm_audit_realized.json` / `audit_ci.json`. No fabrication.
- **Port drift: clean.** Canonical vs `paper_ieeetran.tex` share all key-claim
  tokens; the only divergent decimals are LaTeX layout constants (0.4, 0.6, 0.62,
  4.5). No content divergence.
- Report this honestly: the provenance discipline is real and defuses the usual
  claim-evidence attack.

---

## MAJOR findings

### M1 — The certified contribution (C1) collapses to a single operating point. [reject-capable]
**Evidence:** paper.tex L283–287, tab:cert (L488–502), `numbers.json:main_alpha_frontier`.
Of the four certified targets: α=1% is `vacuous-at-target` (0% acceptance);
α=3% and α=5% "certify at 99.8%" but the accepted-set macro-CER there is **1.91%**
(`main_alpha_frontier/0.03/test_accepted_macro_cer=0.0191`) ≈ the full-set 1.98% —
i.e. the budget exceeds the full-set error, so *accept-almost-everything* trivially
satisfies it (the paper's own "grid-floor saturation"). Whisper is vacuous at every
α (L391); Conformer is `skipped-degraded` (L243); the noise cells are the same
Paraformer backbone. **So the entire non-trivial certified claim is one cell:
α=2%, Paraformer, Aishell, clean.**
**Hostile-reviewer argument:** "A TASLP paper titled *Certified Transcription
Triage* delivers exactly one substantive certificate — a single α on a single
backbone on a single clean corpus. Everything else is vacuous or accept-all. This is
a workshop-sized result dressed as a journal certification framework."
**Remediation (defend + strengthen):** the highest-value fix is the paper's own
stated follow-up (L543) — run the non-vacuous cross-corpus certificate (strong
backbone across corpora) so C1 has ≥2 substantive cells. Cheaper defense: reframe
the title/abstract around the *audit* (C2, which is broad) and demote C1 to "a
worked, honestly-scoped instantiation." Do not leave C1 as the headline with one cell.

### M2 — Contamination threatens the *only* substantive certified cell, and the defenses don't bound it. [reject-capable]
**Evidence:** paper.tex §Paraformer training-data contamination (L504–515).
Paraformer-large's training data is undisclosed; Aishell-1 test contamination
cannot be excluded. The 1.00% accepted-set CER — the entire C1 headline — could be a
memorization artifact.
**Hostile-reviewer argument:** "The two offered defenses do not bound the concern.
(1) Whisper showing confidence *skill* is irrelevant to whether *Paraformer*
memorized Aishell test — different claim. (2) Monotone CER degradation under noise
is *equally* consistent with contamination: noise progressively breaks a lookup of
memorized clean transcripts. Neither rules out that the halved-error headline is
Aishell-1 leakage. On an industrial checkpoint this is disqualifying for a
*certification* claim."
**Remediation (must-address):** add a backbone with *known* training data that
provably excludes Aishell-1 test (a from-scratch or documented-corpus model), even
if weaker, and show a non-vacuous certificate there; OR run the certificate on a
held-out corpus Paraformer's vendor could not have trained on and report the
accepted-set CER. Disclosure alone (current stance) will not satisfy a hostile referee.

### M3 — "0/20 violations" validates nothing about the distribution-free guarantee; it is one fixed test set. [framing, mitigated by candor]
**Evidence:** paper.tex L312–323 (the paper's own disclosure) vs the abstract L58 and
title, which lead with "0/20 violations" as the validity headline.
**Hostile-reviewer argument:** "The δ=0.1 LTT guarantee is over the *calibration*
draw for the *population* risk. The 20 reseeds resample only calibration and apply
every gate to the *same fixed 7,176-utterance test set*, so (a) they are correlated,
not 20 Bernoulli(δ) trials (conceded), and (b) more damningly, the test-set
accepted-CER is itself an estimate of the population risk with its own sampling
error that 0/20 never bounds. The paper never demonstrates control on a *fresh test
distribution* — the actual content of 'distribution-free.' The headline oversells."
**Remediation (disclose harder + reframe):** the body §5.1 already says this; pull
the caveat *into the abstract* ("on a single fixed test set; a fresh-test-set
guarantee is future work") so the abstract is as honest as the body. Ideally add a
second test corpus for a genuine held-out check (ties to M1).

### M4 — C2's bar ("beats random deferral") is trivially low. [contribution thinness]
**Evidence:** paper.tex L185–195, tab:holm; `audit_ci.json` shows random_expectation
= the full-set mean CER (0.0195) at every coverage — random deferral is a coin flip.
**Hostile-reviewer argument:** "Beating *random* deferral requires only that the
confidence score correlate with error *at all* — a bar any non-degenerate score
clears by construction. Of course length-normalized log-posterior beats a coin flip.
The interesting and decision-relevant comparison — against a *calibrated* confidence,
a length baseline, or a competing selective-prediction method — is absent. The
'debunk-or-confirm' framing manufactures suspense around a foregone conclusion."
**Remediation (add a real baseline):** add ≥1 non-trivial comparator (temperature-
calibrated confidence; an utterance-length baseline; or SelectiveNet/entropy-style
selective baselines) so 'beats random' becomes 'beats the operational alternative.'
CPU-only on the frozen scored artifacts.

### M5 — The preregistered multiplicity apparatus is decorative; the real evidence is post-hoc. [statistics + framing]
**Evidence:** paper.tex L414–418 and tab:holm caption (L446–458): all 12 permutation
p-values sit at the resolution floor (5.0e-4), so Holm "has nothing to discriminate"
(`p_holm=0.006` throughout) — the paper says so. The load-bearing quantities are the
excess-AURC CIs, which the caption itself labels "computed post-hoc from the frozen
scored artifacts and prereg-neutral."
**Hostile-reviewer argument:** "The paper foregrounds a preregistered,
Holm-controlled, permutation-tested family — then concedes that test is saturated and
uninformative, and pivots the actual inference to *post-hoc, prereg-neutral* bootstrap
CIs. So the confirmatory scaffolding (permutation null, Holm, roster-derived m) is
theater; the evidence is an exploratory effect-size analysis. Worse, the family grew
from a 'toothless' m=2 in the main run to m=12 only *after* the main run underwhelmed
(§Method L200–212) — the study design itself is a garden of forking paths, each fork
disclosed but the sequence outcome-driven."
**Remediation (reframe honestly):** state plainly that the permutation test
establishes only 'no cell is null' and that the effect-size CIs are the inference;
drop the Holm-m theater from the headline. Defend the m=2→12 expansion as
preregistered-by-rule (it is — EXPANSION-AMENDMENT §Holm-family recomputation), but
acknowledge the expansion was *motivated* by the underpowered main run.

### M6 — The permutation null is utterance-level while the paper's own exchangeability unit is the speaker. [internal inconsistency]
**Evidence:** paper.tex L191–193 (permutation "re-draws which utterances are deferred")
vs L253–261 ("speaker is the natural block"; all CIs speaker-blocked).
**Hostile-reviewer argument:** "The paper argues speaker-level dependence is
load-bearing and speaker-blocks every CI — then runs an *utterance-level*
permutation null that assumes exactly the exchangeability it says is violated. The
permutation p-values are therefore anti-conservative. That they are floor-saturated
only hides the inconsistency; it doesn't fix it."
**Remediation:** either speaker-block the permutation null too (re-draw deferred
*speakers*/blocks), or drop the permutation test entirely and rely on the
speaker-blocked bootstrap CIs (which are already the stated evidence). Cheap, CPU-only.

### M7 — Missed 2025–26 related work that a TASLP referee will know. [scoop / positioning]
**Evidence:** cited set includes `yu2026joint`, `angelopoulos2026nonmonotone`,
`ernez2023conformal`, `traub2024selective`; refs.bib contains none of the following,
all **pre-freeze** (freeze 2026-07-09; K6 rescan "last run 2026-07-08"):
- **arXiv:2606.15153** — *False sense of safety in selective signal classification:
  auditing bound tightness and exchangeability for risk control.* Directly audits the
  exact exchangeability strain this paper concedes as its main limitation. A referee
  will demand engagement.
- **arXiv:2512.12844** — *Selective Conformal Risk Control (SCRC)*: select confident
  samples, then apply CRC on the accepted subset — the general version of asr-gate's
  accept/defer + risk control.
- **arXiv:2605.20270** — *Conformal Selective Acting*: anytime-valid selective risk
  on a Bonferroni grid — nearly the paper's own machinery.
- **arXiv:2503.22712** — conformal coverage guarantees for (affective) speech
  recognition — the speech+conformal adjacency the paper claims is thin.
**Hostile-reviewer argument:** "The paper claims novelty in selective risk control
for ASR while omitting the 2025–26 selective-conformal-risk-control literature that
generalizes its method and an audit paper aimed squarely at its central caveat. The
K6 scoop scan missed them. Positioning is inadequate for a journal."
**Remediation:** cite and differentiate all four; especially confront 2606.15153
head-on (it strengthens the paper's own honesty narrative if engaged, damns it if
ignored). Re-run the K6 scan at submission.

---

## MINOR findings

- **m1 — 20 speaker blocks make the bootstrap CIs coarse.** (L541) Tight CIs like
  `[0.0101, 0.0137]` are over-precise given 20 test blocks; percentile bootstrap on
  20 blocks under-covers. *Remediation:* report the block count next to each CI and
  soften the precision, or use a block-count-aware interval (e.g., BCa or t-based).
- **m2 — "certified"/"Certificate" in the title is strong for an LTT high-prob bound
  shown on one test set.** Defensible by RCPS/LTT convention, but with M3 it
  over-promises. *Remediation:* keep the term but qualify in the abstract.
- **m3 — HB→EB+Bonferroni pivot: disclosed and defensible (not the p-hack it looks
  like).** Both procedures are finite-sample valid; choosing the more powerful valid
  one does not inflate the selected procedure's type-I error, even if the choice was
  informed by pilot data. *Remediation:* one sentence noting EB is a-priori uniformly
  more powerful than HB for bounded means, so the choice is a power upgrade, not a
  validity risk — this converts a suspicious-looking disclosure into a strength.
- **m4 — Conditional-coverage "0/20 in every tercile" is partly mechanical.** The
  deployed gate is Mondrian-in-duration (frozen — FREEZE-NOTE, so *not* post-hoc),
  which calibrates each tercile separately, so per-tercile control is close to
  by-construction and is weaker independent evidence than it reads. *Remediation:*
  say so; present it as a consistency check, not corroboration.
- **m5 — Frozen filenames read `musan5db/15db/25db` but the corpus is ESC-50.**
  Disclosed (L556–559), but an artifact-skimming referee may misread. *Remediation:*
  a README note in the results dir (already partially present).

---

## Attack-axis coverage (for the coordinator)

| Axis | Verdict |
|---|---|
| 1 Statistics (LTT/HB, family, blocking, 0/20, tercile) | M3, M5, M6; tercile & family preregistration CONFIRMED genuine (freeze docs) |
| 2 Claim–evidence (≥15 traced) | CLEAN — 18/18 trace, 0 mismatch |
| 3 Leakage/design (contamination, noise) | M2 (contamination, reject-capable); noise/ESC-50 disclosed |
| 4 Framing (oversell, "certified", vacuity) | M1, M3, M4 |
| 5 Scoop/related work | M7 (4 pre-freeze papers missed, incl. the exchangeability-audit) |
| 6 Port drift | CLEAN |

## Sources (scoop search, 2026-07-11)
- [False sense of safety in selective signal classification (arXiv:2606.15153)](https://arxiv.org/html/2606.15153v1)
- [Selective Conformal Risk Control (arXiv:2512.12844)](https://arxiv.org/pdf/2512.12844)
- [Conformal Selective Acting (arXiv:2605.20270)](https://arxiv.org/abs/2605.20270)
- [Risk-Calibrated Affective Speech Recognition via Conformal Coverage (arXiv:2503.22712)](https://arxiv.org/html/2503.22712v1/)
