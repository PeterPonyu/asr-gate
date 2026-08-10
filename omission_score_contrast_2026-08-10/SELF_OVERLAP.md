# Self-overlap disclosure — omission / destroyed-indicator CRC (draft)

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Home** | `asr-gate/omission_score_contrast_2026-08-10/` |
| **Status** | Draft under asr-gate only — **not** pasted into `submissions/asr` (Open Q2 = asr-gate-only writes) |

---

## What overlaps

Work in `frontier-directions-research/experiments/007-omission-crc/` (E0–E4) and this
asr-gate CPU contrast study the **same scientific object**:

> Group-conditional risk control when the **group indicator is unobserved at
> deploy time** and is **destroyed by the loss** (omission removes the evidence
> the indicator needs), with the **score family** determining whether the
> certificate remains valid.

Shared substrate with the in-flight ASR reliability / TASLP line:

- Frozen landscape decodes under `asr-gate/landscape_pulled_2026-07-15`
- Same Mandarin corpora / backbones already used in asr-gate landscape audits
- Same severity class (jieba entity POS) as 007 E2/E3

007 did **not** invent a new ASR corpus; it re-analyzed asr-gate (and related)
decode artifacts. This contrast is therefore an **extension of the asr-gate
reliability narrative**, not an independent discovery claim.

---

## What is distinct

| Piece | Where it lives | Claim type |
|---|---|---|
| Aggregate LTT / selective ASR certificates (s1–s5) | asr-gate main package + manuscripts | Primary submission substrate |
| Marginal vs class-conditional CRC on Primewords (E0–E1) | 007 | Mostly archived / definitional |
| Destroyed-indicator mechanism W1–W3 (E2–E3) | 007 on landscape | Mechanism on asr-gate data |
| Score-coupled repair (`conf` vs `decoupled`) on radiology (E4) | 007 | Cross-domain qualifier |
| **This run:** s1 vs hyp-only `hyp_entity_mass` deploy-valid on ASR landscape | **asr-gate** (this dir) | Closes the E4 “run the contrast on ASR too” residual |

This directory is **score contrast orchestration only** — it does not convert
asr-gate’s LTT package into a CRC product.

---

## Disclosure language (for later submission revision)

Suggested paragraph (edit before any `submissions/asr` paste):

> A related CPU-only analysis (omission / entity severity under group-conditional
> risk control) was developed in a separate incubator experiment tree and then
> routed into this repository. It uses the same frozen landscape decode
> artifacts. The present note reports a pre-registered coupled-vs-decoupled
> **score** contrast (mean token log-probability vs a hyp-only entity-assertion
> score) with CRC-style deploy-validity as the primary endpoint. Overlap with
> the main ASR selective-prediction results is intentional; novelty, if any,
> is limited to the score-coupling characterization rather than a new corpus or
> backbone bake-off.

---

## Non-claims

- Not a new pursue lane under `frontier-directions-research/directions/007`.
- Not a GPU re-decode or new backbone comparison.
- Not evidence that asr-gate’s LTT guarantees are CRC guarantees (different objects).
