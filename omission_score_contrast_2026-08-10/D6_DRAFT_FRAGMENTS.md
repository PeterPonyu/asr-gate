# D6 — Venue-neutral draft fragments

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Section id** | `crc_omission_contrast` |
| **Status** | **NOT submission-ready** — venue-neutral stubs only |
| **Scope** | asr-gate science home; **0 writes** to `submissions/asr` |

---

> **Banner:** These fragments are packaging drafts under Q5 defer. They use namespaced display scores only. They do **not** extend LTT G1. Do not paste into `submissions/asr` without human Q2 reversal.

---

## Fragment A — Methods (score contrast)

We evaluate group-conditional risk control on frozen Mandarin ASR landscape test decodes when the deploy-time group indicator (`pred_ent`: hyp-side entity assertion) is unavailable and entity omission destroys the evidence the indicator requires. Two score families share the same calibration and omission definition:

- **`s1_crc_decoupled`**: mean 1-best token log-probability (fluency/confidence decoupled from entity assertion).
- **`hyp_entity_mass_coupled`**: hyp-only jieba entity-assertion mass normalized by hyp length (coupled to the same hyp-side observable that defines `pred_ent`).

CRC-style deploy-valid % is computed at α ∈ {0.05, 0.10, 0.20} with speaker-blocked calibration splits (20 seeds). Primary aggregates use the **both-arms deploy-nonvacuous** stratum; **OR deploy-nonvacuous** sensitivity and all-pair medians are reported alongside.

---

## Fragment B — Results (headline numbers)

On 9 landscape cells × 3 α levels (27 cell×α pairs), median deploy-valid Δ (**hyp_entity_mass_coupled** − **s1_crc_decoupled**) was **+0.0** over all pairs. Restricting to pairs where **both** arms are deploy-nonvacuous (`deploy_vacuous_frac < 0.5` on each arm), n=**13** pairs showed median Δ=**+0.6**; effect direction matches the 007 E4 coupled-vs-decoupled polarity on this stratum. OR sensitivity (either arm nonvacuous) retained n=**27**, median Δ=**+0.0**. CPU wall time ≈ **55.2** s on frozen decodes (`code_sha256` pinned in `SUMMARY.md`).

---

## Fragment C — Limitations

Claim strength follows **auto-soft** policy: D2 ablations did not trigger the degenerate-`pred_ent` auto-soft heuristic (both-arms median Δ=0.0 vs headline 0.6; ratio=0.00). Scrambled-POS / leave-one-out POS ablation was **not** run in this CPU session. We therefore characterize the coupled score as a **constructed hyp-side entity mass**, not as proof that “certificate coupling repairs indicators.” AUROC orientation checks are descriptive only (`D7_AUROC_DISCLOSURE.md`); deploy-valid % remains the primary endpoint.

---

## Fragment D — Overlap disclosure

> A related CPU-only analysis (omission / entity severity under group-conditional risk control) was developed in a separate incubator experiment tree and then routed into this repository. It uses the same frozen landscape decode artifacts. The present note reports a pre-registered coupled-vs-decoupled **score** contrast (mean token log-probability vs a hyp-only entity-assertion score) with CRC-style deploy-validity as the primary endpoint. Overlap with the main ASR selective-prediction results is intentional; novelty, if any, is limited to the score-coupling characterization rather than a new corpus or backbone bake-off.

**Non-claims (must not appear as affirmative claims elsewhere):**

- Not a new pursue lane under `directions/007`.
- Not a GPU re-decode or new backbone comparison.
- Not evidence that LTT guarantees are CRC guarantees.

---

## Fragment E — Table stub (namespaced columns)

| corpus | backbone | α | s1_crc_decoupled deploy% | hyp_entity_mass_coupled deploy% | Δ | deploy vac. decoupled | deploy vac. coupled |
|---|---|---:|---:|---:|---:|---:|---:|
| *(see SUMMARY.md per-cell table — display names applied)* | | | | | | | |

Full per-cell values: `SUMMARY.md` § Per-cell deploy-valid (regenerate with display-name column headers before any external paste).

---

## Editorial guards

| Guard | Rule |
|---|---|
| G1 extension | **Forbidden** in all fragments |
| Bare `s1` | **Forbidden** — use `s1_crc_decoupled` |
| Unfreeze implication | **Forbidden** — Q2-reprise = keep frozen |
| Strong repair language | **Forbidden** under Q6 auto-soft default |
