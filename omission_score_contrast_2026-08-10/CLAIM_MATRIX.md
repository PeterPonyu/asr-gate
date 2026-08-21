# Claim matrix — CRC omission contrast (Step 1 firewall)

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Owner** | asr-gate (`omission_score_contrast_2026-08-10/`) |
| **Plan** | `ralplan-asr-gate-paper-packaging` Step 1 |
| **Status** | Active firewall — binds all downstream packaging and writing |

---

## Three completion predicates (must stay separate)

These predicates are **not substitutable**. Satisfying one does **not** imply any other.

| Predicate | Meaning | Who may assert |
|---|---|---|
| **`packaging-complete`** | Steps 1–3 + D1–D3 AC green | Executor verify after hard gate |
| **`CLEAR-for-packaging`** | Optional Architect upgrade after `packaging-complete` | Architect / human only |
| **`unfreeze-authorized`** | Edit `submissions/asr` or merge appendix into frozen kit | **Human Q2 reversal only** |

**Forbidden chain (non-implication):**

```
packaging-complete  ⇏  CLEAR-for-packaging  ⇏  unfreeze-authorized
```

No downstream artifact, status banner, or summary line may collapse these into a single “done” flag.

---

## Explicit FORBIDDEN inferences

The following inferences are **blocked** regardless of empirical outcome (polarity PASS, coupled ≥ decoupled, etc.):

| Forbidden inference | Why blocked |
|---|---|
| **first-wave complete ⇒ paper-ready** | First-wave closes portfolio routing residual, not claim maturity |
| **NEAR-READY (science) ⇒ paper-ready** | Science ~90 / writing ~40; D1–D3 and firewall not yet in manuscript |
| **packaging-complete ⇒ paper-ready** | Packaging = empirics + pins + packet; venue writing and Q5 unset |
| **packaging-complete ⇒ unfreeze-authorized** | Q2 default = keep `submissions/asr` frozen |
| **CLEAR-for-packaging ⇒ unfreeze-authorized** | Architect clearance ≠ human Q2 reversal |
| **first-wave complete ⇒ unfreeze** | Portfolio residual closed does not authorize kit edits |
| **NEAR-READY ⇒ unfreeze** | Empirical packable ≠ submission-ready |

Any report, canvas, README breadcrumb, or team handoff that uses “done”, “ready”, or “complete” must name **which predicate** is meant.

---

## LTT ↔ CRC claim firewall

| Object | Domain | May claim |
|---|---|---|
| **LTT G1** (aggregate selective ASR certificates, s1–s5) | Main asr-gate / TASLP manuscript substrate | Marginal / class-conditional **LTT** coverage guarantees |
| **CRC deploy-valid** (this contrast) | Appendix-only `crc_omission_contrast` section | Group-conditional **CRC-style** deploy-valid % under destroyed-indicator omission |

**Hard rules:**

1. **LTT G1 ≠ CRC deploy-valid** — different estimands, calibration objects, and guarantee types.
2. **Do not narrate CRC deploy-valid as an LTT G1 extension** — no “strengthens G1”, “extends the main certificate”, or “confirms LTT coverage under omission”.
3. **Do not merge CRC contrast into LTT body claims** without namespace + overlap-disclosure boilerplate and human Q2.
4. Read-only SSOT for E4 polarity: `frontier-directions-research/experiments/007-omission-crc/` — **no writes** to that tree.

See also: `DISPLAY_NAMESPACE.md` (score naming).

---

## Display score namespaces (appendix contract)

Internal pipeline keys **must not** appear bare in appendix tables or manuscript fragments.

| Internal key | Appendix display name | E4 analogue |
|---|---|---|
| `s1` | **`s1_crc_decoupled`** | `decoupled` |
| `hyp_entity_mass` | **`hyp_entity_mass_coupled`** | `conf` / coupled |

**Section id:** `crc_omission_contrast`

**Ban:** bare `s1` in appendix tables, figures, and D4–D6 draft fragments. The token `s1` is reserved for the main LTT package; reuse here requires the `_crc_decoupled` suffix in all outward-facing surfaces.

---

## Vacuity stratification contract

Dual vacuity fields (target schema after Step 2 code refresh):

| Field | Side | Meaning |
|---|---|---|
| `oracle_vacuous_frac` | Oracle / calibration-side | Fraction of seeds where oracle-side flag is degenerate (0 or 1) |
| `deploy_vacuous_frac` | Deploy / evaluation-side | Fraction of seeds where deploy-side flag is degenerate |

**Primary stratum — both-arms nonvacuous (headline):**

A cell×α pair enters the **primary** aggregate iff **both** arms satisfy nonvacuous on their respective sides, e.g.:

```
oracle_vacuous_frac < 0.5  AND  deploy_vacuous_frac < 0.5   (per arm, both arms)
```

Exact thresholds and field names follow `MAPPING.md` provenance once Step 2 re-run lands.

**Sensitivity stratum — OR nonvacuous (retained, not headline):**

A pair enters the **OR sensitivity** row iff **either** arm is nonvacuous:

```
nonvacuous_s1 OR nonvacuous_coupled
```

OR sensitivity **must remain visible** alongside the both-arms headline (legacy SUMMARY used OR; Step 2 tables show both).

**Honesty requirements:**

- All-pair median Δ=0 must remain visible (not hidden by stratum choice).
- Vacuous cells are reported, not dropped silently.
- Do not rename-only “fix” vacuity without full re-run when field semantics change.

---

## Claim strength default (pending D2)

| State | Wording allowed | Wording blocked |
|---|---|---|
| **Auto-soft (default until D2 passes)** | “constructed hyp-side entity mass”; “score-coupling contrast on ASR landscape”; “CRC deploy-valid differs by score family under omission” | “certificate coupling **repairs** indicators”; “coupling **fixes** destroyed-indicator validity”; any causal repair language |
| **Strong (only after D2 ablations green)** | Score-family coupling characterization with ablation-backed non-tautology | Still blocked: LTT G1 extension; unfreeze implication |

**D2 auto-soft heuristic (operational):**

If degenerate `pred_ent` score achieves ≥80% of coupled−decoupled median Δ on the **both-arms nonvacuous** headline set, **or** coupled advantage disappears under scrambled-POS / POS LOO, then:

1. Commit **auto-soft** claim strength (Q6 default = accept auto-soft).
2. Block strong “certificate coupling repairs indicators” language in all fragments.

D2 artifacts: Step 2 deliverable. Until recorded, treat claim strength as **auto-soft pending D2**.

---

## Allowed claims (this contrast only)

When predicates and namespaces above are satisfied:

- On ASR landscape (9 cells, frozen decodes), **CRC-style deploy-valid %** differs between decoupled and coupled score families under the same omission/group definition.
- Polarity matches 007 E4 direction on stated strata (coupled ≥ decoupled on nonvacuous pairs where both arms qualify).
- Overlap with main asr-gate LTT results is **intentional**; novelty is limited to score-coupling characterization.

---

## Non-goals (binding)

- Unfreeze or edit `submissions/asr` (frozen).
- Writes to `experiments/007-*` or frontier `007` pursue lane revival.
- Un-PARK 001; GPU salvage 002–006.
- GPU re-decode or new backbone bake-off.
- Treating portfolio first-wave complete as paper-ready or unfreeze-authorized.
