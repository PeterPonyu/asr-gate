# D5 — Positioning firewall and namespaced scores

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Section id** | `crc_omission_contrast` |
| **Status** | Venue-neutral positioning draft — **NOT submission-ready** |
| **Scope** | Claim boundaries; 0 writes to `submissions/asr` |

---

## Positioning (one paragraph)

This contrast is a **CPU-only score-family comparison** on frozen ASR landscape decodes under group-conditional risk control when the **group indicator is unobserved at deploy time** and **destroyed by the omission loss**. It closes the 007 E4 residual—“run the coupled-vs-decoupled contrast on ASR landscape too”—using the same corpora and backbones as the main asr-gate reliability line. The endpoint is **CRC-style deploy-valid %** at fixed α, not aggregate LTT selective-ASR certificates. Overlap with the main package is **intentional**; any novelty is limited to **score-coupling characterization**, not a new corpus or backbone bake-off.

---

## LTT ↔ CRC firewall (hard rules)

| Object | Domain | May claim |
|---|---|---|
| **LTT G1** (aggregate selective ASR certificates, s1–s5) | Main asr-gate / TASLP manuscript substrate | Marginal / class-conditional **LTT** coverage guarantees |
| **CRC deploy-valid** (this contrast) | Appendix-only `crc_omission_contrast` | Group-conditional **CRC-style** deploy-valid % under destroyed-indicator omission |

**Forbidden:**

1. **LTT G1 ≠ CRC deploy-valid** — different estimands, calibration objects, and guarantee types.
2. **Do not narrate CRC deploy-valid as an LTT G1 extension** — no “strengthens G1”, “extends the main certificate”, or “confirms LTT coverage under omission”.
3. **Do not merge** CRC contrast body claims into LTT narrative without namespace + overlap-disclosure boilerplate and human Q2 reversal.
4. **Do not claim** “G1 extension” anywhere in D4–D7 fragments.

---

## Display score namespaces (appendix contract)

Internal keys **must not** appear bare in appendix tables or manuscript fragments.

| Internal key | Appendix display name | E4 analogue |
|---|---|---|
| `s1` | **`s1_crc_decoupled`** | `decoupled` |
| `hyp_entity_mass` | **`hyp_entity_mass_coupled`** | `conf` / coupled |

**Ban:** bare `s1` in appendix tables, figures, and D4–D6 draft fragments.

---

## Claim strength (Q6 auto-soft policy)

| State | Wording allowed | Wording blocked |
|---|---|---|
| **Auto-soft (applied)** | “constructed hyp-side entity mass”; “score-coupling contrast on ASR landscape”; “CRC deploy-valid differs by score family under omission” | “certificate coupling **repairs** indicators”; “coupling **fixes** destroyed-indicator validity”; causal repair language |
| **Strong repair** | — | **Blocked** — D2 heuristic did not trip, but scrambled-POS ablation skipped; no strong repair claims without further human override |

D2 auto-soft heuristic (2026-08-10): degenerate `pred_ent` both-arms median Δ=**0.0** vs headline **0.6**; ratio=**0.00** → **NOT_TRIPPED**.

---

## Allowed claims (this contrast only)

When namespaces and strata above are satisfied:

- On ASR landscape (9 cells, frozen decodes), **CRC-style deploy-valid %** differs between **`s1_crc_decoupled`** and **`hyp_entity_mass_coupled`** under the same omission/group definition.
- Polarity matches 007 E4 direction on the both-arms primary stratum (coupled ≥ decoupled where both arms qualify).
- Overlap with main asr-gate LTT results is intentional; novelty limited to score-coupling characterization.

---

## Overlap — non-claims (paste boundary)

The following are **explicit non-claims** for any downstream submission revision:

- **Not** a new pursue lane under `frontier-directions-research/directions/007`.
- **Not** a GPU re-decode or new backbone comparison.
- **Not** evidence that asr-gate’s LTT guarantees are CRC guarantees (different objects).

---

## Completion predicates (do not collapse)

```
packaging-complete  ⇏  CLEAR-for-packaging  ⇏  unfreeze-authorized
```

`packaging-complete` for this contrast does **not** authorize editing `submissions/asr`. Q2-reprise default: **keep frozen**.

---

## Non-goals reaffirmed

- Unfreeze or edit `submissions/asr` (frozen).
- Writes to `experiments/007-*` or frontier 007 pursue lane revival.
- Un-PARK **001**; GPU salvage **002–006**.
- Portfolio first-wave complete ⇒ paper-ready or unfreeze-authorized.
