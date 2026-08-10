# D4 — α grid, dual vacuity, and stratification protocol

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Section id** | `crc_omission_contrast` |
| **Status** | Venue-neutral protocol draft — **NOT submission-ready** |
| **Scope** | Appendix methods contract; 0 writes to `submissions/asr` |

---

## Purpose

Pre-register the α grid, dual vacuity fields, primary both-arms stratum, and OR sensitivity stratum for the CRC-style deploy-valid contrast between **`s1_crc_decoupled`** and **`hyp_entity_mass_coupled`** on the frozen ASR landscape.

Internal pipeline keys (`s1`, `hyp_entity_mass`) appear only in code and JSON; outward-facing surfaces use display names per `DISPLAY_NAMESPACE.md`.

---

## α grid

| Parameter | Value |
|---|---|
| **α levels** | 0.05, 0.10, 0.20 |
| **Rationale** | Mirror 007 E4 W3 score-coupling contrast |
| **Calibration rule** | Accept when score ≥ λ̂; λ̂ = inf{λ : (n·R̂_cal(λ)+1)/(n+1) ≤ α} |
| **Calibration group** | Observable `pred_ent` (hyp-side entity assertion) |
| **Evaluation group** | True-entity test utterances (`true_ent`); endpoint = deploy-valid % |

Each **cell×α** pair (9 landscape cells × 3 α = 27 pairs) is one unit in aggregate tables unless stratified out by vacuity.

---

## Score families (display names)

| Display name | Internal key | E4 analogue | Role |
|---|---|---|---|
| **`s1_crc_decoupled`** | `s1` | `decoupled` | Mean 1-best token log-probability — fluency/confidence **decoupled** from entity-assertion indicator |
| **`hyp_entity_mass_coupled`** | `hyp_entity_mass` | `conf` / coupled | Hyp-only jieba entity-assertion mass — **coupled** to the same hyp-side observable that defines `pred_ent` |

**Ban:** bare `s1` in appendix tables and manuscript fragments from this section.

---

## Dual vacuity fields

Vacuity is computed **per arm** (decoupled and coupled separately) over speaker-blocked seed splits (N=20 seeds, base 5000).

| Field | Side | Definition |
|---|---|---|
| **`oracle_vacuous_frac`** | Oracle / calibration-side | Fraction of seeds where oracle flag rate ∈ {0.0, 1.0} |
| **`deploy_vacuous_frac`** | Deploy / evaluation-side | Fraction of seeds where deploy flag rate ∈ {0.0, 1.0} |

A seed contributes to vacuity statistics independently per score arm. Vacuous cells are **reported**, not silently dropped.

---

## Primary stratum — both-arms deploy-nonvacuous (headline)

A cell×α pair enters the **primary headline** aggregate iff **both** arms satisfy deploy-nonvacuous:

```
deploy_vacuous_frac(s1_crc_decoupled) < 0.5
AND
deploy_vacuous_frac(hyp_entity_mass_coupled) < 0.5
```

**Observed (2026-08-10 re-run):** n=**13** pairs; median Δ (coupled − decoupled deploy-valid %) = **+0.6**; effect direction matches 007 E4 on this stratum.

---

## Sensitivity stratum — OR deploy-nonvacuous

A cell×α pair enters the **OR sensitivity** row iff **either** arm is deploy-nonvacuous:

```
deploy_vacuous_frac(s1) < 0.5  OR  deploy_vacuous_frac(coupled) < 0.5
```

OR sensitivity **must remain visible** alongside the primary headline; it is not a substitute headline.

**Observed (2026-08-10 re-run):** n=**27** pairs; median Δ = **+0.0**.

---

## All-pairs aggregate (honesty row)

| Stratum | n | median Δ (coupled − decoupled) |
|---|---:|---:|
| all_pairs | 27 | +0.000 |
| both_arms_deploy_nonvacuous (primary) | 13 | +0.600 |
| or_deploy_nonvacuous_sensitivity | 27 | +0.000 |
| both_arms_oracle_nonvacuous | 10 | +0.775 |

All-pair median Δ=0 **must remain visible** regardless of primary stratum choice.

---

## Protocol invariants

1. **No stratum shopping** — primary and OR sensitivity reported together; vacuous fractions per cell in per-cell tables.
2. **No rename-only vacuity fixes** — field semantic changes require full `run_contrast.py` re-run with refreshed hashes.
3. **CPU-only** — frozen landscape decodes; no GPU re-decode.
4. **007 read-only** — E4 polarity SSOT at `experiments/007-omission-crc/`; no writes.
5. **Not LTT G1** — this protocol scopes CRC deploy-valid % only; see `D5_POSITIONING_FIREWALL.md`.

---

## Reproducibility pins (2026-08-10 run)

| Field | Value |
|---|---|
| `wall_s` | ≈ 55.2 |
| `code_sha256` | `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78` |
| Artifacts | `results.json`, `D1_STRATIFIED.md`, `SUMMARY.md`, `run.log` |
