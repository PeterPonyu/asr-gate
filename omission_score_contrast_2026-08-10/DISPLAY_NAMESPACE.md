# Display namespace map — appendix / manuscript surfaces

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Section id** | `crc_omission_contrast` |
| **Scope** | Outward-facing tables, figures, D4–D6 fragments only |

Internal keys in `results.json`, `run_contrast.py`, and `run.log` stay unchanged.  
Appendix and manuscript surfaces **must** use the display names below.

---

## Score-family map

| Internal key | Appendix display name | Meaning |
|---|---|---|
| `s1` | **`s1_crc_decoupled`** | Mean 1-best token log-probability — fluency/confidence score **decoupled** from entity-assertion indicator (E4 `decoupled` arm) |
| `hyp_entity_mass` | **`hyp_entity_mass_coupled`** | Hyp-only jieba entity-assertion mass — score **coupled** to the same hyp-side observable that defines `pred_ent` (E4 `conf` / coupled arm) |

---

## Vacuity / stratum labels (display)

| Internal / JSON field | Appendix label | Meaning |
|---|---|---|
| `oracle_vacuous_frac` | oracle vacuity % | Calibration/oracle-side degenerate-flag fraction per arm |
| `deploy_vacuous_frac` | deploy vacuity % | Deploy/evaluation-side degenerate-flag fraction per arm |
| both-arms nonvacuous stratum | **primary headline** | Pair included only when **both** arms nonvacuous on respective sides |
| OR nonvacuous stratum | **sensitivity row** | Pair included when **either** arm nonvacuous — retained for comparison, not headline |

---

## Ban

**Bare `s1` is forbidden** in appendix tables, figure legends, and submission-bound fragments.

Use `s1_crc_decoupled` or prose that explicitly scopes the score to the CRC omission contrast (`crc_omission_contrast` section).

The bare token `s1` remains valid **only** inside this directory’s internal artifacts (`results.json`, code, logs) and cross-references to the main LTT package where context is unambiguous.
