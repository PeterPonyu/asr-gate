# D7 — AUROC inverted disclosure

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Section id** | `crc_omission_contrast` |
| **Status** | Descriptive audit — **NOT submission-ready** |
| **Primary endpoint** | Deploy-valid % (AUROC is secondary / orientation check only) |

---

## Definition

For each landscape cell, we compute **AUROC of score ranking high entity-omission loss above low loss** on true-entity test utterances. For CRC accept-when-score≥λ, a score that ranks **high loss above low loss** (AUROC **> 0.55**) is flagged **`auroc_inverted_warning`** — the score orientation is misaligned with the accept≥λ gate for that loss direction.

Source: `run_contrast.py` (`auroc_high_loss`); per-cell fields in `results.json`.

**Display names in tables below:** `s1_crc_decoupled` (internal `s1`); `hyp_entity_mass_coupled` (internal `hyp_entity_mass`).

---

## Per-cell AUROC (true-class high-loss ranking)

| corpus | backbone | s1_crc_decoupled AUROC | hyp_entity_mass_coupled AUROC | s1 inverted? | coupled inverted? |
|---|---|---:|---:|:---:|:---:|
| aishell | paraformer | 0.166 | 0.513 | no | no |
| aishell | belle | 0.203 | 0.488 | no | no |
| aishell | zipformer | 0.375 | 0.418 | no | no |
| magicdata | paraformer | 0.193 | 0.420 | no | no |
| magicdata | belle | 0.214 | 0.370 | no | no |
| magicdata | zipformer | 0.254 | 0.356 | no | no |
| thchs30 | paraformer | 0.270 | **0.593** | no | **yes** |
| thchs30 | belle | 0.241 | 0.543 | no | no |
| thchs30 | zipformer | 0.350 | **0.586** | no | **yes** |

**Summary counts:** `s1_crc_decoupled` inverted **0 / 9** cells; `hyp_entity_mass_coupled` inverted **2 / 9** cells (both thchs30: paraformer, zipformer).

---

## Interpretation (disclosure language)

1. **Primary claim uses deploy-valid %**, not AUROC. AUROC rows are **descriptive orientation checks** only.
2. **`s1_crc_decoupled`** AUROC values are uniformly **below 0.55** on this run — no inverted-warning flags on the decoupled arm. Low AUROC indicates weak rank separation for omission loss; this is consistent with the E4 narrative that fluency score does not detect silent entity drop.
3. **`hyp_entity_mass_coupled`** triggers inverted-warning on **two** thchs30 cells (AUROC > 0.55). Report alongside deploy-valid deltas; do not over-interpret as a standalone success metric.
4. **`SUMMARY.md`** contains generic AUROC-warning boilerplate (“Several cells show s1 AUROC > 0.55”); the **2026-08-10 re-run** `results.json` does **not** show s1 inverted flags. Prefer this D7 table for outward disclosure until SUMMARY boilerplate is refreshed.

---

## Relation to D3 Pareto

`D3_PARETO.md` notes that **`s1_crc_decoupled`** tends toward lower deploy_flag at low α while **`hyp_entity_mass_coupled`** trades higher abstention for deploy-valid gains on some cells. AUROC inversion on coupled arms does not replace that Pareto readout.

---

## Claim firewall reminder

AUROC orientation does **not** establish LTT coverage, CRC certificate repair, or G1 extension. See `D5_POSITIONING_FIREWALL.md` and `CLAIM_MATRIX.md`.

---

## Provenance

| Field | Value |
|---|---|
| Source artifact | `results.json` → `auroc_high_loss_on_true_class`, `auroc_inverted_warning` |
| Run | 2026-08-10 CPU re-run, `wall_s` ≈ 55.2 |
| `code_sha256` | `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78` |
