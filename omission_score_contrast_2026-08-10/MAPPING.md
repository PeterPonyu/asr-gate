# Score-family mapping — 007 E4 → ASR landscape (CPU contrast)

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Owner** | asr-gate (`omission_score_contrast_2026-08-10/`) |
| **Evidence SSOT (read-only)** | `frontier-directions-research/experiments/007-omission-crc/findings.md` |
| **Status** | packaging wave 2026-08-10 — metric contracts + provenance (Step 1) |

---

## Provenance & reproducibility

| Field | Value |
|---|---|
| **Mapping draft** | Original score-family mapping drafted **before first CPU count** (2026-08-10) |
| **Packaging wave** | 2026-08-10 regenerates artifacts with **dual-vacuity semantics** (`oracle_vacuous_frac`, `deploy_vacuous_frac`) |
| **Post-result edits** | Allowed for **honesty fields** (vacuity labels, stratum definitions, display namespaces, provenance) — **not** for hand-edited headline numbers |
| **`results.json` sha256** | `fb550a18237983a37ffd35af2ff64faac13a9358de211fb1b3503ca1c880bcb6` |
| **`run_contrast.py` code_sha256** | `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78` |
| **`wall_s` (from `results.json`)** | `55.2` |
| **Primary stratum** | **Both-arms nonvacuous** — cell×α pair in headline iff both `s1_crc_decoupled` and `hyp_entity_mass_coupled` arms are nonvacuous on their respective oracle/deploy vacuity sides |
| **Sensitivity stratum** | **OR nonvacuous retained** — pair included if either arm nonvacuous (`nonvacuous_s1 OR nonvacuous_coupled`); reported alongside primary, not as headline substitute |

Any post-result packaging edit to honesty/stratum fields **must** record the three hashes/timestamps above from the **run that produced the cited numbers**. Hand-editing `SUMMARY.md` / `results.json` aggregates without a logged re-run or documented array re-aggregation (input sha256 + exact command) is forbidden.

**Jieba vendor pin:** `vendor/` (jieba 0.42.1, tracked for reproducibility).  
**Private cache policy:** `.jieba_cache/` is gitignored (local runtime only); `__pycache__/` likewise ignored. Frontier path `frontier-directions-research/experiments/007-omission-crc/vendor/` is no longer used.

**Display namespaces:** See `DISPLAY_NAMESPACE.md`. Appendix section id: `crc_omission_contrast`.  
**Claim firewall:** See `CLAIM_MATRIX.md`.

---

## Inventory (ordered)

| Order | Source | Result |
|---|---|---|
| (a) | `asr-gate/landscape_pulled_2026-07-15` + decode `*.jsonl` with `nbest[].token_logps` | **USABLE** — 9 cells, ~98k utts (aishell / magicdata / thchs30 × paraformer / belle / zipformer) |
| (b) | Zenodo DOI `10.5281/zenodo.21392289` | Not needed — (a) present |
| (c) | Legacy `asr-fm-reliability-research` trees | Skipped — primary asr-gate landscape succeeded |

**Pinned corpus/backbone for this run:** all 9 landscape test cells above (same frozen decodes as 007 E3). No GPU re-decode.

---

## E4 polarity (quoted)

From `experiments/007-omission-crc/findings.md` W3 table (~557–570) and surrounding prose:

> Holding corpus, class, loss, indicator and generator **all fixed** and changing **only the certificate's score** … `conf` — reads the same observable that defines the group → deploy valid **100%**; `decoupled` — carries no information about what was asserted → deploy valid **20% / 0% / 0%**.

> **ASR's score (mean token logprob) cannot [detect silent entity drop]** — fluent hyp that omitted an entity still scores high. That is a property of **that score family**, not of generative certification in general.

> A group indicator destroyed by the loss breaks a group-conditional certificate **when the score cannot detect the destruction.** … a score coupled to the indicator does.

---

## Mapping (polarity gate — must pass before results count)

| Analogue | ASR score | E4 arm | Rationale |
|---|---|---|---|
| **Decoupled** | **s1** = mean 1-best `token_logps` (existing asr-gate PRIMARY score) | `decoupled` | Fluency/confidence; **cannot** see silent entity omission |
| **Coupled** | **constructed hyp-only** entity-assertion mass (below) | `conf` | Reads the **same hyp-side observable** that defines `pred_ent` |

**Polarity check (fail if inverted):** do **not** label s1 as coupled; do **not** treat another stock PRIMARY_SCORE as a substitute for construction.

---

## Coupled-score CPU construction recipe (frozen)

**Name:** `hyp_entity_mass`

**Inference-legal inputs only:** `hyp_text` (after byte-fallback decode) + jieba POS on hyp.  
**Forbidden in the score:** `ref_text`, edits-vs-ref, entity omission vs ref (refs are for **group/loss labels only**).

**Formula (pre-registered):**

```
hyp' = debyte(hyp_text)                         # zipformer <0xNN> → UTF-8
entity_mask = jieba POS ∈ {nr, ns, nt, nz} on hyp'
score_coupled = sum(entity_mask) / max(|hyp'|, 1)
```

- When the hyp asserts **no** entity tokens → score = **0** (bottom of scale) → CRC abstains first on silent drops (mirrors E4 `conf`).
- Helper reuse: same jieba / `pred_ent` machinery as 007 `run_e2.py` / `run_e3.py` (entity POS set + `debyte`), adapted to a **continuous hyp-only** mass rather than a binary flag.
- **Not** E0 certificate score; **not** shopping s2–s5 as coupled.

**Group / loss (labels; may use ref):**

- `true_ent`: ref has ≥1 entity token (oracle class; unavailable at deploy)
- `pred_ent`: hyp' has ≥1 entity token (observable / deploy grouping)
- `l_entity`: LCS-aligned fraction of ref entity chars omitted (same as E2/E3)

**CRC:** accept when `score ≥ λ̂`; `λ̂ = inf{λ : (n·R̂_cal(λ)+1)/(n+1) ≤ α}` (Angelopoulos et al.).  
Calibrate on **observable** group (`pred_ent`); evaluate **true** entity risk on all true-entity test utts (deploy-valid). Speaker-blocked cal/test splits.

**α grid:** `0.05 / 0.10 / 0.20` (mirror E4).  
**Seeds:** 20 speaker splits (`rng = 5000 + seed`).  
**Compute:** CPU-only; no new model calls.

---

## Primary endpoint

**CRC-style deploy-valid %** at each α for `s1` vs `hyp_entity_mass` on the same omission/group definition.  
Secondary (descriptive only): AUROC orientation check — does **not** replace deploy-valid as the success claim.

**Stratification (Step 1 contract):**

| Stratum | Rule | Role |
|---|---|---|
| **Primary (headline)** | Both-arms nonvacuous — oracle **and** deploy vacuity below threshold on **both** arms | Main aggregate in SUMMARY / D1 tables |
| **Sensitivity** | OR nonvacuous — either arm nonvacuous | Retained row for comparison; not headline substitute |

Vacuity fields: `oracle_vacuous_frac` and `deploy_vacuous_frac` per arm (dual semantics; Step 2 code re-run mandatory).

**Expected direction (from E4):** coupled deploy-valid ≫ decoupled deploy-valid on **both-arms nonvacuous** primary stratum.
