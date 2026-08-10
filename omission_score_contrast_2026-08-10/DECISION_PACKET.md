# Decision packet — CRC omission contrast (Step 5 / hard-gate terminus)

| Field | Value |
|---|---|
| **Date** | 2026-08-10 |
| **Owner** | asr-gate team `asr-gate-paper-packaging` / worker-packet |
| **Plan** | `ralplan-asr-gate-paper-packaging` Step 5 |
| **Science home** | `asr-gate/omission_score_contrast_2026-08-10/` |
| **Status** | Hard-gate terminus emitted — **not** submission-ready, **not** unfreeze-authorized |

---

## Executive summary

Empirical packaging for the CRC omission score contrast is **complete under approved defaults**, with one **pending** acceptance criterion: **evidence 3a git commit** (gitignore carve + staged commit). Jieba vendor pin and private cache remediate security **MEDIUM** findings locally. Venue writing remains **deferred** (Q5); `submissions/asr` stays **frozen** (Q2-reprise). Claim strength follows **auto-soft policy** (Q6): D2 heuristic did **not** trip; strong “certificate coupling repairs indicators” language remains **blocked**.

---

## Three completion predicates (reaffirmed)

These predicates are **not substitutable**. This packet satisfies the **`DECISION_PACKET` emitted** limb of `packaging-complete` only when paired with Steps 1–3 + D1–D3 AC green.

| Predicate | Status after this packet |
|---|---|
| **`packaging-complete`** | **Conditional green** — Steps 1–2 + D1–D3 green; Step 3 vendor/cache green; Step 3 **3a git commit pending**; this packet present |
| **`CLEAR-for-packaging`** | **Not asserted** — Architect / human upgrade only |
| **`unfreeze-authorized`** | **Not authorized** — human Q2 reversal required |

**Forbidden chain (non-implication):**

```
packaging-complete  ⇏  CLEAR-for-packaging  ⇏  unfreeze-authorized
```

No downstream artifact may collapse these into a single “done” or “paper-ready” flag.

---

## Open questions — decided defaults (binding)

### Q2-reprise — keep `submissions/asr` frozen

| Decision | **KEEP FROZEN** |
|---|---|
| Rationale | First-wave Q2 was asr-gate-only writes; appendix merge into the frozen kit is **not scheduled** |
| Effect | D4–D7 fragments live under science home only; **zero writes** to `submissions/asr` |
| Override | Human-only Q2 reversal required for any kit edit or appendix paste |

### Q5 — venue choice

| Decision | **DEFER venue / keep fragments venue-neutral** |
|---|---|
| Rationale | TASLP appendix vs Option D standalone vs other venue unset; writing maturity ~40 vs science ~90 |
| Effect | D4–D6 (+ D7) drafted as **NOT submission-ready** venue-neutral fragments; no G1 extension framing |
| Override | Human Q5 tendency when venue is chosen; fragments may be retargeted then |

### Q6 — claim strength (auto-soft policy)

| Decision | **Accept auto-soft policy outcome** |
|---|---|
| D2 heuristic | **NOT_TRIPPED** — degenerate `pred_ent` both-arms: n=5, median Δ=**0.0** vs headline both-arms median Δ=**0.6**; ratio=**0.00** (< 80% threshold) |
| Allowed wording | “constructed hyp-side entity mass”; “score-coupling contrast on ASR landscape”; “CRC deploy-valid differs by score family under omission” |
| Blocked wording | “certificate coupling **repairs** indicators”; “coupling **fixes** destroyed-indicator validity”; any causal repair language; LTT G1 extension |
| Note | Heuristic non-trip does **not** prove strong repair claims; scrambled-POS / POS LOO ablation (D2-c) **skipped** — strong coupling language stays blocked regardless |

### Evidence 3a — git audit pin

| Item | Status |
|---|---|
| **Required** | Gitignore carve in `ml-reliability-research` + commit of minimal pin set |
| **Filesystem** | `MANIFEST.sha256` **present** (2026-08-10; lists results, code, D1–D3, firewall docs) |
| **Git** | **Pending** — `git check-ignore` still matches `/asr-gate/` for contrast dir; commit not landed at packet emission |
| **3b mirror** | Optional; **not** a substitute for 3a |

### Security MEDIUM — remediation status

| Finding | Remediation | Status |
|---|---|---|
| **MEDIUM #1** — jieba `/tmp` marshal cache | Private cache `HERE/.jieba_cache` before import; `TMPDIR`/`TEMP`/`TMP` redirected | **REMEDIATED** — `run.log` loads from `.jieba_cache/jieba.cache` |
| **MEDIUM #2** — unpinned cross-repo vendor | Local `vendor/` (jieba 0.42.1) under science home; no frontier `sys.path` | **REMEDIATED** — `run_contrast.py` uses `HERE/vendor` |
| Formal review | Cited below | **PRESENT** |
| Residual LOW | Path redaction, JSONL bounds, latent paddle helper | Open for public release; not blocking single-user offline packaging |

**Formal security review (verified):**  
`/home/zeyufu/Desktop/frontier-directions-research/.omc/research/research-2026-08-10-paper-readiness/findings/verified/security-review.md`

Local copy: `SECURITY_REVIEW.md` in this directory.

---

## Non-goals reaffirmed (001 PARK + 002–006)

| ID | Non-goal | Status |
|---|---|---|
| **001** | Un-PARK / rewrite-gate revival | **PARK** — not in scope |
| **002–006** | GPU salvage pursue lanes | **Out of scope** — no GPU re-decode, no new backbone bake-off |
| — | Unfreeze or edit `submissions/asr` | **Blocked** (Q2-reprise default) |
| — | Writes to `experiments/007-*` or frontier 007 lane | **Blocked** |
| — | Portfolio first-wave complete ⇒ paper-ready / unfreeze | **Blocked** (`CLAIM_MATRIX.md`) |
| — | CRC deploy-valid narrated as LTT G1 extension | **Blocked** (`DISPLAY_NAMESPACE.md`, `SELF_OVERLAP.md`) |

---

## Hard-gate checklist — Steps 1–3 + D1–D3

### Step 1 — Claim firewall & metric contracts

| AC | Status |
|---|---|
| `CLAIM_MATRIX.md` forbids forbidden inference chain | **Green** |
| `DISPLAY_NAMESPACE.md` — `s1_crc_decoupled`, `hyp_entity_mass_coupled` | **Green** |
| Dual vacuity in code/export; both-arms primary + OR sensitivity | **Green** |
| `MAPPING.md` provenance block | **Green** |
| No `submissions/asr` edits | **Green** |

### Step 2 — D1–D3 + re-run

| AC | Status |
|---|---|
| Re-run after dual-vacuity semantics | **Green** — `wall_s` ≈ **55.2** s CPU |
| `code_sha256` | **Green** — `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78` |
| **PRIMARY** both-arms nonvacuous | **Green** — n=**13**, median Δ=**0.6**, `effect_direction_matches_e4=True` |
| OR sensitivity retained | **Green** — n=**27**, median Δ=**0.0** |
| All-pairs median Δ visible | **Green** — median Δ=**0.0** (27 cell×α pairs) |
| D1 stratified tables | **Green** — `D1_STRATIFIED.md` |
| D2 ablations + auto-soft heuristic | **Green** — heuristic **NOT_TRIPPED**; D2-c skipped (documented) |
| D3 Pareto | **Green** — `D3_PARETO.md` |
| 007 tree untouched | **Green** |

### Step 3 — Reproducibility & evidence pin

| AC | Status |
|---|---|
| No absolute frontier jieba path in entrypoint | **Green** |
| Local `vendor/` + private `.jieba_cache/` | **Green** |
| Security MEDIUM pin + cache policy | **Green** |
| Formal security review cited | **Green** |
| **3a gitignore exception + commit** | **Pending** — `MANIFEST.sha256` on disk; git still ignores contrast dir |

### D1–D3 deliverables

| Artifact | Status |
|---|---|
| `D1_STRATIFIED.md` / `.json` | **Green** |
| `D2_ABLATIONS.md` | **Green** (D2-c deferred) |
| `D3_PARETO.md` | **Green** |

---

## Re-run metrics (cited)

| Metric | Value |
|---|---|
| Wall time | ≈ **55.2** s CPU |
| Primary (both-arms deploy-nonvacuous) | n=**13**, median Δ (coupled − decoupled) = **0.6** |
| OR sensitivity | n=**27**, median Δ = **0.0** |
| All pairs | n=**27**, median Δ = **0.0** |
| E4 direction match (primary) | **True** |
| `code_sha256` | `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78` |
| α grid | 0.05, 0.10, 0.20 |
| Landscape | `landscape_pulled_2026-07-15` (9 cells, frozen decodes) |

---

## Step 4 status (soft gate)

| Item | Status |
|---|---|
| D4 α/vacuity protocol | **Drafted** — `D4_ALPHA_VACUITY_PROTOCOL.md` |
| D5 positioning firewall | **Drafted** — `D5_POSITIONING_FIREWALL.md` |
| D6 draft fragments | **Drafted** — `D6_DRAFT_FRAGMENTS.md` (NOT submission-ready) |
| D7 AUROC disclosure | **Drafted** — `D7_AUROC_DISCLOSURE.md` |
| Writes to `submissions/asr` | **0** |

Step 4 complete under **Q5 defer** default (venue-neutral fragments only).

---

## Recommended defaults applied

| Question | Applied default |
|---|---|
| Q2-reprise | Keep `submissions/asr` **frozen** |
| Q5 | **Defer** venue; fragments venue-neutral |
| Q6 | **Accept auto-soft policy**; heuristic NOT_TRIPPED; careful constructed hyp-side entity mass wording |
| Evidence | **3a required**; `MANIFEST.sha256` present; **git commit pending** |
| Security MEDIUM | **Remediated** in runnable path (local vendor + private cache) |

---

## Remaining AC gaps (for G006 verify)

1. **3a git commit** — carve gitignore exception in `ml-reliability-research`, stage minimal pin set, commit; refresh `MANIFEST.sha256` to include post-packet artifacts when landed.
2. **D2-c** — scrambled-POS / POS LOO ablation skipped; document in any future strong-claim review.
3. **LOW security** — path redaction before public artifact release; JSONL bounds optional hardening.
4. **`packaging-complete` final verify** — executor G006 after 3a commit lands.

---

## Human override hook

To change any binding default above, record an explicit human decision (Q2-reprise / Q5 / Q6 / evidence waiver) with date and override rationale. Do not infer overrides from packaging-complete or CLEAR-for-packaging.
