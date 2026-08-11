#!/usr/bin/env python3
"""Mandarin calibration-pool sweep (the free odds-mover; 0 GPU-hours).

Mirrors ``english_calsweep_2026-07-13/run_calsweep.py``'s methodology for the
Mandarin Paraformer arm, on CACHED scores only: hold the EVALUATION set fixed
and vary only the calibration budget ``n_cal``, per target ``alpha``, and report
the smallest ``n_cal`` at which the certificate first becomes non-vacuous (the
pool -> alpha frontier).

Why this moves TASLP odds (COMPUTE-PLAN-2026-07-13.md §1.3): the paper's own
English calsweep already showed sub-target vacuity is a CALIBRATION-BUDGET
artifact (LTT/EB slack ~ 1/sqrt(n_cal)), not an intrinsic property of the bound.
On the Mandarin arm the frozen main run certifies alpha=2% on a dev cal carve of
n_cal~3,567 but is VACUOUS at alpha=1.5% (0/20; alpha015_2026-07-13). The
question this answers: does a LARGER speaker-disjoint dev calibration pool push
the binding sub-1.98%-base-rate target alpha=1.5% into a non-vacuous certificate
on the FIXED official test set? If so, the certified frontier tightens for ~0
compute -- the single highest odds-moved-per-GPU-hour item in the plan.

Design (post-hoc, disclosed) -- the LEGITIMATE frozen protocol, no test-partition
calibration (cleaner than the English design, which drew cal and eval from one
split):
  * EVAL = the fixed official Aishell-1 test set (main_results_2026-07-09/
    test_scored.jsonl, 7,176 utts / 20 speakers; the FROZEN artifact -- never
    touched by calibration). It is speaker-disjoint from dev by construction.
  * CAL POOL = the Aishell-1 dev split (pilot_results_2026-07-09/
    dev_canonical.jsonl, ~14.3k ref-bearing utts / 40 speakers), scored ONCE
    with the frozen pipeline (score_table + compute_cer_batch, numeral_policy=
    keep) -- the same pipeline that produced test_scored.jsonl.
  * SWEEP: for each target n_cal, take the first n_cal utts of a seeded shuffle
    of the dev pool and certify with the EXACT frozen G1 machinery.

Two certification views, both reported:
  1. PRIMARY -- flat asr_gate.ltt.ltt_certify (EB p-value, Bonferroni-over-grid,
     delta=0.1, n_grid=200, min_accept_frac=0.1, score=s1, loss=CER), mirroring
     english_calsweep EXACTLY. This isolates the pure calibration-BUDGET axis
     (n_cal = exactly the # calibration examples the bound sees), the same
     isolation the English sweep used.
  2. CONFIRMATION -- the frozen MANDARIN certificate machinery
     (gate.calibrate_gate with strata=[duration_tercile], fit_frac=0.5, the
     config alpha015/speaker_partition/the main run use) on the FULL dev pool at
     the tight alphas, applied to the fixed test set. Shows the budget effect
     survives under the actual reported certificate (duration-tercile Mondrian +
     fit/conformalize split), not only under the flat bound.

Consistency guard: re-scores test_canonical.jsonl with this runner's pipeline and
asserts it reproduces the frozen test_scored.jsonl s1/CER (proving the cal-pool
scoring is the frozen pipeline, so the swept cal scores are on the same scale as
the frozen eval scores).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent.parent))  # reliability-commons/ (for `relmetrics`)

from asr_gate import cer as _cer        # noqa: E402
from asr_gate import gate as _gate       # noqa: E402
from asr_gate import ltt as _ltt         # noqa: E402
from asr_gate import scores as _scores   # noqa: E402

DEV_CANON = ROOT / "pilot_results_2026-07-09" / "dev_canonical.jsonl"
TEST_SCORED = ROOT / "main_results_2026-07-09" / "test_scored.jsonl"
TEST_CANON = ROOT / "main_results_2026-07-09" / "test_canonical.jsonl"

# Frozen G1 config (english_calsweep + FREEZE-NOTE-2026-07-09).
DELTA = 0.1
N_GRID = 200
MIN_ACCEPT_FRAC = 0.1
PROCEDURE = "bonferroni"
P_VALUE = "eb"
NUMERAL_POLICY = "keep"
SEED = 0

# Binding band around the 1.98% base rate (mirrors alpha015) + frozen grid tail.
ALPHAS = [0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.03, 0.05]
N_CAL_TARGETS = [613, 1000, 1500, 2000, 3000, 3567, 5000, 7000, 10000, 14000]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def score_pool(recs):
    """Frozen pipeline: s1-s4 (score_table) + CER (compute_cer_batch, keep).
    Keeps only ref-bearing utts with a usable s1 and cer (the cal/eval contract)."""
    recs = [r for r in recs if r.get("ref_text")]
    scored = _scores.score_table(recs)
    scored = _cer.compute_cer_batch(scored, numeral_policy=NUMERAL_POLICY)
    return [r for r in scored if r.get("s1") is not None and r.get("cer") is not None]


def consistency_guard(test_scored):
    """Re-score test_canonical with THIS pipeline and confirm it reproduces the
    frozen test_scored.jsonl s1/CER -- proves the swept cal scores are on the
    frozen scale. Returns a dict of match stats (never raises; loud warning on
    mismatch)."""
    if not TEST_CANON.exists():
        return {"checked": False, "reason": "test_canonical.jsonl absent"}
    rescored = score_pool(load_jsonl(TEST_CANON))
    frozen = {r["utt_id"]: r for r in test_scored}
    n, s1_max, cer_max = 0, 0.0, 0.0
    for r in rescored:
        f = frozen.get(r["utt_id"])
        if f is None or f.get("s1") is None:
            continue
        n += 1
        s1_max = max(s1_max, abs(float(r["s1"]) - float(f["s1"])))
        cer_max = max(cer_max, abs(float(r["cer"]) - float(f["cer"])))
    ok = (n > 0 and s1_max < 1e-6 and cer_max < 1e-9)
    if not ok:
        print(f"WARNING: consistency guard mismatch (n={n} s1_max_abs_diff={s1_max:.3e} "
              f"cer_max_abs_diff={cer_max:.3e}) -- cal/eval scores may not be on the "
              f"same scale; interpret the sweep with care.", file=sys.stderr)
    return {"checked": True, "n_compared": n, "s1_max_abs_diff": s1_max,
            "cer_max_abs_diff": cer_max, "reproduces_frozen": bool(ok)}


def shuffled_pool(cal_recs, seed=SEED):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cal_recs))
    return [cal_recs[i] for i in order]


def certify_flat(cal, alpha):
    losses = np.array([r["cer"] for r in cal], dtype=float)
    scores = np.array([r["s1"] for r in cal], dtype=float)
    return _ltt.ltt_certify(
        losses, scores, alpha=alpha, delta=DELTA, n_grid=N_GRID,
        min_accept_frac=MIN_ACCEPT_FRAC, procedure=PROCEDURE, p_value=P_VALUE,
    )


def eval_threshold(lambda_star, eval_recs):
    """Apply an s1 >= lambda_star threshold to the fixed test set."""
    if lambda_star is None:
        return {"eval_accept_rate": 0.0, "eval_accepted_macro_cer": None}
    s1 = np.array([r["s1"] for r in eval_recs], dtype=float)
    cer = np.array([r["cer"] for r in eval_recs], dtype=float)
    mask = s1 >= lambda_star
    acc = float(mask.mean())
    return {"eval_accept_rate": acc,
            "eval_accepted_macro_cer": (float(cer[mask].mean()) if mask.any() else None)}


def run_flat_sweep(cal_pool, eval_recs, seed=SEED):
    pool = shuffled_pool(cal_pool, seed=seed)
    cells, crossover = [], {f"{a}": None for a in ALPHAS}
    for target in N_CAL_TARGETS:
        n_cal = min(target, len(pool))
        cal = pool[:n_cal]
        for alpha in ALPHAS:
            res = certify_flat(cal, alpha)
            ev = eval_threshold(res["lambda_star"], eval_recs)
            non_vacuous = bool(res["certified"] and res["accepted_fraction"] > 0)
            viol = (ev["eval_accepted_macro_cer"] is not None
                    and ev["eval_accepted_macro_cer"] > alpha)
            cells.append({
                "n_cal_target": target, "n_cal": n_cal, "alpha": alpha,
                "certified": bool(res["certified"]), "non_vacuous": non_vacuous,
                "lambda_star": res["lambda_star"],
                "cal_accepted_fraction": res["accepted_fraction"], "K": res.get("K"),
                "eval_violation": bool(viol), **ev,
            })
            if non_vacuous and crossover[f"{alpha}"] is None:
                crossover[f"{alpha}"] = n_cal
    return cells, crossover


def frozen_machinery_confirmation(dev_pool, test_recs):
    """CONFIRMATION view: the frozen Mandarin certificate (calibrate_gate,
    strata=duration_tercile, fit_frac=0.5) on the FULL dev pool at the tight
    alphas, applied to the fixed test set. Contrasts with alpha015's frozen
    ~3,567-carve result (vacuous at 1.5%)."""
    out = []
    for alpha in [0.015, 0.017, 0.019, 0.02]:
        try:
            gate = _gate.calibrate_gate(
                [dict(r) for r in dev_pool], tune_instances=None, alpha=alpha,
                delta=DELTA, g1_score="s1", guarantee="ltt",
                strata=["duration_tercile"], fit_frac=0.5,
                numeral_policy=NUMERAL_POLICY, seed=SEED,
            )
            applied = _gate.apply_gate(gate, [dict(r) for r in test_recs])
            by_id = {r["utt_id"]: r for r in test_recs}
            acc_ids = [d["utt_id"] for d in applied["decisions"] if d["action"] == "ACCEPT"]
            cers = [by_id[i]["cer"] for i in acc_ids if i in by_id]
            macro = float(np.mean(cers)) if cers else None
            out.append({
                "alpha": alpha, "n_cal": gate["n_cal"], "n_fit": gate["n_fit"],
                "certified": bool(gate["g1"]["certified"]),
                "lambda_star": gate["g1"]["lambda_star"],
                "cal_accepted_fraction": gate["g1"]["accepted_fraction"],
                "eval_n": applied["n"], "eval_accept_rate": applied["n_accept"] / applied["n"],
                "eval_accepted_macro_cer": macro,
                "eval_violation": bool(macro is not None and macro > alpha),
            })
        except Exception as exc:  # noqa: BLE001 - confirmation is best-effort
            out.append({"alpha": alpha, "error": f"{type(exc).__name__}: {exc}"})
    return out


def main():
    dev_pool = score_pool(load_jsonl(DEV_CANON))
    test_recs = load_jsonl(TEST_SCORED)
    test_recs = [r for r in test_recs if r.get("s1") is not None and r.get("cer") is not None]
    full_macro = float(np.mean([r["cer"] for r in test_recs]))

    guard = consistency_guard(test_recs)

    # PRIMARY sweep (seed 0) + per-cell eval on the fixed test set.
    cells, crossover = run_flat_sweep(dev_pool, test_recs, seed=SEED)

    # Seed robustness of the crossover (seeds 0..4).
    seed_rob = {f"{a}": [] for a in ALPHAS}
    for seed in range(5):
        _, xo = run_flat_sweep(dev_pool, test_recs, seed=seed)
        for a in ALPHAS:
            seed_rob[f"{a}"].append(xo[f"{a}"])

    confirmation = frozen_machinery_confirmation(dev_pool, test_recs)

    out = {
        "config": {
            "delta": DELTA, "n_grid": N_GRID, "min_accept_frac": MIN_ACCEPT_FRAC,
            "procedure": PROCEDURE, "p_value": P_VALUE, "numeral_policy": NUMERAL_POLICY,
            "seed": SEED, "g1_score": "s1", "loss": "cer",
            "alphas": ALPHAS, "n_cal_targets": N_CAL_TARGETS,
            "backbone": "Paraformer-zh (B2)", "corpus": "Aishell-1",
            "eval": "fixed official test (test_scored.jsonl, frozen)",
            "cal_pool": "Aishell-1 dev (dev_canonical.jsonl), scored once with the "
                        "frozen pipeline; speaker-disjoint from test by construction",
            "design": "hold eval fixed; vary calibration budget n_cal over the dev pool "
                      "(seeded utterance shuffle); PRIMARY=flat ltt_certify (mirrors "
                      "english_calsweep), CONFIRMATION=frozen calibrate_gate machinery "
                      "on full dev pool (post-hoc, disclosed)",
        },
        "consistency_guard": guard,
        "eval_full_set_macro_cer": full_macro,
        "cal_pool_size": len(dev_pool),
        "eval_n": len(test_recs),
        "crossover_seed0": crossover,
        "crossover_seed_robustness": {"seeds": [0, 1, 2, 3, 4], "by_alpha": seed_rob},
        "cells_seed0": cells,
        "frozen_machinery_confirmation": confirmation,
    }
    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print("=== Mandarin calibration-pool sweep (Paraformer, Aishell-1; 0 GPU) ===")
    print(f"cal pool (dev) = {len(dev_pool)} utts;  eval (fixed test) = {len(test_recs)} utts")
    print(f"full-set test macro-CER (base rate) = {full_macro*100:.2f}%  "
          f"(alpha < this BINDS)")
    print(f"consistency guard reproduces frozen test s1/CER: {guard.get('reproduces_frozen')} "
          f"(s1_max_diff={guard.get('s1_max_abs_diff')})")
    print("\nsmallest n_cal that certifies non-vacuously on the FIXED test (seed 0):")
    for a in ALPHAS:
        xo = crossover[f"{a}"]
        binds = " (BINDS)" if a < full_macro else ""
        print(f"  alpha={a*100:.1f}%{binds}: crossover n_cal = "
              f"{xo if xo is not None else 'NONE up to %d' % min(max(N_CAL_TARGETS), len(dev_pool))}")
    print("\ncrossover seed-robustness (seeds 0-4):")
    for a in ALPHAS:
        print(f"  alpha={a*100:.1f}%: {seed_rob[f'{a}']}")
    print("\nfrozen-machinery confirmation (full dev pool, calibrate_gate + duration_tercile):")
    for c in confirmation:
        if "error" in c:
            print(f"  alpha={c['alpha']*100:.1f}%: ERROR {c['error']}")
            continue
        mc = f"{c['eval_accepted_macro_cer']*100:.2f}%" if c["eval_accepted_macro_cer"] is not None else "  -  "
        print(f"  alpha={c['alpha']*100:.1f}%: n_cal={c['n_cal']} cert={c['certified']} "
              f"eval_accept={c['eval_accept_rate']*100:.1f}% eval_acc_macroCER={mc} "
              f"viol={c['eval_violation']}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
