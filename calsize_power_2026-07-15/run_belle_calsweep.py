#!/usr/bin/env python3
"""Belle-whisper-large-v3-zh (B3') calibration-size power curve (0 GPU-hours).

Deferred CPU work: the landscape (FREEZE-AMENDMENT-2026-07-13, pulled
2026-07-15) certifies Belle on the official Aishell-1 test only at
alpha in {5,10}% at the frozen calibration budget (n_cal~=3,567 after the
duration-tercile Mondrian fit/conformalize split) -- vacuous at
alpha in {1.5,2,3}%. The paper's own English arm
(english_calsweep_2026-07-13) and Mandarin/Paraformer arm
(mandarin_calsweep_2026-07-13) both showed that sub-target vacuity is a
CALIBRATION-BUDGET artifact (LTT/EB bound slack ~ 1/sqrt(n_cal)), not an
intrinsic property of the distribution-free bound. This runner asks the
same question for Belle -- the independence argument's load-bearing cell
(the M2 contamination answer rests on Belle certifying non-vacuously) --
and additionally reports it as a POWER CURVE: for each calibration size
n_cal, what FRACTION of R=20 reseeds certify non-vacuously (not just the
seed-0 crossover), since a single-seed crossover can be a lucky draw.

Design (post-hoc, disclosed), mirrors mandarin_calsweep_2026-07-13 EXACTLY
(same legitimate frozen protocol: calibrate on dev, certify on the fixed
official test, so there is no test-partition calibration caveat):
  * EVAL (fixed) = landscape_pulled_2026-07-15/backbone_belle/
    aishell_test_scored.jsonl -- the pulled, byte-verified frozen box
    artifact, 7,176 s1/cer-bearing utts, official Aishell-1 test, never
    touched by calibration.
  * CAL POOL = landscape_pulled_2026-07-15/backbone_belle/
    dev_canonical.jsonl (Aishell-1 dev, 14,329 utts / 14,326 ref-bearing),
    scored ONCE with the frozen pipeline (score_table + compute_cer_batch,
    numeral_policy=keep) -- the same pipeline that produced
    aishell_test_scored.jsonl. Speaker-disjoint from test by construction.
  * SWEEP: for each n_cal in {250, 500, 1000, 2000, 4000, full pool}, and
    each of R=20 reseeds (seeded utterance shuffles of the dev pool, seeds
    0..19 -- distinct from the frozen landscape's 20 SPEAKER-level reseeds,
    which resample the cal/tune carve rather than subsample by budget),
    take the first n_cal utts of that reseed's shuffle and certify with the
    frozen G1 machinery (asr_gate.ltt.ltt_certify, EB p-value,
    Bonferroni-over-grid, delta=0.1, n_grid=200, min_accept_frac=0.1,
    g1-score=s1, loss=CER -- IDENTICAL config to english_calsweep_2026-07-13
    and mandarin_calsweep_2026-07-13's PRIMARY view). No bound machinery
    changed.
  * ALPHA GRID = the frozen landscape grid {0.015, 0.02, 0.03, 0.05, 0.10}
    (FREEZE-AMENDMENT-2026-07-13 Sec.4).

Reported per (n_cal, alpha) cell: fraction of the 20 reseeds that certify
non-vacuously, and -- among the certifying reseeds -- the mean lambda*,
mean calibration accepted-fraction, mean eval (fixed-test) acceptance rate,
mean eval accepted-set macro-CER, and whether any certifying reseed's
eval-side accepted macro-CER violates alpha (i.e. exceeds it; the accepted
macro-CER is descriptive here, not itself certified -- the LTT guarantee is
over the calibration draw at the stated (alpha,delta), same caveat as every
other calsweep in this repo).

Consistency guard: re-scores aishell_test_scored_canonical.jsonl with this
runner's pipeline and confirms it reproduces the frozen
aishell_test_scored.jsonl s1/CER exactly, proving the cal-pool scoring is on
the identical frozen scale as the eval scores (same guard as
mandarin_calsweep_2026-07-13).

zipformer (B4) is NOT swept: the pulled landscape digest
(landscape_pulled_2026-07-15/LANDSCAPE-DIGEST.md) reports it exposes
unusable posteriors on Aishell (45.88% full-set CER, s1 excess-AURC
NEGATIVE at -0.0158 -- worse than random abstention) and is vacuous at
every (corpus, alpha) in the frozen landscape. More calibration budget
cannot fix a degraded/uninformative score; the LTT bound only tightens
around whatever risk the accepted set actually has, and zipformer's s1
does not separate correct from incorrect hypotheses at all. Skipped, not
computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/zeyufu/Desktop/ml-reliability-research/reliability-commons/tools/asr-gate")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent.parent))  # reliability-commons/ (for `relmetrics`)

from asr_gate import cer as _cer        # noqa: E402
from asr_gate import ltt as _ltt        # noqa: E402
from asr_gate import scores as _scores  # noqa: E402

LANDSCAPE = ROOT / "landscape_pulled_2026-07-15" / "backbone_belle"
DEV_CANON = LANDSCAPE / "dev_canonical.jsonl"
TEST_SCORED = LANDSCAPE / "aishell_test_scored.jsonl"
TEST_CANON = LANDSCAPE / "aishell_test_scored_canonical.jsonl"

# Frozen G1 config (FREEZE-AMENDMENT-2026-07-13 Sec.4; identical to
# english_calsweep_2026-07-13 / mandarin_calsweep_2026-07-13 PRIMARY view).
DELTA = 0.1
N_GRID = 200
MIN_ACCEPT_FRAC = 0.1
PROCEDURE = "bonferroni"
P_VALUE = "eb"
NUMERAL_POLICY = "keep"

ALPHAS = [0.015, 0.02, 0.03, 0.05, 0.10]
# Requested anchor grid {250,500,1k,2k,4k,full} plus interpolation points added
# post-hoc (disclosed) to pin the alpha=5% crossover (which fell strictly
# between 1000 and 2000 on the anchor grid alone) and to trace the alpha=3%
# frontier between 4000 and the full pool. Same machinery, more grid density.
N_CAL_GRID_BASE = [250, 500, 1000, 1250, 1500, 1750, 2000, 3000, 4000,
                    6000, 8000, 10000, 12000]  # + "full" (pool size) appended at runtime
R_RESEEDS = 20
CERT_FRAC_TARGET = 0.90  # headline: min n_cal with >=90% of reseeds certifying at alpha=5%


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p)]


def score_pool(recs):
    """Frozen pipeline: s1-s4 (score_table) + CER (compute_cer_batch, keep).
    Keeps only ref-bearing utts with a usable s1 and cer (the cal/eval
    contract), matching english_calsweep / mandarin_calsweep exactly."""
    recs = [r for r in recs if r.get("ref_text")]
    scored = _scores.score_table(recs)
    scored = _cer.compute_cer_batch(scored, numeral_policy=NUMERAL_POLICY)
    return [r for r in scored if r.get("s1") is not None and r.get("cer") is not None]


def consistency_guard(test_scored):
    """Re-score the canonical test with THIS pipeline and confirm it
    reproduces the frozen aishell_test_scored.jsonl s1/CER exactly --
    proves the swept cal-pool scores are on the frozen scale."""
    if not TEST_CANON.exists():
        return {"checked": False, "reason": "aishell_test_scored_canonical.jsonl absent"}
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
              f"cer_max_abs_diff={cer_max:.3e}) -- interpret the sweep with care.",
              file=sys.stderr)
    return {"checked": True, "n_compared": n, "s1_max_abs_diff": s1_max,
            "cer_max_abs_diff": cer_max, "reproduces_frozen": bool(ok)}


def certify_flat(cal, alpha):
    losses = np.array([r["cer"] for r in cal], dtype=float)
    scores = np.array([r["s1"] for r in cal], dtype=float)
    return _ltt.ltt_certify(
        losses, scores, alpha=alpha, delta=DELTA, n_grid=N_GRID,
        min_accept_frac=MIN_ACCEPT_FRAC, procedure=PROCEDURE, p_value=P_VALUE,
    )


def eval_threshold(lambda_star, eval_recs):
    if lambda_star is None:
        return {"eval_accept_rate": 0.0, "eval_accepted_macro_cer": None, "eval_n_accept": 0}
    s1 = np.array([r["s1"] for r in eval_recs], dtype=float)
    cer = np.array([r["cer"] for r in eval_recs], dtype=float)
    mask = s1 >= lambda_star
    acc = float(mask.mean())
    return {
        "eval_accept_rate": acc,
        "eval_accepted_macro_cer": (float(cer[mask].mean()) if mask.any() else None),
        "eval_n_accept": int(mask.sum()),
    }


def main():
    dev_pool = score_pool(load_jsonl(DEV_CANON))
    test_recs = load_jsonl(TEST_SCORED)
    test_recs = [r for r in test_recs if r.get("s1") is not None and r.get("cer") is not None]
    full_macro = float(np.mean([r["cer"] for r in test_recs]))

    guard = consistency_guard(test_recs)

    n_cal_grid = list(N_CAL_GRID_BASE) + [len(dev_pool)]

    # Per (n_cal, alpha): 20 reseeds, each a fresh seeded shuffle of the FULL
    # dev pool truncated to n_cal utts (mirrors english/mandarin calsweep's
    # per-target-size seeded subsample, extended from their 5 seeds to 20 to
    # get a reseed-fraction power curve rather than a single crossover).
    cells = []
    per_reseed_raw = []  # flat list, every (n_cal, alpha, reseed) cell -- full transparency
    for n_target in n_cal_grid:
        n_cal = min(n_target, len(dev_pool))
        for alpha in ALPHAS:
            reseed_results = []
            for reseed in range(R_RESEEDS):
                rng = np.random.default_rng(reseed)
                order = rng.permutation(len(dev_pool))
                cal = [dev_pool[i] for i in order[:n_cal]]
                res = certify_flat(cal, alpha)
                ev = eval_threshold(res["lambda_star"], test_recs)
                non_vacuous = bool(res["certified"] and res["accepted_fraction"] > 0)
                viol = (ev["eval_accepted_macro_cer"] is not None
                        and ev["eval_accepted_macro_cer"] > alpha)
                row = {
                    "n_cal_target": n_target, "n_cal": n_cal, "alpha": alpha, "reseed": reseed,
                    "certified": bool(res["certified"]), "non_vacuous": non_vacuous,
                    "lambda_star": res["lambda_star"],
                    "cal_accepted_fraction": res["accepted_fraction"], "K": res.get("K"),
                    "eval_violation": bool(viol), **ev,
                }
                reseed_results.append(row)
                per_reseed_raw.append(row)

            n_cert = sum(1 for r in reseed_results if r["non_vacuous"])
            cert_frac = n_cert / R_RESEEDS
            cert_rows = [r for r in reseed_results if r["non_vacuous"]]
            any_violation = any(r["eval_violation"] for r in cert_rows)

            def _mean(key):
                vals = [r[key] for r in cert_rows if r[key] is not None]
                return float(np.mean(vals)) if vals else None

            cells.append({
                "n_cal_target": n_target, "n_cal": n_cal, "alpha": alpha,
                "n_reseeds": R_RESEEDS, "n_certified_non_vacuous": n_cert,
                "cert_fraction": cert_frac,
                "mean_lambda_star": _mean("lambda_star"),
                "mean_cal_accepted_fraction": _mean("cal_accepted_fraction"),
                "mean_eval_accept_rate": _mean("eval_accept_rate"),
                "mean_eval_accepted_macro_cer": _mean("eval_accepted_macro_cer"),
                "mean_eval_n_accept": _mean("eval_n_accept"),
                "any_eval_violation_among_certified": bool(any_violation),
            })

    # Headline: for each alpha, the smallest n_cal in the grid with
    # cert_fraction >= CERT_FRAC_TARGET.
    headline = {}
    for alpha in ALPHAS:
        alpha_cells = [c for c in cells if c["alpha"] == alpha]
        hit = next((c["n_cal"] for c in sorted(alpha_cells, key=lambda c: c["n_cal"])
                    if c["cert_fraction"] >= CERT_FRAC_TARGET), None)
        headline[f"{alpha}"] = hit

    out = {
        "config": {
            "delta": DELTA, "n_grid": N_GRID, "min_accept_frac": MIN_ACCEPT_FRAC,
            "procedure": PROCEDURE, "p_value": P_VALUE, "numeral_policy": NUMERAL_POLICY,
            "g1_score": "s1", "loss": "cer", "alphas": ALPHAS,
            "n_cal_grid": n_cal_grid, "r_reseeds": R_RESEEDS,
            "cert_frac_target": CERT_FRAC_TARGET,
            "backbone": "Belle-whisper-large-v3-zh (B3')", "corpus": "Aishell-1",
            "eval": "fixed official test (backbone_belle/aishell_test_scored.jsonl, "
                    "pulled+byte-verified 2026-07-15, frozen)",
            "cal_pool": "Aishell-1 dev (backbone_belle/dev_canonical.jsonl), scored once "
                        "with the frozen pipeline; speaker-disjoint from test by construction",
            "design": "hold eval fixed; for each n_cal in the grid, R=20 seeded subsample "
                      "reseeds of the dev pool; PRIMARY flat ltt_certify (mirrors "
                      "english_calsweep_2026-07-13 / mandarin_calsweep_2026-07-13 exactly); "
                      "report fraction of reseeds certifying non-vacuously per (n_cal, alpha) "
                      "-- a power curve, not a single-seed crossover (post-hoc, disclosed)",
            "zipformer_note": "not swept -- landscape digest reports degraded/unusable s1 "
                               "posteriors (Aishell full-set CER 45.88%, excess-AURC -0.0158, "
                               "vacuous at every grid alpha); more calibration budget cannot "
                               "fix an uninformative score",
        },
        "consistency_guard": guard,
        "eval_full_set_macro_cer": full_macro,
        "eval_n": len(test_recs),
        "cal_pool_size": len(dev_pool),
        "headline_min_n_cal_for_cert_frac_90pct": headline,
        "cells": cells,
        "cells_per_reseed": per_reseed_raw,
    }
    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ----
    print("=== Belle-whisper (B3') calibration-size POWER CURVE, Aishell-1 (0 GPU) ===")
    print(f"cal pool (dev) = {len(dev_pool)} utts;  eval (fixed test) = {len(test_recs)} utts")
    print(f"full-set test macro-CER (base rate) = {full_macro*100:.2f}%")
    print(f"consistency guard reproduces frozen test s1/CER: {guard.get('reproduces_frozen')} "
          f"(s1_max_diff={guard.get('s1_max_abs_diff')})")
    print(f"\nHEADLINE: minimum n_cal certifying non-vacuously in >={CERT_FRAC_TARGET*100:.0f}% "
          f"of {R_RESEEDS} reseeds, by alpha:")
    for alpha in ALPHAS:
        v = headline[f"{alpha}"]
        print(f"  alpha={alpha*100:>4.1f}%: n_cal = {v if v is not None else f'NONE up to {n_cal_grid[-1]}'}")
    print("\nfull power-curve grid (n_cal / alpha -> cert_fraction, mean eval accept, mean eval acc-CER):")
    for c in cells:
        mc = (f"{c['mean_eval_accepted_macro_cer']*100:.2f}%"
              if c["mean_eval_accepted_macro_cer"] is not None else "  -   ")
        ma = (f"{c['mean_eval_accept_rate']*100:.1f}%"
              if c["mean_eval_accept_rate"] is not None else "  -  ")
        print(f"  n_cal={c['n_cal']:>6} alpha={c['alpha']*100:>4.1f}%: "
              f"{c['n_certified_non_vacuous']:>2}/{c['n_reseeds']} "
              f"({c['cert_fraction']*100:>5.1f}%)  eval_acc={ma}  eval_acc_cer={mc}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
