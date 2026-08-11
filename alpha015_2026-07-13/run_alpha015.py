#!/usr/bin/env python3
"""PART C / red-team M1 repair: the alpha=1.5% binding-regime point.

The frozen main run certifies alpha=2% with 0/20 violations, but the full-set
Paraformer macro-CER on clean Aishell-1 is 1.98% (< 2%), so accept-ALL already
meets a 2% budget -- the certificate's *information* on clean data is the
CONDITIONAL accepted-set quality and the binding regimes, not "0 violations at
2%". Red-team M1 turns that into a "thin certified contribution" attack.

This runner adds the BINDING point the frozen ladder skips. alpha=1% is vacuous
and alpha=2% certifies at ~86% acceptance; alpha=1.5% sits BELOW the 1.98% base
rate, so accept-all cannot satisfy it and the target genuinely BINDS. It
replicates the exact frozen main protocol -- reuse each reseed's dev cal carve
(main_results_2026-07-09/reseed_*/cal20_scored.jsonl), calibrate at
alpha=1.5%, delta=0.1, strata=duration_tercile, apply to the FIXED official
test set (test_scored.jsonl) -- across all 20 reseeds, changing only alpha.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from asr_gate import gate as _gate  # noqa: E402

ALPHA = 0.015
DELTA = 0.1
R = 20
TEST_SCORED = ROOT / "main_results_2026-07-09" / "test_scored.jsonl"
RESEED_DIR = ROOT / "main_results_2026-07-09"
FULL_SET_MACRO_CER = 0.019805491935060385  # numbers.json: main_audit_fixed/macro_cer


def load(p):
    return [json.loads(l) for l in open(p)]


def accepted_stats(applied, by_id):
    acc_ids = [d["utt_id"] for d in applied["decisions"] if d["action"] == "ACCEPT"]
    if not acc_ids:
        return {"n_accept": 0, "accepted_macro_cer": None, "accepted_micro_cer": None}
    cers = np.array([by_id[i]["cer"] for i in acc_ids], dtype=float)
    edits = np.array([by_id[i].get("edits", np.nan) for i in acc_ids], dtype=float)
    reflens = np.array([by_id[i].get("ref_len", np.nan) for i in acc_ids], dtype=float)
    micro = float(np.nansum(edits) / np.nansum(reflens)) if np.nansum(reflens) > 0 else None
    return {"n_accept": len(acc_ids), "accepted_macro_cer": float(cers.mean()),
            "accepted_micro_cer": micro}


def main():
    test = load(TEST_SCORED)
    by_id = {r["utt_id"]: r for r in test}
    reseeds = []
    for seed in range(R):
        cal = load(RESEED_DIR / f"reseed_{seed}" / "cal20_scored.jsonl")
        gate = _gate.calibrate_gate(
            cal, tune_instances=None, alpha=ALPHA, delta=DELTA,
            g1_score="s1", guarantee="ltt", strata=["duration_tercile"],
            fit_frac=0.5, numeral_policy="keep", seed=seed,
        )
        applied = _gate.apply_gate(gate, [dict(r) for r in test])
        st = accepted_stats(applied, by_id)
        accept_rate = applied["n_accept"] / applied["n"]
        violation = (st["accepted_macro_cer"] is not None
                     and st["accepted_macro_cer"] > ALPHA)
        reseeds.append({
            "seed": seed, "n_cal": gate["n_cal"], "n_fit": gate["n_fit"],
            "certified": bool(gate["g1"]["certified"]),
            "lambda_star": gate["g1"]["lambda_star"],
            "cal_accepted_fraction": gate["g1"]["accepted_fraction"],
            "eval_n": applied["n"], "eval_n_accept": st["n_accept"],
            "eval_accept_rate": accept_rate,
            "eval_accepted_macro_cer": st["accepted_macro_cer"],
            "eval_accepted_micro_cer": st["accepted_micro_cer"],
            "violation": bool(violation),
        })

    certified = [r for r in reseeds if r["certified"]]
    macro = [r["eval_accepted_macro_cer"] for r in certified if r["eval_accepted_macro_cer"] is not None]
    micro = [r["eval_accepted_micro_cer"] for r in certified if r["eval_accepted_micro_cer"] is not None]
    acc = [r["eval_accept_rate"] for r in certified]
    n_viol = sum(1 for r in reseeds if r["violation"])
    summary = {
        "alpha": ALPHA, "delta": DELTA, "R": R,
        "full_set_macro_cer": FULL_SET_MACRO_CER,
        "target_binds": ALPHA < FULL_SET_MACRO_CER,
        "n_certified": len(certified), "n_vacuous": R - len(certified),
        "violations": n_viol, "violation_string": f"{n_viol}/{R}",
        "acceptance_range": [min(acc), max(acc)] if acc else None,
        "acceptance_mean": float(np.mean(acc)) if acc else None,
        "accepted_macro_cer_range": [min(macro), max(macro)] if macro else None,
        "accepted_macro_cer_mean": float(np.mean(macro)) if macro else None,
        "accepted_micro_cer_range": [min(micro), max(micro)] if micro else None,
    }
    # --- binding-band sweep: locate the tightest alpha that still certifies ---
    # non-vacuously at the frozen n_cal, across all 20 reseeds. Maps the
    # certificate's non-trivial operating band around the 1.98% base rate.
    band_alphas = [0.015, 0.016, 0.017, 0.018, 0.019, 0.020]
    cal_cache = [load(RESEED_DIR / f"reseed_{s}" / "cal20_scored.jsonl") for s in range(R)]
    band = []
    for a in band_alphas:
        certs, accs, macros = 0, [], []
        for seed in range(R):
            g = _gate.calibrate_gate(
                cal_cache[seed], tune_instances=None, alpha=a, delta=DELTA,
                g1_score="s1", guarantee="ltt", strata=["duration_tercile"],
                fit_frac=0.5, numeral_policy="keep", seed=seed,
            )
            if g["g1"]["certified"]:
                certs += 1
                ap = _gate.apply_gate(g, [dict(r) for r in test])
                st = accepted_stats(ap, by_id)
                accs.append(ap["n_accept"] / ap["n"])
                if st["accepted_macro_cer"] is not None:
                    macros.append(st["accepted_macro_cer"])
        band.append({
            "alpha": a, "binds_vs_base_rate": a < FULL_SET_MACRO_CER,
            "certified": f"{certs}/{R}",
            "mean_acceptance": float(np.mean(accs)) if accs else None,
            "mean_accepted_macro_cer": float(np.mean(macros)) if macros else None,
        })

    out = {
        "config": {
            "alpha": ALPHA, "delta": DELTA, "R": R, "strata": "duration_tercile",
            "backbone": "Paraformer-zh (B2)", "corpus": "Aishell-1 clean",
            "protocol": "frozen main reseed loop (dev cal carve -> apply to fixed "
                        "official test) at alpha=1.5% (only alpha changed vs the "
                        "frozen 2% run)",
            "full_set_macro_cer": FULL_SET_MACRO_CER,
            "note": "alpha=1.5% < full-set 1.98% base rate, so accept-all cannot "
                    "satisfy the budget -- the target BINDS (unlike clean alpha>=2%)",
        },
        "summary": summary, "binding_band_sweep": band, "reseeds": reseeds,
    }
    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print("=== alpha=1.5% binding-regime point (Aishell-1, Paraformer, clean) ===")
    print(f"full-set macro-CER = {FULL_SET_MACRO_CER*100:.2f}%  >  alpha=1.5%  =>  target BINDS")
    print(f"certified reseeds: {len(certified)}/{R}  (vacuous: {R-len(certified)})")
    print(f"VIOLATIONS (accepted macro-CER > 1.5%): {n_viol}/{R}")
    if acc:
        print(f"acceptance: {min(acc)*100:.1f}%--{max(acc)*100:.1f}%  (mean {np.mean(acc)*100:.1f}%)")
    if macro:
        print(f"accepted macro-CER: {min(macro)*100:.2f}%--{max(macro)*100:.2f}%  (mean {np.mean(macro)*100:.2f}%)")
    if micro:
        print(f"accepted micro-CER: {min(micro)*100:.2f}%--{max(micro)*100:.2f}%")
    print("\nper-reseed:")
    for r in reseeds:
        mc = f"{r['eval_accepted_macro_cer']*100:.2f}%" if r['eval_accepted_macro_cer'] is not None else "  -  "
        print(f"  seed {r['seed']:>2}: n_cal={r['n_cal']} cert={str(r['certified']):>5} "
              f"accept={r['eval_accept_rate']*100:5.1f}% acc_macroCER={mc} viol={r['violation']}")
    print("\n=== binding-band sweep (tightest alpha that still certifies, n_cal~3567) ===")
    print("  alpha  binds?  certified   mean_accept  mean_accepted_macroCER")
    for b in band:
        ma = f"{b['mean_acceptance']*100:5.1f}%" if b['mean_acceptance'] is not None else "   -  "
        mc = f"{b['mean_accepted_macro_cer']*100:.2f}%" if b['mean_accepted_macro_cer'] is not None else "  -  "
        print(f"  {b['alpha']*100:.1f}%   {str(b['binds_vs_base_rate']):>5}   {b['certified']:>6}      {ma}       {mc}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
