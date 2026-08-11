#!/usr/bin/env python3
"""PART B / red-team M2 repair: speaker-partition validation on Aishell-1.

The frozen main run's 20 reseeds (main_results_2026-07-09/reseed_*) resample
only the DEV cal/tune carve and apply every gate to the SAME fixed 7,176-utt /
20-speaker official test set. That check exercises *calibration-draw*
randomness but holds the EVAL speaker population fixed -- it is structurally
blind to speaker-level variability in the evaluated population (red-team M2).

This runner adds the missing axis: it repartitions the FULL Aishell-1 speaker
set (dev 40 speakers + official test 20 speakers = 60 speakers, all scored)
into a calibration cohort and a *disjoint* eval cohort, R=20 times with
different seeds, reruns calibrate+apply per partition, and reports whether the
alpha=2% certificate holds when the EVAL SPEAKERS themselves change.

Design (post-hoc, disclosed):
  - pool = dev_canonical (40 spk) + test_canonical (20 spk), scored once with
    the frozen pipeline (score_table + compute_cer_batch, numeral_policy=keep);
  - per seed r in 0..19: shuffle the 60 speakers, take the first 20 as the
    CAL COHORT (calibrate_gate auto-splits it by speaker into fit ~10 spk +
    conformalize ~10 spk, giving n_cal ~ 3567 -- the frozen scale) and the
    next 20 as the EVAL COHORT (~7,000 utts, the frozen test scale),
    speaker-disjoint from the cal cohort; the remaining 20 are unused that seed;
  - calibrate at alpha=2%, delta=0.1, strata=duration_tercile (frozen config),
    apply to the eval cohort, join ACCEPT decisions back to true per-utterance
    CER, and flag a VIOLATION iff the accepted-set macro-CER exceeds alpha.
Because this necessarily calibrates on partitions of the official test set, it
is a post-hoc validity CHECK of the speaker-exchangeability axis, not a
headline result; the main-run numbers still calibrate on dev only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from asr_gate import cer as _cer      # noqa: E402
from asr_gate import gate as _gate     # noqa: E402
from asr_gate import scores as _scores  # noqa: E402

DEV_CANON = ROOT / "pilot_results_2026-07-09" / "dev_canonical.jsonl"
TEST_CANON = ROOT / "main_results_2026-07-09" / "test_canonical.jsonl"
ALPHA = 0.02
DELTA = 0.1
R = 20
CAL_COHORT_SPK = 20   # frozen dev cal20 scale
EVAL_COHORT_SPK = 20  # frozen test scale
NUMERAL_POLICY = "keep"


def load_and_score():
    recs = []
    for p in (DEV_CANON, TEST_CANON):
        for line in open(p):
            recs.append(json.loads(line))
    # keep only ref-bearing utterances (calibrate/audit refuse without refs)
    recs = [r for r in recs if r.get("ref_text")]
    scored = _scores.score_table(recs)
    scored = _cer.compute_cer_batch(scored, numeral_policy=NUMERAL_POLICY)
    return scored


def accepted_stats(applied, eval_by_id):
    """Join ACCEPT decisions to true CER -> accepted-set macro/micro CER."""
    acc_ids = [d["utt_id"] for d in applied["decisions"] if d["action"] == "ACCEPT"]
    if not acc_ids:
        return {"accept_rate": applied["n_accept"] / applied["n"], "n_accept": 0,
                "accepted_macro_cer": None, "accepted_micro_cer": None}
    cers = np.array([eval_by_id[i]["cer"] for i in acc_ids], dtype=float)
    edits = np.array([eval_by_id[i].get("edits", np.nan) for i in acc_ids], dtype=float)
    reflens = np.array([eval_by_id[i].get("ref_len", np.nan) for i in acc_ids], dtype=float)
    micro = float(np.nansum(edits) / np.nansum(reflens)) if np.nansum(reflens) > 0 else None
    return {"accept_rate": applied["n_accept"] / applied["n"], "n_accept": len(acc_ids),
            "accepted_macro_cer": float(cers.mean()), "accepted_micro_cer": micro}


def main():
    scored = load_and_score()
    by_spk = {}
    for r in scored:
        by_spk.setdefault(r["speaker_id"], []).append(r)
    speakers = sorted(by_spk)
    assert len(speakers) == 60, f"expected 60 pooled speakers, got {len(speakers)}"

    partitions = []
    for seed in range(R):
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(len(speakers)))
        cal_spk = [speakers[i] for i in perm[:CAL_COHORT_SPK]]
        eval_spk = [speakers[i] for i in perm[CAL_COHORT_SPK:CAL_COHORT_SPK + EVAL_COHORT_SPK]]
        cal_cohort = [r for s in cal_spk for r in by_spk[s]]
        eval_cohort = [r for s in eval_spk for r in by_spk[s]]
        gate = _gate.calibrate_gate(
            cal_cohort, tune_instances=None, alpha=ALPHA, delta=DELTA,
            g1_score="s1", guarantee="ltt", strata=["duration_tercile"],
            fit_frac=0.5, numeral_policy=NUMERAL_POLICY, seed=seed,
        )
        applied = _gate.apply_gate(gate, [dict(r) for r in eval_cohort])
        eval_by_id = {r["utt_id"]: r for r in eval_cohort}
        st = accepted_stats(applied, eval_by_id)
        violation = (st["accepted_macro_cer"] is not None
                     and st["accepted_macro_cer"] > ALPHA)
        partitions.append({
            "seed": seed,
            "cal_speakers": len(cal_spk), "eval_speakers": len(eval_spk),
            "n_cal": gate["n_cal"], "n_fit": gate["n_fit"],
            "certified": bool(gate["g1"]["certified"]),
            "lambda_star": gate["g1"]["lambda_star"],
            "cal_accepted_fraction": gate["g1"]["accepted_fraction"],
            "eval_n": applied["n"], "eval_ood_refuse": applied["n_ood_refuse"],
            "eval_accept_rate": st["accept_rate"], "eval_n_accept": st["n_accept"],
            "eval_accepted_macro_cer": st["accepted_macro_cer"],
            "eval_accepted_micro_cer": st["accepted_micro_cer"],
            "violation": bool(violation),
            "ks_warn": applied["domain_fingerprint_check"]["warn"]
                       if applied["domain_fingerprint_check"] else None,
        })

    certified = [p for p in partitions if p["certified"]]
    macro = [p["eval_accepted_macro_cer"] for p in certified
             if p["eval_accepted_macro_cer"] is not None]
    micro = [p["eval_accepted_micro_cer"] for p in certified
             if p["eval_accepted_micro_cer"] is not None]
    acc = [p["eval_accept_rate"] for p in certified]
    n_viol = sum(1 for p in partitions if p["violation"])
    summary = {
        "alpha": ALPHA, "delta": DELTA, "R": R,
        "n_certified_partitions": len(certified),
        "n_vacuous_partitions": R - len(certified),
        "violations": n_viol,
        "violation_string": f"{n_viol}/{R}",
        "acceptance_range": [min(acc), max(acc)] if acc else None,
        "accepted_macro_cer_range": [min(macro), max(macro)] if macro else None,
        "accepted_micro_cer_range": [min(micro), max(micro)] if micro else None,
        "n_cal_range": [min(p["n_cal"] for p in partitions),
                        max(p["n_cal"] for p in partitions)],
        "eval_n_range": [min(p["eval_n"] for p in partitions),
                         max(p["eval_n"] for p in partitions)],
    }
    out = {
        "config": {
            "pool": "Aishell-1 dev (40 spk) + official test (20 spk) = 60 speakers",
            "cal_cohort_speakers": CAL_COHORT_SPK, "eval_cohort_speakers": EVAL_COHORT_SPK,
            "alpha": ALPHA, "delta": DELTA, "R": R, "strata": "duration_tercile",
            "numeral_policy": NUMERAL_POLICY, "backbone": "Paraformer-zh (B2)",
            "note": "post-hoc speaker-exchangeability check; calibrates on partitions "
                    "of the official test set, so it is a validity CHECK, not a headline result",
        },
        "summary": summary,
        "partitions": partitions,
    }
    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print("=== speaker-partition validation (Aishell-1, Paraformer, alpha=2%) ===")
    print(f"pool = 60 speakers (dev 40 + test 20); R={R} partitions; "
          f"cal cohort {CAL_COHORT_SPK} spk / eval cohort {EVAL_COHORT_SPK} spk (disjoint)")
    print(f"certified partitions: {len(certified)}/{R}  (vacuous: {R-len(certified)})")
    print(f"VIOLATIONS (accepted macro-CER > alpha): {n_viol}/{R}")
    if acc:
        print(f"acceptance range:          {min(acc)*100:.1f}%--{max(acc)*100:.1f}%")
    if macro:
        print(f"accepted macro-CER range:  {min(macro)*100:.2f}%--{max(macro)*100:.2f}%")
    if micro:
        print(f"accepted micro-CER range:  {min(micro)*100:.2f}%--{max(micro)*100:.2f}%")
    print(f"n_cal range: {summary['n_cal_range']}   eval_n range: {summary['eval_n_range']}")
    print("\nper-partition:")
    for p in partitions:
        mc = f"{p['eval_accepted_macro_cer']*100:.2f}%" if p['eval_accepted_macro_cer'] is not None else "  -  "
        print(f"  seed {p['seed']:>2}: n_cal={p['n_cal']:>4} cert={str(p['certified']):>5} "
              f"eval_n={p['eval_n']:>4} accept={p['eval_accept_rate']*100:5.1f}% "
              f"acc_macroCER={mc} viol={p['violation']}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
