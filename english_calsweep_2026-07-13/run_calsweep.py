#!/usr/bin/env python3
"""PART A / F1 kill: English-arm calibration-size sweep.

The frozen English arm certifies non-vacuously only at alpha=5% and is
VACUOUS at alpha in {1,2,3}% on all three backbones, on a 613-utterance
calibration carve (english_arm_fixed_2026-07-12 / english_arm2_2026-07-13,
n_cal=613 conformalize pool). The paper's original sentence framed that
sub-5% vacuity as "a property of the conservative distribution-free bound"
-- i.e. an intrinsic property. The red-team F1 finding falsifies that: the
LTT/EB bound slack shrinks like ~1/sqrt(n_cal), so with more calibration
data the SAME test-clean utterances certify tighter targets. This is a
calibration-BUDGET property, not an intrinsic-bound one.

This runner re-runs the exact frozen G1 machinery (asr_gate.ltt.ltt_certify,
EB p-value, Bonferroni-over-grid, delta=0.1, n_grid=200, min_accept_frac=0.1,
g1-score=s1, loss=CER) while varying the calibration sample size
n_cal in {613, 1000, 1500, 2000} at alpha in {0.02, 0.03, 0.05}, per
backbone (whisper large-v3, wav2vec2 base, wav2vec2 large LV60k). It reports,
per backbone and alpha, the smallest n_cal at which the certificate first
becomes non-vacuous.

Design (post-hoc, disclosed): to isolate the *calibration budget* axis we
hold the EVALUATION set fixed and vary only the calibration size:
  - a FIXED held-out eval block = the last speakers (seeded permutation)
    whose cumulative utterances first reach >= EVAL_MIN_UTTS; eval is
    speaker-disjoint from every calibration carve (mirrors the frozen
    speaker-disjoint cal/eval protocol);
  - the calibration candidate pool = the complementary speakers; for each
    target n_cal we take the first n_cal utterances of a seeded shuffle of
    that pool (exact n_cal, still speaker-disjoint from eval).
n_cal=613 reproduces the frozen carve size as the sweep's anchor point.
Same delta=0.1 EB/Bonferroni machinery throughout; no bound machinery changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _portal_commons_root():
    import os
    from pathlib import Path
    for key in ("COMMONS_ROOT", "RELIABILITY_COMMONS"):
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if p.is_dir():
                return p
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for cand in (parent / "reliability-commons", parent.parent / "reliability-commons"):
            if cand.is_dir():
                return cand
    raise RuntimeError(
        "Set COMMONS_ROOT to the reliability-commons checkout (or place it as a sibling of this repo)."
    )

def _portal_repo_root():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    return here

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # for relmetrics if needed

from asr_gate import ltt as _ltt  # noqa: E402

# Frozen G1 config (english arm run script + CLI defaults).
DELTA = 0.1
N_GRID = 200
MIN_ACCEPT_FRAC = 0.1
PROCEDURE = "bonferroni"
P_VALUE = "eb"
SEED = 0
ALPHAS = [0.02, 0.03, 0.05]
N_CAL_TARGETS = [613, 1000, 1500, 2000]
EVAL_MIN_UTTS = 500

BACKBONES = {
    "whisper": ROOT / "english_arm_fixed_2026-07-12" / "whisper_test-clean_scored.jsonl",
    "wav2vec2_base": ROOT / "english_arm_fixed_2026-07-12" / "wav2vec2_test-clean_scored.jsonl",
    "wav2vec2_large": ROOT / "english_arm2_2026-07-13" / "wav2vec2L_test-clean_scored.jsonl",
}


def load_scored(path: Path):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r.get("s1") is not None and r.get("cer") is not None]
    return recs


def build_splits(recs, seed=SEED):
    """Fixed eval block (speaker-disjoint) + calibration candidate pool."""
    by_spk = {}
    for r in recs:
        by_spk.setdefault(r["speaker_id"], []).append(r)
    speakers = sorted(by_spk)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(speakers)))
    # Reserve eval from the END of the permutation until >= EVAL_MIN_UTTS.
    eval_speakers, cal_speakers = [], []
    cum = 0
    for idx in reversed(perm):
        spk = speakers[idx]
        if cum < EVAL_MIN_UTTS:
            eval_speakers.append(spk)
            cum += len(by_spk[spk])
        else:
            cal_speakers.append(spk)
    eval_recs = [r for spk in eval_speakers for r in by_spk[spk]]
    cal_pool = [r for spk in cal_speakers for r in by_spk[spk]]
    # Seeded shuffle of the calibration candidate pool (utterance order).
    rng2 = np.random.default_rng(seed)
    order = rng2.permutation(len(cal_pool))
    cal_pool = [cal_pool[i] for i in order]
    return eval_recs, cal_pool, len(eval_speakers), len(cal_speakers)


def certify(cal_recs, alpha):
    losses = np.array([r["cer"] for r in cal_recs], dtype=float)
    scores = np.array([r["s1"] for r in cal_recs], dtype=float)
    res = _ltt.ltt_certify(
        losses, scores, alpha=alpha, delta=DELTA, n_grid=N_GRID,
        min_accept_frac=MIN_ACCEPT_FRAC, procedure=PROCEDURE, p_value=P_VALUE,
    )
    return res


def eval_at(lambda_star, eval_recs):
    if lambda_star is None:
        return {"eval_accept_rate": 0.0, "eval_accepted_macro_cer": None, "eval_n": len(eval_recs)}
    s1 = np.array([r["s1"] for r in eval_recs], dtype=float)
    cer = np.array([r["cer"] for r in eval_recs], dtype=float)
    mask = s1 >= lambda_star
    acc = float(mask.mean())
    acc_cer = float(cer[mask].mean()) if mask.any() else None
    return {"eval_accept_rate": acc, "eval_accepted_macro_cer": acc_cer, "eval_n": len(eval_recs)}


def main():
    out = {
        "config": {
            "delta": DELTA, "n_grid": N_GRID, "min_accept_frac": MIN_ACCEPT_FRAC,
            "procedure": PROCEDURE, "p_value": P_VALUE, "seed": SEED,
            "g1_score": "s1", "loss": "cer", "alphas": ALPHAS,
            "n_cal_targets": N_CAL_TARGETS, "eval_min_utts": EVAL_MIN_UTTS,
            "design": "fixed speaker-disjoint held-out eval; vary calibration size by "
                      "subsampling the complementary speaker pool (post-hoc, disclosed)",
        },
        "backbones": {},
        "crossover": {},
    }
    for bname, path in BACKBONES.items():
        recs = load_scored(path)
        eval_recs, cal_pool, n_eval_spk, n_cal_spk = build_splits(recs)
        brec = {
            "n_total": len(recs),
            "n_eval": len(eval_recs), "n_eval_speakers": n_eval_spk,
            "cal_pool_size": len(cal_pool), "n_cal_speakers": n_cal_spk,
            "eval_full_macro_cer": float(np.mean([r["cer"] for r in eval_recs])),
            "cells": [],
        }
        cross = {f"{a}": None for a in ALPHAS}
        for target in N_CAL_TARGETS:
            n_cal = min(target, len(cal_pool))
            cal = cal_pool[:n_cal]
            for alpha in ALPHAS:
                res = certify(cal, alpha)
                ev = eval_at(res["lambda_star"], eval_recs)
                non_vacuous = bool(res["certified"] and res["accepted_fraction"] > 0)
                cell = {
                    "n_cal_target": target, "n_cal": n_cal, "alpha": alpha,
                    "certified": bool(res["certified"]), "non_vacuous": non_vacuous,
                    "lambda_star": res["lambda_star"],
                    "cal_accepted_fraction": res["accepted_fraction"],
                    "K": res["K"],
                    **ev,
                }
                brec["cells"].append(cell)
                key = f"{alpha}"
                if non_vacuous and cross[key] is None:
                    cross[key] = n_cal
        out["backbones"][bname] = brec
        out["crossover"][bname] = cross
    # Seed-robustness of the crossover (seeds 0..4), to show the crossover
    # n_cal is not a single-draw artifact.
    out["crossover_seed_robustness"] = {"seeds": [0, 1, 2, 3, 4], "backbones": {}}
    for bname, path in BACKBONES.items():
        recs = load_scored(path)
        rob = {f"{a}": [] for a in ALPHAS}
        for seed in range(5):
            eval_recs, cal_pool, _, _ = build_splits(recs, seed=seed)
            for alpha in ALPHAS:
                xo = None
                for target in N_CAL_TARGETS:
                    n_cal = min(target, len(cal_pool))
                    res = certify(cal_pool[:n_cal], alpha)
                    if res["certified"] and res["accepted_fraction"] > 0:
                        xo = n_cal
                        break
                rob[f"{alpha}"].append(xo)
        out["crossover_seed_robustness"]["backbones"][bname] = rob
    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    # Console summary.
    print("=== calibration-size sweep: smallest n_cal that certifies non-vacuously ===")
    for bname in BACKBONES:
        print(f"\n[{bname}]  cal_pool={out['backbones'][bname]['cal_pool_size']}  "
              f"eval_n={out['backbones'][bname]['n_eval']} "
              f"(full-set eval macro-CER={out['backbones'][bname]['eval_full_macro_cer']:.4f})")
        for alpha in ALPHAS:
            xo = out["crossover"][bname][f"{alpha}"]
            print(f"  alpha={alpha}: crossover n_cal = {xo if xo is not None else 'NONE up to 2000'}")
        print("  per-cell (n_cal / alpha -> certified, cal_acc_frac, eval_acc, eval_acc_cer):")
        for c in out["backbones"][bname]["cells"]:
            accc = f"{c['eval_accepted_macro_cer']:.4f}" if c['eval_accepted_macro_cer'] is not None else "  -   "
            print(f"    n={c['n_cal']:>4} a={c['alpha']}: cert={str(c['certified']):>5} "
                  f"cal_acc={c['cal_accepted_fraction']:.3f} eval_acc={c['eval_accept_rate']:.3f} eval_acc_cer={accc}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
