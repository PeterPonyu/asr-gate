#!/usr/bin/env python3
"""Descriptive uncertainty for the C2 audit (referee round W2/W6).

Computes, POST-HOC and prereg-neutral, a speaker-blocked bootstrap 95% CI on each
cell's excess-AURC (the whole-curve advantage of a confidence score over analytic
random deferral), plus a non-median-split robustness check (W6): the accepted-set
mean CER at coverage 25/50/75% vs the random-deferral expectation.

Reads ONLY frozen scored JSONLs; every excess-AURC point estimate is cross-checked
against the frozen holm_audit_realized.json value. Requires the relmetrics venv:

    .venv/bin/python manuscripts/compute_audit_ci.py

Writes results/audit_ci.json.
"""
import json, os
import numpy as np
from relmetrics import aurc as _aurc
from relmetrics import bootstrap as _bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main_results_2026-07-09")
RES = os.path.join(HERE, "results")
EXP = os.path.join(RES, "expansion")

# (cell key, scored jsonl, matches holm backbone:condition) -------------------
SCORED = {
    ("paraformer", "aishell_clean"): os.path.join(MAIN, "test_scored.jsonl"),
    ("paraformer", "musan5db"): os.path.join(EXP, "scored", "aishell_paraformer_musan5db_scored.jsonl"),
    ("paraformer", "musan15db"): os.path.join(EXP, "scored", "aishell_paraformer_musan15db_scored.jsonl"),
    ("paraformer", "musan25db"): os.path.join(EXP, "scored", "aishell_paraformer_musan25db_scored.jsonl"),
    ("whisper", "aishell_clean"): os.path.join(EXP, "scored", "aishell_whisper_clean_scored.jsonl"),
    # 2026-07-12 fix: the original THCHS-30 decode (thchs30_whisper_scored.jsonl,
    # n=1339) was tail-truncated + repeated a ~250-utt subset; repointed to the
    # corrected full re-decode (n=2495, canonical scored schema from asr_gate.cli
    # ingest+score). Old file kept in-tree for provenance.
    ("whisper", "thchs30_crosscorpus"): os.path.join(EXP, "scored", "thchs30_whisper_scored_fixed_2026-07-12.jsonl"),
}


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def cell_arrays(rows, score_key):
    keep = [r for r in rows if r.get(score_key) is not None and r.get("cer") is not None]
    cer = np.array([r["cer"] for r in keep], float)
    score = np.array([r[score_key] for r in keep], float)
    spk = np.array([r.get("speaker_id", i) for i, r in enumerate(keep)])
    return cer, score, spk


def nonmedian_gaps(cer, score):
    """Accepted-set mean CER minus random-deferral expectation (= overall mean)
    at coverage 25/50/75%. Positive gap in our sign => accepted-set is BELOW
    random (score helps). We report random_minus_accepted so positive = score wins."""
    n = len(cer)
    order = np.argsort(-score)  # most confident first
    loss_sorted = cer[order]
    overall = float(cer.mean())
    out = {}
    for cov in (0.25, 0.50, 0.75):
        k = max(1, int(round(cov * n)))
        acc = float(loss_sorted[:k].mean())
        out[f"cov{int(cov*100)}"] = {
            "accepted_mean_cer": acc,
            "random_expectation": overall,
            "advantage": overall - acc,  # >0 => score beats random at this split
        }
    return out


# frozen reference values for the cross-check ---------------------------------
holm = json.load(open(os.path.join(EXP, "holm", "holm_audit_realized.json")))
ref = {(r["backbone"], r["condition"], r["score"]): r["excess_aurc"] for r in holm["rows"]}

out = {"n_boot": 2000, "ci_level": 0.95, "method": "percentile-blocked-by-speaker", "cells": {}}
mismatch = []
for (bb, cond), path in SCORED.items():
    rows = load_jsonl(path)
    for sk in ("s1", "s2"):
        cer, score, spk = cell_arrays(rows, sk)
        point = _aurc.excess_aurc_gain(cer, score)
        boot = _bootstrap.blocked_bootstrap(
            _aurc.excess_aurc_gain, [cer, score], block_ids=spk,
            n_boot=2000, seeds=(0,), ci_level=0.95, method="percentile",
        )
        rp = ref.get((bb, cond, sk))
        if rp is not None and abs(rp - point) > 5e-4:
            mismatch.append((bb, cond, sk, rp, point))
        out["cells"][f"{bb}:{cond}:{sk}"] = {
            "backbone": bb, "condition": cond, "score": sk, "n": int(len(cer)),
            "n_blocks": int(len(set(spk.tolist()))),
            "excess_aurc": float(point),
            "excess_aurc_frozen_ref": rp,
            "ci_lo": float(boot["ci"][0]), "ci_hi": float(boot["ci"][1]),
            "ci_excludes_zero": bool(boot["ci"][0] > 0),
            "nonmedian": nonmedian_gaps(cer, score),
        }

out["all_ci_exclude_zero"] = all(c["ci_excludes_zero"] for c in out["cells"].values())
out["all_nonmedian_positive"] = all(
    g["advantage"] > 0 for c in out["cells"].values() for g in c["nonmedian"].values()
)
out["point_estimate_crosscheck_ok"] = len(mismatch) == 0

with open(os.path.join(RES, "audit_ci.json"), "w") as f:
    json.dump(out, f, indent=2)

print("cross-check vs frozen holm excess-AURC:", "OK" if not mismatch else mismatch)
print(f"{'cell':40s} {'exAURC':>8s}  95% CI (speaker-blocked)   excl0  nonmedian@25/50/75")
for k, c in out["cells"].items():
    nm = c["nonmedian"]
    tag = "+".join("Y" if nm[f'cov{p}']['advantage'] > 0 else "N" for p in (25, 50, 75))
    print(f"{k:40s} {c['excess_aurc']:8.4f}  [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]   "
          f"{'Y' if c['ci_excludes_zero'] else 'N':>3s}    {tag}  (nblk={c['n_blocks']})")
print("\nall CIs exclude zero:", out["all_ci_exclude_zero"])
print("all non-median advantages positive:", out["all_nonmedian_positive"])
print("wrote", os.path.join(RES, "audit_ci.json"))
