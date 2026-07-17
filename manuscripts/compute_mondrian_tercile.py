#!/usr/bin/env python3
"""G2 — per-duration-tercile Mondrian analysis (CPU-only, cached artifacts).

Does the G1 selective-risk certificate hold WITHIN each duration tercile, or is
there heterogeneity to disclose? The deployed gate already uses per-duration-tercile
thresholds (gate.json strata.thresholds) and every cached applied_test.json decision
carries its `stratum` label; this script joins those decisions to per-utterance CER
and reports, per tercile, at the frozen alpha=0.02:
  - acceptance rate (accepted / tercile size on test)
  - coverage (accepted / all test)
  - accepted-set macro-CER  (the certified estimand, restricted to the tercile)
  - violation vs alpha, across all 20 reseeds
  - speaker-blocked bootstrap 95% CI on the accepted-set macro-CER (reference
    reseed 0), matching the paper's CI methodology (relmetrics.blocked_bootstrap,
    block=speaker, 2000 resamples, percentile).

Frozen inputs are READ-ONLY. Writes results/mondrian_tercile_analysis.json.
Requires the relmetrics venv:  ../.venv/bin/python compute_mondrian_tercile.py
"""
import json, os, sys, platform
from datetime import datetime, timezone
import numpy as np
from relmetrics import bootstrap as _bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main_results_2026-07-09")
RES = os.path.join(HERE, "results")
ALPHA = 0.02
N_RESEEDS = 20
TERCILES = ["dur0", "dur1", "dur2"]
TERCILE_LABEL = {"dur0": "short", "dur1": "medium", "dur2": "long"}


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def cer_speaker_maps(scored_path):
    cer, spk = {}, {}
    for r in load_jsonl(scored_path):
        cer[r["utt_id"]] = r["cer"]
        spk[r["utt_id"]] = r["speaker_id"]
    return cer, spk


CER, SPK = cer_speaker_maps(os.path.join(MAIN, "test_scored.jsonl"))
N_TEST_SCORED = len(CER)


def per_reseed(reseed):
    ap = json.load(open(os.path.join(MAIN, f"reseed_{reseed}", "applied_test.json")))
    decisions = ap["decisions"]
    n_total = ap["n"]
    out = {}
    for t in TERCILES:
        in_t = [d for d in decisions if d.get("stratum") == t]
        acc = [d["utt_id"] for d in in_t if d["action"] == "ACCEPT"]
        cers = [CER[u] for u in acc if u in CER]
        macro = float(np.mean(cers)) if cers else None
        out[t] = {
            "n_stratum_test": len(in_t),
            "n_accept": len(acc),
            "acceptance_rate_within_tercile": (len(acc) / len(in_t)) if in_t else 0.0,
            "coverage_of_all_test": len(acc) / n_total,
            "accepted_macro_cer": macro,
            "violation": bool(cers and macro > ALPHA),
        }
    # overall (all strata) accepted-set macro-CER for cross-check vs POST-ANALYSIS
    acc_all = [d["utt_id"] for d in decisions if d["action"] == "ACCEPT"]
    out["_overall"] = {
        "n_accept": len(acc_all),
        "accepted_macro_cer": float(np.mean([CER[u] for u in acc_all if u in CER])),
        "coverage": len(acc_all) / n_total,
    }
    return out


def blocked_ci_for_tercile(reseed, tercile):
    """Speaker-blocked bootstrap 95% CI on accepted-set macro-CER for one tercile."""
    ap = json.load(open(os.path.join(MAIN, f"reseed_{reseed}", "applied_test.json")))
    acc = [d["utt_id"] for d in ap["decisions"]
           if d["action"] == "ACCEPT" and d.get("stratum") == tercile]
    cers = np.array([CER[u] for u in acc], float)
    spk = np.array([SPK[u] for u in acc])
    b = _bootstrap.blocked_bootstrap(
        lambda x: float(np.mean(x)), [cers], block_ids=spk,
        n_boot=2000, seeds=(0,), ci_level=0.95, method="percentile",
    )
    return {"point": float(b["point"]), "ci_lo": float(b["ci"][0]),
            "ci_hi": float(b["ci"][1]), "n_blocks": int(b["n_blocks"]),
            "n_accept": int(len(cers))}


# ---- per-reseed sweep + aggregation -----------------------------------------
per = {r: per_reseed(r) for r in range(N_RESEEDS)}
agg = {}
for t in TERCILES:
    accr = [per[r][t]["acceptance_rate_within_tercile"] for r in range(N_RESEEDS)]
    cov = [per[r][t]["coverage_of_all_test"] for r in range(N_RESEEDS)]
    mc = [per[r][t]["accepted_macro_cer"] for r in range(N_RESEEDS)
          if per[r][t]["accepted_macro_cer"] is not None]
    viol = sum(per[r][t]["violation"] for r in range(N_RESEEDS))
    agg[t] = {
        "label": TERCILE_LABEL[t],
        "acceptance_rate_mean": float(np.mean(accr)),
        "acceptance_rate_min": float(np.min(accr)),
        "acceptance_rate_max": float(np.max(accr)),
        "coverage_mean": float(np.mean(cov)),
        "accepted_macro_cer_mean": float(np.mean(mc)),
        "accepted_macro_cer_min": float(np.min(mc)),
        "accepted_macro_cer_max": float(np.max(mc)),
        "violations_over_reseeds": int(viol),
        "n_reseeds": N_RESEEDS,
        "ref_reseed0_ci": blocked_ci_for_tercile(0, t),
    }

# reference-reseed-0 overall CI (cross-check vs paper headline 1.00% accepted-set CER)
ap0 = json.load(open(os.path.join(MAIN, "reseed_0", "applied_test.json")))
acc0 = [d["utt_id"] for d in ap0["decisions"] if d["action"] == "ACCEPT"]
c0 = np.array([CER[u] for u in acc0], float); s0 = np.array([SPK[u] for u in acc0])
b0 = _bootstrap.blocked_bootstrap(lambda x: float(np.mean(x)), [c0], block_ids=s0,
                                  n_boot=2000, seeds=(0,), ci_level=0.95, method="percentile")
overall_ci = {"point": float(b0["point"]), "ci_lo": float(b0["ci"][0]),
              "ci_hi": float(b0["ci"][1]), "n_blocks": int(b0["n_blocks"]),
              "n_accept": int(len(c0))}

result = {
    "analysis": "per-duration-tercile Mondrian (G2)",
    "alpha": ALPHA, "delta": 0.1, "backbone": "paraformer",
    "corpus": "aishell1_test", "n_reseeds": N_RESEEDS,
    "estimand": "accepted-set macro-CER restricted to each duration tercile",
    "tercile_edges_note": "duration-tercile edges are frozen on each reseed's calibration split and applied to test; test tercile sizes are therefore unequal and reseed-dependent",
    "all_terciles_control_holds": all(agg[t]["violations_over_reseeds"] == 0 for t in TERCILES),
    "per_tercile": agg,
    "reference_reseed0_overall": overall_ci,
    "per_reseed": {str(r): {t: per[r][t] for t in TERCILES + ["_overall"]} for r in range(N_RESEEDS)},
    "provenance": {
        "script": "manuscripts/compute_mondrian_tercile.py",
        "inputs_readonly": [
            "main_results_2026-07-09/reseed_{0..19}/applied_test.json",
            "main_results_2026-07-09/reseed_{0..19}/gate.json (strata thresholds)",
            "main_results_2026-07-09/test_scored.jsonl",
        ],
        "ci_method": "relmetrics.blocked_bootstrap, block=speaker_id, n_boot=2000, seed=0, percentile, 95%",
        "n_test_scored": N_TEST_SCORED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    },
}

os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "mondrian_tercile_analysis.json"), "w") as f:
    json.dump(result, f, indent=2)

# ---- console digest ---------------------------------------------------------
print(f"Per-duration-tercile Mondrian, Paraformer/Aishell test, alpha={ALPHA}, {N_RESEEDS} reseeds")
print(f"{'tercile':16s} {'accept%':>18s} {'accCER% (mean[min,max])':>26s} {'viol/20':>8s}   reseed0 CI")
for t in TERCILES:
    a = agg[t]
    ci = a["ref_reseed0_ci"]
    print(f"{t}/{a['label']:<11s} "
          f"{a['acceptance_rate_mean']*100:6.1f} [{a['acceptance_rate_min']*100:.1f},{a['acceptance_rate_max']*100:.1f}]  "
          f"{a['accepted_macro_cer_mean']*100:6.2f} [{a['accepted_macro_cer_min']*100:.2f},{a['accepted_macro_cer_max']*100:.2f}]  "
          f"{a['violations_over_reseeds']:>5d}/20   [{ci['ci_lo']*100:.2f},{ci['ci_hi']*100:.2f}]% (n={ci['n_accept']},blk={ci['n_blocks']})")
print(f"\nreference reseed-0 OVERALL accepted-set macro-CER = {overall_ci['point']*100:.4f}% "
      f"CI[{overall_ci['ci_lo']*100:.2f},{overall_ci['ci_hi']*100:.2f}]  (paper headline: 1.00%)")
print("all terciles control holds (0 violations every tercile x reseed):", result["all_terciles_control_holds"])
print("wrote", os.path.join(RES, "mondrian_tercile_analysis.json"))
