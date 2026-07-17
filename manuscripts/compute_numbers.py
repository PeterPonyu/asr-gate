#!/usr/bin/env python3
"""asr-gate manuscript: compute every quoted number from frozen result JSONs.

Reads ONLY on-disk frozen result files (main_results_2026-07-09 + the extracted
expansion result tree staged under results/), never hardcoded arrays. Emits
results/numbers.json (consumed by the figure scripts and cross-checked against
the paper body). House rule: every number in the paper traces to this file's
output, which in turn traces to a frozen result JSON.

Run:  python3 compute_numbers.py
"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "main_results_2026-07-09")
RES = os.path.join(HERE, "results")
EXP = os.path.join(RES, "expansion")


def load(p):
    with open(p) as f:
        return json.load(f)


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cer_map(scored_path):
    """utt_id -> per-utterance clipped CER (the certified loss)."""
    m = {}
    for r in load_jsonl(scored_path):
        m[r["utt_id"]] = r["cer"]
    return m


def accepted_set_stats(applied_path, cmap, alpha, capped_utts=None):
    """Join ACCEPT decisions to per-utterance CER; return acc-set macro-CER etc.

    When ``capped_utts`` is given (FREEZE-AMENDMENT-2026-07-13 §2 speaker-disjoint
    ~4k test cap, applied post-hoc to the cached MagicData scores because the box
    decode/apply ran the full ~24k test set), the decisions are restricted to that
    speaker-disjoint capped eval set and every count/fraction is recomputed over it.
    """
    a = load(applied_path)
    decisions = a["decisions"]
    if capped_utts is not None:
        decisions = [d for d in decisions if d["utt_id"] in capped_utts]
        n_total = len(decisions)
        n_accept = sum(1 for d in decisions if d["action"] == "ACCEPT")
        n_defer = sum(1 for d in decisions if d["action"] == "DEFER")
        n_ood = n_total - n_accept - n_defer
    else:
        n_total = a["n"]
        n_accept = a["n_accept"]
        n_defer = a["n_defer"]
        n_ood = a["n_ood_refuse"]
    acc = [d["utt_id"] for d in decisions if d["action"] == "ACCEPT"]
    cers = [cmap[u] for u in acc if u in cmap]
    frac = len(acc) / n_total if n_total else 0.0
    macro = float(np.mean(cers)) if cers else None
    return {
        "n_accept": n_accept,
        "n_defer": n_defer,
        "n_ood_refuse": n_ood,
        "acc_fraction": frac,
        "acc_set_macro_cer": macro,
        "violation": bool(cers and macro > alpha),
        "vacuous": len(acc) == 0,
    }


def risk_coverage(scored_path, score_key):
    """Sort by score descending (higher score = more confident = accept first);
    return coverage grid and accepted-set macro-CER at each coverage. Also the
    analytic random-deferral curve (constant = full-set macro-CER) and oracle
    (sort by true CER ascending)."""
    rows = [r for r in load_jsonl(scored_path) if r.get(score_key) is not None]
    s = np.array([r[score_key] for r in rows])
    loss = np.array([r["cer"] for r in rows])
    n = len(rows)
    order = np.argsort(-s)  # most confident first
    loss_sorted = loss[order]
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(loss_sorted) / np.arange(1, n + 1)
    # oracle: accept lowest-CER first
    oracle_sorted = np.sort(loss)
    oracle_risk = np.cumsum(oracle_sorted) / np.arange(1, n + 1)
    full_macro = float(np.mean(loss))
    # subsample to <=200 points for plotting
    idx = np.linspace(0, n - 1, min(200, n)).astype(int)
    return {
        "n": n,
        "coverage": cov[idx].tolist(),
        "risk": risk[idx].tolist(),
        "oracle_risk": oracle_risk[idx].tolist(),
        "random_line": full_macro,
        "full_macro_cer": full_macro,
    }


def summarise(per, backbone):
    """Aggregate per-reseed accepted-set stats; handle vacuous (0 accepted) arms."""
    fr = [x["acc_fraction"] for x in per]
    mc = [x["acc_set_macro_cer"] for x in per if x["acc_set_macro_cer"] is not None]
    vac = all(x["vacuous"] for x in per)
    return {
        "backbone": backbone, "n_reseeds": len(per), "alpha": 0.02,
        "vacuous": vac,
        "acc_fraction_mean": float(np.mean(fr)), "acc_fraction_min": float(np.min(fr)), "acc_fraction_max": float(np.max(fr)),
        "acc_set_macro_cer_mean": float(np.mean(mc)) if mc else None,
        "acc_set_macro_cer_min": float(np.min(mc)) if mc else None,
        "acc_set_macro_cer_max": float(np.max(mc)) if mc else None,
        "violations": int(sum(x["violation"] for x in per)),
        "per_reseed": per,
    }


out = {}

# ------------------------------------------------------------------ MAIN RUN
pa = os.path.join(MAIN, "post_analysis")
attain = load(os.path.join(pa, "attainment_table.json"))
out["main_attainment"] = attain
# derive summary
rows = attain["rows"] if isinstance(attain, dict) and "rows" in attain else attain
# attainment_table.json structure unknown; normalise below after inspection
out["main_alpha_frontier"] = load(os.path.join(pa, "alpha_frontier.json"))
out["main_audit_fixed"] = load(os.path.join(pa, "audit_test_fixed.json"))

# ------------------------------------------------------------------ EXPANSION AUDIT
out["holm_realized"] = load(os.path.join(EXP, "holm", "holm_audit_realized.json"))
aud = {}
for p in sorted(glob.glob(os.path.join(EXP, "audit", "*.json"))):
    d = load(p)
    key = os.path.basename(p).replace("audit_", "").replace(".json", "")
    mc = d.get("micro_cer")
    aud[key] = {
        "n": d.get("n"),
        "macro_cer": d.get("macro_cer"),
        "micro_cer": mc.get("point") if isinstance(mc, dict) else mc,
        "micro_ci": mc.get("ci") if isinstance(mc, dict) else None,
        "results": [
            {"score": r["score"], "excess_aurc": r["excess_aurc"], "p_value": r["p_value"]}
            for r in d.get("results", [])
        ],
    }
out["expansion_audit"] = aud

# ------------------------------------------------------------------ EXPANSION ATTAINMENT
scored = {
    "musan5db": os.path.join(EXP, "scored", "aishell_paraformer_musan5db_scored.jsonl"),
    "musan15db": os.path.join(EXP, "scored", "aishell_paraformer_musan15db_scored.jsonl"),
    "musan25db": os.path.join(EXP, "scored", "aishell_paraformer_musan25db_scored.jsonl"),
    # 2026-07-12 fix: repointed from the tail-truncated original THCHS decode
    # (n=1339) to the corrected full re-decode (n=2495); feeds both the THCHS
    # attainment lookup and the whisper_thchs30 risk-coverage curve below.
    "thchs30": os.path.join(EXP, "scored", "thchs30_whisper_scored_fixed_2026-07-12.jsonl"),
}
cmaps = {k: cer_map(v) for k, v in scored.items()}

exp_attain = {}
# noise arms: paraformer gate calibrated on clean Aishell dev, applied to noisy test
for cond in ["musan5db", "musan15db", "musan25db"]:
    per = []
    for r in range(20):
        ap = os.path.join(EXP, "backbone_paraformer", f"reseed_{r}", f"applied_{cond}_alpha0.02.json")
        per.append(accepted_set_stats(ap, cmaps[cond], 0.02))
    exp_attain[cond] = summarise(per, "paraformer")
# cross-corpus: whisper gate calibrated on Aishell dev, applied to THCHS-30 test
per = []
for r in range(20):
    ap = os.path.join(EXP, "backbone_whisper", f"reseed_{r}", "applied_thchs30_alpha0.02.json")
    per.append(accepted_set_stats(ap, cmaps["thchs30"], 0.02))
exp_attain["thchs30_whisper"] = summarise(per, "whisper")
out["expansion_attainment"] = exp_attain

# ------------------------------------------------------------------ RISK-COVERAGE (figures)
rc = {}
rc["paraformer_clean"] = risk_coverage(os.path.join(MAIN, "test_scored.jsonl"), "s1")
rc["paraformer_musan5db"] = risk_coverage(scored["musan5db"], "s1")
rc["paraformer_musan15db"] = risk_coverage(scored["musan15db"], "s1")
rc["paraformer_musan25db"] = risk_coverage(scored["musan25db"], "s1")
rc["whisper_clean"] = risk_coverage(os.path.join(EXP, "scored", "aishell_whisper_clean_scored.jsonl"), "s1")
rc["whisper_thchs30"] = risk_coverage(scored["thchs30"], "s1")
out["risk_coverage"] = rc

# ------------------------------------------------------------------ LANDSCAPE (2026-07-13)
# Registry stubs for the FREEZE-AMENDMENT-2026-07-13 backbone x corpus landscape
# (COMPUTE-PLAN-2026-07-13.md). This block is ADDITIVE and fully GUARDED: it
# never alters any existing key above and never raises if the forthcoming box
# result JSONs are absent -- it records each landscape cell as "pending" with
# the exact expected path (so the number-tracing pipeline covers them the moment
# they land), and reads whichever cells DO exist with the same accepted-set /
# audit machinery the expansion section uses. It also traces the CACHED
# (0-GPU, already-produced) Mandarin calibration-pool sweep result.
LANDSCAPE = os.path.join(RES, "landscape")  # box results pulled here (mirrors results/expansion)
# THCHS-30 official-test re-decode (2026-07-16): the frozen landscape THCHS cells
# above were decoded on the n=1339 mislabeled mirror pool (DATA-AUDIT finding C2);
# the corrected n=2495 official D-prefix test re-decode was pulled under
# thchs_official_pulled_2026-07-16/. Both trees stay on disk (frozen mirror
# artifacts are NOT overwritten); the landscape THCHS reads are repointed here so
# tab:landscape / tab:data quote the official pool. File names carry the
# "thchs30_official" infix (scored, applied_*, audit_*_official).
THCHS_OFFICIAL = os.path.join(HERE, "..", "thchs_official_pulled_2026-07-16")
LANDSCAPE_BACKBONES = ["paraformer", "belle", "zipformer"]  # B2, B3', B4 (FREEZE-AMENDMENT §1)
LANDSCAPE_CORPORA = ["aishell", "thchs30", "aidatatang", "magicdata"]
LANDSCAPE_ALPHAS = [0.015, 0.02, 0.03, 0.05, 0.10]  # binding probe + frozen grid (§4)
LANDSCAPE_N_RESEEDS = 20
# On-disk alpha labels (the box writes applied_*_alpha0.10.json with the trailing
# zero, which str(0.10)=='0.1' would MISS); keep the label table explicit.
LANDSCAPE_ALPHA_LABELS = {0.015: "0.015", 0.02: "0.02", 0.03: "0.03",
                          0.05: "0.05", 0.10: "0.10"}
# aidatatang_200zh (SLR62) was WITHDRAWN upstream on launch day; MagicData (SLR68)
# substituted per the SIGNED FREEZE-AMENDMENT §D1 (disclosed deviation). Its cells
# stay honestly pending-withdrawn, never realized.
LANDSCAPE_WITHDRAWN = {"aidatatang"}
# MagicData (SLR68) box decode/apply/audit ran the FULL ~24k test set; the frozen
# §2 speaker-disjoint ~4k eval cap (MAGICDATA_TEST_CAP=4000) is applied post-hoc to
# the cached scores here (attainment) and in capped_magicdata_2026-07-15/ (audit).
LANDSCAPE_CAPPED = {"magicdata"}
LANDSCAPE_TEST_CAP = 4000
LANDSCAPE_CAP_DIR = os.path.join(LANDSCAPE, "capped_magicdata_2026-07-15")


def _safe_load(path):
    try:
        return load(path)
    except (OSError, ValueError):
        return None


def _capped_utts(scored_path, cap=LANDSCAPE_TEST_CAP):
    """Reproduce the FREEZE-AMENDMENT §2 cap deterministically: order the test
    speakers by ascending speaker-id (string sort), accumulate WHOLE speakers until
    the cumulative utterance count first reaches ``cap``, and return the utt_id set
    of exactly those speakers (last speaker included whole)."""
    by_spk = {}
    for r in load_jsonl(scored_path):
        by_spk.setdefault(str(r["speaker_id"]), []).append(r["utt_id"])
    utts = set()
    cum = 0
    for spk in sorted(by_spk):
        utts.update(by_spk[spk])
        cum += len(by_spk[spk])
        if cum >= cap:
            break
    return utts


def _landscape_attainment_cell(backbone, corpus, alpha):
    """Accepted-set stats for one (backbone, corpus, alpha) cell across reseeds,
    if the box wrote it; else a pending stub naming the expected inputs."""
    if corpus in LANDSCAPE_WITHDRAWN:
        return {"status": "pending-withdrawn", "corpus": corpus,
                "note": "SLR62 withdrawn upstream; MagicData substituted (§D1)"}
    label = LANDSCAPE_ALPHA_LABELS[alpha]
    if corpus == "thchs30":
        # official n=2495 re-decode tree (2026-07-16); "thchs30_official" infix
        bb_dir = os.path.join(THCHS_OFFICIAL, f"backbone_{backbone}")
        scored = os.path.join(bb_dir, "thchs30_official_test_scored.jsonl")
        applied_glob = os.path.join(bb_dir, "reseed_*", f"applied_thchs30_official_alpha{label}.json")
    else:
        bb_dir = os.path.join(LANDSCAPE, f"backbone_{backbone}")
        scored = os.path.join(bb_dir, f"{corpus}_test_scored.jsonl")
        applied_glob = os.path.join(bb_dir, "reseed_*", f"applied_{corpus}_alpha{label}.json")
    applied_paths = sorted(glob.glob(applied_glob))
    if not (os.path.exists(scored) and applied_paths):
        return {"status": "pending", "expected_scored": scored,
                "expected_applied_glob": applied_glob}
    cmap = cer_map(scored)
    capped = _capped_utts(scored) if corpus in LANDSCAPE_CAPPED else None
    per = [accepted_set_stats(p, cmap, alpha, capped_utts=capped) for p in applied_paths]
    summ = summarise(per, backbone)
    summ["alpha"] = alpha
    summ["status"] = "realized"
    summ["corpus"] = corpus
    summ["eval_cap_applied"] = bool(capped)
    if capped is not None:
        summ["eval_n_capped"] = len(capped)
    return summ


def _landscape_audit_cell(backbone, corpus):
    """Excess-AURC audit for one (backbone, corpus) cell, if present. MagicData
    reads the post-hoc capped audit (capped_magicdata_2026-07-15/)."""
    if corpus in LANDSCAPE_WITHDRAWN:
        return {"status": "pending-withdrawn", "corpus": corpus}
    if corpus == "thchs30":
        # official n=2495 re-decode audit (2026-07-16)
        path = os.path.join(THCHS_OFFICIAL, "audit", f"audit_thchs30_official_{backbone}.json")
    elif corpus in LANDSCAPE_CAPPED:
        path = os.path.join(LANDSCAPE_CAP_DIR, f"audit_{corpus}_{backbone}.json")
    else:
        path = os.path.join(LANDSCAPE, "audit", f"audit_{corpus}_{backbone}.json")
    d = _safe_load(path)
    if d is None:
        return {"status": "pending", "expected_audit": path}
    mc = d.get("micro_cer")
    return {
        "status": "realized", "n": d.get("n"), "macro_cer": d.get("macro_cer"),
        "micro_cer": mc.get("point") if isinstance(mc, dict) else mc,
        "micro_ci": mc.get("ci") if isinstance(mc, dict) else None,
        "holm_family_size": d.get("holm_family_size"),
        "eval_cap_applied": corpus in LANDSCAPE_CAPPED,
        "results": [{"score": r["score"], "excess_aurc": r["excess_aurc"],
                     "p_value": r["p_value"]} for r in d.get("results", [])],
    }


landscape = {"results_dir": LANDSCAPE, "attainment": {}, "audit": {},
             "n_pending": 0, "n_realized": 0, "n_withdrawn": 0,
             "magicdata_eval_cap": {"rule": "FREEZE-AMENDMENT §2 speaker-disjoint",
                                    "cap": LANDSCAPE_TEST_CAP,
                                    "audit_dir": "capped_magicdata_2026-07-15"}}
for _bb in LANDSCAPE_BACKBONES:
    for _corpus in LANDSCAPE_CORPORA:
        for _a in LANDSCAPE_ALPHAS:
            cell = _landscape_attainment_cell(_bb, _corpus, _a)
            landscape["attainment"][f"{_bb}_{_corpus}_a{_a}"] = cell
            st = cell.get("status")
            if st == "realized":
                landscape["n_realized"] += 1
            elif st == "pending-withdrawn":
                landscape["n_withdrawn"] += 1
            else:
                landscape["n_pending"] += 1
        acell = _landscape_audit_cell(_bb, _corpus)
        landscape["audit"][f"{_bb}_{_corpus}"] = acell
# Trace the cached (0-GPU) Mandarin calibration-pool sweep, if present.
_calsweep = _safe_load(os.path.join(HERE, "..", "mandarin_calsweep_2026-07-13", "results.json"))
if _calsweep is not None:
    landscape["mandarin_calsweep"] = {
        "eval_full_set_macro_cer": _calsweep.get("eval_full_set_macro_cer"),
        "cal_pool_size": _calsweep.get("cal_pool_size"),
        "crossover_seed0": _calsweep.get("crossover_seed0"),
        "crossover_seed_robustness": _calsweep.get("crossover_seed_robustness"),
        "frozen_machinery_confirmation": _calsweep.get("frozen_machinery_confirmation"),
        "consistency_guard": _calsweep.get("consistency_guard"),
    }
else:
    landscape["mandarin_calsweep"] = {"status": "pending",
        "expected": "mandarin_calsweep_2026-07-13/results.json"}
out["landscape"] = landscape

with open(os.path.join(RES, "numbers.json"), "w") as f:
    json.dump(out, f, indent=2)

# ------------------------------------------------------------------ console digest
print("MAIN attainment_table keys:", list(attain.keys()) if isinstance(attain, dict) else type(attain))
print("\nEXPANSION ATTAINMENT (alpha=0.02, 20 reseeds):")
for k, v in exp_attain.items():
    if v["vacuous"]:
        print(f"  {k:16s} VACUOUS-AT-TARGET (0% acceptance at alpha=0.02)")
        continue
    print(f"  {k:16s} acc={v['acc_fraction_mean']*100:5.1f}% [{v['acc_fraction_min']*100:.1f},{v['acc_fraction_max']*100:.1f}]  "
          f"accCER={v['acc_set_macro_cer_mean']*100:.2f}% [{v['acc_set_macro_cer_min']*100:.2f},{v['acc_set_macro_cer_max']*100:.2f}]  "
          f"viol={v['violations']}/20")
print("\nEXPANSION AUDIT (full-set macro/micro CER + excess-AURC):")
for k, v in aud.items():
    ex = ", ".join(f"{r['score']}:{r['excess_aurc']:.4f}" for r in v["results"])
    print(f"  {k:34s} n={v['n']:5d} macroCER={v['macro_cer']*100:6.2f}% microCER={v['micro_cer']*100:6.2f}%  exAURC[{ex}]")
print("\nHOLM realized m =", out["holm_realized"]["m"], " all reject:",
      all(r["reject_holm_global"] for r in out["holm_realized"]["rows"]),
      " p_holm =", out["holm_realized"]["rows"][0]["p_holm_global"])
print(f"\nLANDSCAPE (FREEZE-AMENDMENT-2026-07-13): "
      f"{out['landscape']['n_realized']} realized / {out['landscape']['n_pending']} pending / "
      f"{out['landscape']['n_withdrawn']} withdrawn (aidatatang SLR62, §D1) "
      f"attainment cells; mandarin_calsweep="
      f"{out['landscape']['mandarin_calsweep'].get('crossover_seed0', out['landscape']['mandarin_calsweep'].get('status'))}")
print("\nLANDSCAPE CERTIFIED FRONTIER (non-vacuous, 0/20 reseed violations; "
      "MagicData eval capped to ~4k per §2):")
for _k, _v in out["landscape"]["attainment"].items():
    if _v.get("status") != "realized" or _v.get("vacuous"):
        continue
    if _v["violations"] == 0 and _v["acc_fraction_mean"] > 0.005:
        cap = " [capped]" if _v.get("eval_cap_applied") else ""
        print(f"  {_v['backbone']:10s}/{_v['corpus']:9s} a={_v['alpha']:<5} "
              f"acc={_v['acc_fraction_mean']*100:5.1f}%  "
              f"accCER={_v['acc_set_macro_cer_mean']*100:5.2f}%  "
              f"viol=0/{_v['n_reseeds']}{cap}")
print("\nLANDSCAPE AUDIT (full-set macro/micro CER; MagicData capped):")
for _k, _v in out["landscape"]["audit"].items():
    if _v.get("status") != "realized":
        continue
    ex = ", ".join(f"{r['score']}:{r['excess_aurc']:.4f}" for r in _v["results"]) or "(no auditable score)"
    cap = " [capped]" if _v.get("eval_cap_applied") else ""
    print(f"  {_k:24s} n={_v['n']:6d} macroCER={_v['macro_cer']*100:6.2f}% "
          f"microCER={(_v['micro_cer'] or 0)*100:6.2f}% holm_m={_v['holm_family_size']}  exAURC[{ex}]{cap}")
print("\nwrote", os.path.join(RES, "numbers.json"))
