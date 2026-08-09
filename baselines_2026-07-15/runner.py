#!/usr/bin/env python3
"""Named confidence-baseline comparison for the asr-gate landscape (WORKLOAD-BENCHMARK
-2026-07-15, asr row; REVISION-PLAN §P3 step 2).

Reviewers at the target venue expect a method-vs-method table naming standard ASR
confidence estimators. We evaluate the TeLeS-family named baselines THROUGH THE SAME
excess-AURC audit machinery (asr_gate.audit.run_audit) on the SAME cells our primary
scores s1/s2 use, so every number is directly comparable to the paper's audit rows.

Derivable from the cached decode artifacts (per-emitted-token log-probs, nbest[0]
['token_logps']):
  - class_prob_mean    = mean_t exp(logp_t)              (arithmetic mean token prob)
  - class_prob_min     = exp(min_t logp_t)               (== our s2, exactly; noted)
  - lengthnorm_pathprob= exp( mean_t logp_t )            (== exp(s1); monotone in s1,
                                                          so identical excess-AURC; noted)
  - cem                = learned confidence estimation module: logistic regression on
                         [mean_logp, min_logp, std_logp, log1p(n_tok), duration_s] fit on
                         the Aishell-1 DEV split (per backbone; refs => CER label
                         1[CER>0]); score = -P(error) (higher = more confident). Fit on
                         dev only (leakage-free, mirrors the gate's dev calibration),
                         applied to every test corpus.

NOT derivable (disclosed honestly): Tsallis entropy needs the full per-token vocabulary
distribution (token_full_posteriors), which is None in every dump (the artifacts carry
only the emitted-token logps, the s4 carrier is absent) -- so Tsallis entropy is reported
as UNAVAILABLE, not imputed.

MagicData uses the frozen FREEZE-AMENDMENT §2 speaker-disjoint ~4k eval cap (same subset
the certificate uses).

Run:
  PYTHONPATH=<reliability-commons> python3 runner.py
"""
import json, os, glob, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from asr_gate import audit as _audit
from asr_gate import cer as _cer
from asr_gate import scores as _scores
from asr_gate import io as _io

HERE = os.path.dirname(os.path.abspath(__file__))
PULL = os.path.join(HERE, "..", "landscape_pulled_2026-07-15")
BACKBONES = ["paraformer", "belle", "zipformer"]
CORPORA = ["aishell", "thchs30", "magicdata"]
CAP = 4000
BASELINE_SCORES = ["class_prob_mean", "class_prob_min", "lengthnorm_pathprob", "cem"]
AUDIT_SCORES = ["s1", "s2"] + BASELINE_SCORES  # our scores + named baselines, same audit


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def token_logps(row):
    nb = row.get("nbest")
    if not nb:
        return None
    return nb[0].get("token_logps")


def add_baseline_columns(rows, cem_model=None, cem_scaler=None):
    """Attach the token-logp baselines (and, if a fitted model is supplied, cem) to
    each row in-place; leave None where token_logps is unavailable (audit excludes it,
    exactly as it does for s1/s2)."""
    feats = []
    idx = []
    for i, r in enumerate(rows):
        tl = token_logps(r)
        if not tl:
            r["class_prob_mean"] = None
            r["class_prob_min"] = None
            r["lengthnorm_pathprob"] = None
            r["cem"] = None
            continue
        tl = np.asarray(tl, dtype=float)
        r["class_prob_mean"] = float(np.mean(np.exp(tl)))
        r["class_prob_min"] = float(np.exp(np.min(tl)))
        r["lengthnorm_pathprob"] = float(np.exp(np.mean(tl)))
        feats.append([float(np.mean(tl)), float(np.min(tl)), float(np.std(tl)),
                      float(np.log1p(len(tl))), float(r.get("duration_s") or 0.0)])
        idx.append(i)
        r["cem"] = None
    if cem_model is not None and feats:
        X = cem_scaler.transform(np.asarray(feats))
        p_err = cem_model.predict_proba(X)[:, 1]
        for j, i in enumerate(idx):
            rows[i]["cem"] = float(-p_err[j])  # higher = more confident
    return rows


def fit_cem(dev_rows):
    """Fit the CEM on a dev split (refs present). Returns (model, scaler) or (None,None)
    if the split is degenerate (all-clean or all-error)."""
    scored = _scores.score_table(dev_rows)
    scored = _cer.compute_cer_batch(scored, numeral_policy="keep")
    X = []
    y = []
    for r in scored:
        tl = token_logps(r)
        if not tl or r.get("cer") is None:
            continue
        tl = np.asarray(tl, dtype=float)
        X.append([float(np.mean(tl)), float(np.min(tl)), float(np.std(tl)),
                  float(np.log1p(len(tl))), float(r.get("duration_s") or 0.0)])
        y.append(1 if r["cer"] > 0 else 0)
    X = np.asarray(X)
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None, None, {"n": int(len(y)), "err_rate": float(np.mean(y)) if len(y) else None,
                            "status": "degenerate"}
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X), y)
    return model, scaler, {"n": int(len(y)), "err_rate": float(np.mean(y)), "status": "fit"}


def capped_utts(rows, cap=CAP):
    by_spk = collections.OrderedDict()
    for r in rows:
        by_spk.setdefault(str(r["speaker_id"]), []).append(r["utt_id"])
    out = set()
    cum = 0
    for s in sorted(by_spk):
        out.update(by_spk[s])
        cum += len(by_spk[s])
        if cum >= cap:
            break
    return out


def main():
    out = {"cells": {}, "note": {
        "tsallis_entropy": "UNAVAILABLE -- needs full per-token vocab distribution "
                           "(token_full_posteriors); None in every dump.",
        "class_prob_min": "== our s2 (exp(min token logp)) exactly.",
        "lengthnorm_pathprob": "== exp(s1); monotone in s1 => identical excess-AURC.",
        "magicdata": "eval capped to ~4k speaker-disjoint (FREEZE-AMENDMENT §2).",
        "audit": "excess-AURC via asr_gate.audit.run_audit, n_perm=2000, seed=0, alpha=0.05.",
    }, "cem_fit": {}}
    # fit one CEM per backbone on its Aishell-1 dev split
    cems = {}
    for bb in BACKBONES:
        dev = load_jsonl(os.path.join(PULL, f"backbone_{bb}", "dev_canonical.jsonl"))
        model, scaler, info = fit_cem(dev)
        cems[bb] = (model, scaler)
        out["cem_fit"][bb] = info

    for bb in BACKBONES:
        model, scaler = cems[bb]
        for corpus in CORPORA:
            rows = load_jsonl(os.path.join(PULL, f"backbone_{bb}", f"{corpus}_test_scored.jsonl"))
            if corpus == "magicdata":
                keep = capped_utts(rows)
                rows = [r for r in rows if r["utt_id"] in keep]
            add_baseline_columns(rows, model, scaler)
            res = _audit.run_audit(rows, score_names=AUDIT_SCORES, backbone_field=None,
                                   n_perm=2000, alpha=0.05, seed=0)
            payload = _io.to_jsonable(res)
            per_score = {r["score"]: {"excess_aurc": r["excess_aurc"],
                                      "aurc_method": r["aurc_method"],
                                      "aurc_random": r["aurc_random"],
                                      "p_value": r["p_value"], "n": r["n"],
                                      "n_excluded": r["n_excluded"]}
                         for r in payload["results"]}
            skipped = {s["score"]: s["skipped_reason"] for s in payload.get("skipped", [])}
            out["cells"][f"{bb}_{corpus}"] = {
                "backbone": bb, "corpus": corpus, "n": payload["n"],
                "macro_cer": payload["macro_cer"],
                "eval_cap_applied": corpus == "magicdata",
                "per_score": per_score, "skipped": skipped,
            }
            print(f"{bb}/{corpus}: n={payload['n']} macroCER={payload['macro_cer']*100:.2f}%")
            for s in AUDIT_SCORES:
                if s in per_score:
                    print(f"    {s:20s} exAURC={per_score[s]['excess_aurc']:+.4f} p={per_score[s]['p_value']:.4f}")
                else:
                    print(f"    {s:20s} SKIPPED ({skipped.get(s,'?')[:40]})")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
