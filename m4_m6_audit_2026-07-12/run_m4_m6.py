#!/usr/bin/env python3
"""Red-team remediation computes M4 + M6 (asr paper).

M4 (comparator thinness): extend the audit family beyond "beats random
deferral" with the calibrated comparators the report demands -- s5
(temperature-scaled s1, fit on tune20 per design SS2.3's leakage rule) in
the primary run_audit family, and s6 (learned CER-regressor, fit on tune20)
as the separate exploratory table run_audit's contract requires.

M6 (permutation-null exchangeability unit): the paper speaker-blocks every
CI but ran the matched-abstention permutation at utterance level. Re-run
the null SPEAKER-BLOCKED (strata = speaker_id -- permutations exchange
deferral labels only within a speaker) alongside the original
utterance-level null; report BOTH.

Substrate: main_results_2026-07-09/test_scored.jsonl (the paper's audit
substrate; Paraformer degraded mode s3/s4=None), tune split
pilot_results_2026-07-09/tune20_scored.jsonl for s5/s6 fitting.
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

A = _portal_repo_root()
sys.path.insert(0, str(A))
sys.path.insert(0, str(_portal_commons_root()))
from asr_gate import audit as _audit  # noqa: E402
from asr_gate import io as _io        # noqa: E402
from asr_gate import scores as _scores  # noqa: E402

OUT = A / "m4_m6_audit_2026-07-12"
N_PERM = 2000
SEED = 0


def main():
    # plain JSONL read: io.load_utterances strips non-canonical columns
    # (cer/s1-s4), but the *_scored files carry them and run_audit needs them
    def _load(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    tune = _load(A / "pilot_results_2026-07-09/tune20_scored.jsonl")
    test = _load(A / "main_results_2026-07-09/test_scored.jsonl")
    tune = [u for u in tune if u.get("cer") is not None and u.get("s1") is not None]
    n0 = len(test)
    test = [u for u in test if u.get("cer") is not None and u.get("s1") is not None]
    if len(test) != n0:
        # mirrors run_audit's own <=1% degraded-tail exclusion convention
        print(f"excluded {n0 - len(test)}/{n0} test row(s) missing cer/s1 "
              f"({(n0 - len(test)) / n0:.2%} -- degraded tail)", flush=True)

    # ---- fit on tune ONLY (design SS2.3 leakage rule) ----
    s1_tune = np.array([u["s1"] for u in tune], dtype=float)
    cer_tune = np.array([u["cer"] for u in tune], dtype=float)
    T = _scores.fit_temperature(s1_tune, cer_tune)
    reg = _scores.fit_cer_regressor(tune, cer_tune)

    s1_test = np.array([u["s1"] for u in test], dtype=float)
    s5_test = _scores.apply_temperature(s1_test, T)
    s6_test = _scores.apply_cer_regressor(test, reg)
    for u, v5, v6 in zip(test, s5_test, s6_test):
        u["s5"] = float(v5)
        # apply_cer_regressor returns PREDICTED CER (higher = worse); the
        # audit family is confidence-oriented (matched abstention defers the
        # LOWEST scores), so s6 enters negated -- feeding it raw inverts the
        # deferral set (first run: p=1.0 artifact, 2026-07-12).
        u["s6"] = float(-v6)
    print(f"fit: T={T:.4f}; s5/s6 attached to {len(test)} test rows", flush=True)

    # ---- M4: primary audit family incl. s5; s6 as separate exploratory table
    primary = _audit.run_audit(test, score_names=["s1", "s2", "s5"],
                               n_perm=N_PERM, seed=SEED)
    s6_table = _audit.run_audit(test, score_names=["s6"], n_perm=N_PERM, seed=SEED)

    # ---- M6: both permutation nulls per score ----
    err = np.array([u["cer"] for u in test], dtype=float)
    spk = np.array([u["speaker_id"] for u in test])
    m6 = {}
    for sname in ("s1", "s2", "s5", "s6"):
        sc = np.array([u[sname] for u in test], dtype=float)
        utt_null = _audit._matched_permutation_p(err, sc, strata=None,
                                                 n_perm=N_PERM, seed=SEED)
        spk_null = _audit._matched_permutation_p(err, sc, strata=spk,
                                                 n_perm=N_PERM, seed=SEED)
        m6[sname] = {"utterance_level": utt_null, "speaker_blocked": spk_null}
        print(f"M6 {sname}: utt p={utt_null['p_value']:.5f} "
              f"spk-blocked p={spk_null['p_value']:.5f}", flush=True)

    payload = {
        "m4": {"temperature": T, "regressor_params": _io.to_jsonable(reg),
               "primary_audit_s1_s2_s5": _io.to_jsonable(primary),
               "exploratory_audit_s6": _io.to_jsonable(s6_table)},
        "m6_permutation_nulls": _io.to_jsonable(m6),
        "protocol": {"fit_split": "tune20 (leakage rule SS2.3)",
                     "n_perm": N_PERM, "seed": SEED,
                     "substrate": "main_results_2026-07-09/test_scored.jsonl"},
    }
    json.dump(payload, open(OUT / "M4-M6-RESULT.json", "w"), indent=1, default=str)
    print("M4_M6_DONE ->", OUT / "M4-M6-RESULT.json")


if __name__ == "__main__":
    main()
