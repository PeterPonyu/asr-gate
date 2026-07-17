"""The excess-AURC audit: is any field-standard confidence score better
than honest random deferral? (design §2.2 audit/, §3.4).

Zero new math beyond bookkeeping: every statistic is a direct call into
``relmetrics.aurc``, ``relmetrics.nulls``, ``relmetrics.multiplicity``, or
``relmetrics.bootstrap`` -- mirrors ``ope-audit``'s "thin wrapper" design
exactly (including its one documented MVP simplification: a single
median-split matched-abstention permutation test standing in for the
selective-ope research direction's full curve-sweeping null, which
``relmetrics.nulls`` does not implement).

Holm family size (critical correctness requirement #7)
----------------------------------------------------------
``m`` is ALWAYS ``len(roster)`` -- the actual (score, backbone) pairs
present with a fully-populated score column -- never hardcoded. The design
doc's own headline number (m = 10 = 5 scores x 2 backbones) is simply what
``m`` evaluates to when the roster matches that shape; running with a
different score/backbone roster changes ``m`` automatically.

Macro vs micro CER
--------------------
The AURC/excess-AURC statistics operate on PER-UTTERANCE CER (consistent
with the certified macro statistic in ``gate.py``/``ltt.py``). MICRO CER
(total edits / total ref chars) is reported separately, with a
SPEAKER-BLOCKED bootstrap CI (block = speaker_id, per design §3.1(b)), and
is never certified or audited via AURC -- see ``cer.py``'s
``micro_cer``/``macro_cer`` docstrings for the full rationale.

No-ground-truth honesty
-------------------------
``run_audit`` requires every instance to carry a computed ``cer`` (i.e.
``ref_text`` was present and ``cer.compute_cer_batch`` was already run);
it refuses outright (raises) rather than silently degrading, per design
§2.5 ("audit/calibrate refuse without refs" -- unlike ``ope-audit``, which
degrades to a disagreement-only mode, this tool's spec is a hard refusal).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from relmetrics import aurc as _aurc
from relmetrics import bootstrap as _bootstrap
from relmetrics import multiplicity as _multiplicity
from relmetrics import nulls as _nulls
from relmetrics import provenance as _provenance

from asr_gate import cer as _cer
from asr_gate import scores as _scores

__all__ = ["AuditError", "run_audit"]


class AuditError(ValueError):
    """Raised on any audit precondition violation (missing refs/scores)."""


def _matched_permutation_p(
    err: np.ndarray, score: np.ndarray, strata: Optional[np.ndarray], n_perm: int, seed: int
) -> Dict[str, Any]:
    """Matched-abstention permutation p-value for one score's median split.

    Convention: score HIGHER = more confident = accepted FIRST (matches
    ``relmetrics.aurc``'s convention directly, no sign flip needed for the
    AURC calls above); the score's "defer" (abstain) set is therefore the
    BOTTOM ``floor(n/2)`` by score. The null re-draws which utterances are
    deferred, matching the same per-stratum count, and compares selective
    risk (mean CER on the accepted set): a good score achieves LOWER
    selective risk than matched-random deferral, so ``p_value_less`` is the
    one-sided p-value of interest.
    """
    n = len(err)
    order = np.argsort(score, kind="stable")  # ascending: lowest-score first
    k = n // 2
    abstain_mask = np.zeros(n, dtype=bool)
    abstain_mask[order[:k]] = True
    result = _nulls.matched_abstention_null(
        losses=err, abstain_mask=abstain_mask, strata=strata, n_perm=n_perm, seed=seed
    )
    return {
        "p_value": result["p_value_less"],
        "abstention_fraction": float(k) / n,
        "n_perm": result["n_perm"],
    }


def _speaker_blocked_micro_cer_ci(
    utterances: List[Dict[str, Any]], n_boot: int = 1000, seed: int = 0
) -> Dict[str, Any]:
    edits = np.array([u["edits"] for u in utterances], dtype=float)
    ref_len = np.array([u["ref_len"] for u in utterances], dtype=float)
    speaker_id = np.array([u["speaker_id"] for u in utterances])

    def _micro_cer_stat(e: np.ndarray, r: np.ndarray) -> float:
        total_r = r.sum()
        return float(e.sum() / total_r) if total_r > 0 else float("nan")

    boot = _bootstrap.blocked_bootstrap(
        _micro_cer_stat, [edits, ref_len], block_ids=speaker_id, n_boot=n_boot, seeds=[seed]
    )
    return {
        "point": boot["point"],
        "ci": boot["ci"],
        "n_blocks": boot["n_blocks"],
        "n_boot": boot["n_boot"],
        "method": boot["method"],
    }


def run_audit(
    utterances: List[Dict[str, Any]],
    score_names: Optional[Sequence[str]] = None,
    backbone_field: Optional[str] = None,
    n_perm: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the excess-AURC / matched-abstention / Holm audit.

    Parameters
    ----------
    utterances:
        Utterance table, s1-s5 already computed (``scores.score_table`` +
        s5 applied) and ``cer`` already computed (``cer.compute_cer_batch``)
        for EVERY row -- refused (:class:`AuditError`) otherwise.
    score_names:
        Score columns to test; default: every column in
        ``scores.PRIMARY_SCORES`` (s1-s5). Per (score, backbone), rows with
        a missing (``None``/NaN) value are excluded-and-counted (mirrors
        ``gate.py``'s ``excluded_missing_s1`` / G1's
        ``n_dropped_missing_score`` convention) as long as <=1% of the
        group is missing; a family that is missing for EVERY row (e.g. s3/
        s4 in a degraded-mode-only backbone) or for >1% of rows is skipped
        entirely -- not silently, but recorded in ``skipped`` with an
        explicit ``skipped_reason``. s6 is never included here (it is
        exploratory/BH-only by design, §2.2 -- callers wanting s6
        comparisons should treat it as a separate, clearly-labeled table).
    backbone_field:
        Utterance-table column identifying the backbone (e.g.
        ``"backbone"``); if ``None`` or absent, every utterance is treated
        as one implicit backbone.
    n_perm, alpha:
        Passed to the matched-abstention permutation null / Holm.
    seed:
        Shared seed for the permutation null draws.

    Returns
    -------
    dict
        ``results`` (one row per (score, backbone) KEPT in the roster:
        ``score``, ``backbone``, ``n`` (post-exclusion count),
        ``n_excluded``, ``aurc_method``, ``aurc_random``, ``excess_aurc``,
        ``p_value``, ``abstention_fraction``, ``p_holm``, ``reject_holm``),
        ``skipped`` (one row per (score, backbone) DROPPED from the roster:
        ``score``, ``backbone``, ``n_total``, ``n_missing``,
        ``skipped_reason``), ``holm_family_size`` (== ``len(results)``,
        never hardcoded), ``micro_cer`` (speaker-blocked bootstrap CI,
        computed once over ALL utterances), ``macro_cer`` (point estimate,
        the certified statistic, for reference), and a provenance stamp.
    """
    if not utterances:
        raise AuditError("run_audit: utterances must be non-empty")
    missing_cer = [u["utt_id"] for u in utterances if "cer" not in u]
    if missing_cer:
        raise AuditError(
            f"run_audit: {len(missing_cer)} utterance(s) have no computed 'cer' "
            f"(e.g. {missing_cer[:3]}); audit refuses to run without refs "
            "(design §2.5)."
        )

    candidate_scores = list(score_names) if score_names is not None else list(_scores.PRIMARY_SCORES)

    if backbone_field:
        backbones = sorted({u.get(backbone_field, "default") for u in utterances})
    else:
        backbones = ["default"]

    def _is_missing(v: Any) -> bool:
        return v is None or (isinstance(v, float) and np.isnan(v))

    # Exclude-and-count convention (mirrors gate.py's excluded_missing_s1 /
    # G1's n_dropped_missing_score): per (score, backbone), rows with a
    # missing value are excluded loudly and counted, as long as the missing
    # fraction is <=1% (a degraded tail, not wholesale). A family missing
    # for EVERY row (e.g. s3/s4 on a backbone that never emits N-best /
    # full posteriors -- a permitted degraded mode) or missing for >1% of
    # rows (wholesale -- the column was effectively never computed) is
    # skipped entirely, recorded in ``skipped`` with an explicit reason
    # rather than silently vanishing from the roster.
    roster = []
    skipped: List[Dict[str, Any]] = []
    for bb in backbones:
        subset = (
            [u for u in utterances if u.get(backbone_field, "default") == bb]
            if backbone_field
            else utterances
        )
        for sname in candidate_scores:
            n_total = len(subset)
            missing = [u for u in subset if _is_missing(u.get(sname))]
            n_missing = len(missing)
            if n_missing == n_total:
                skipped.append(
                    {
                        "score": sname,
                        "backbone": bb,
                        "n_total": n_total,
                        "n_missing": n_missing,
                        "skipped_reason": (
                            f"all {n_total} row(s) missing '{sname}' for "
                            f"backbone={bb!r} -- permitted degraded mode "
                            "(score never computable for this backbone), "
                            "not a partial-audit bug"
                        ),
                    }
                )
                continue
            frac_missing = n_missing / n_total
            if frac_missing > 0.01:
                skipped.append(
                    {
                        "score": sname,
                        "backbone": bb,
                        "n_total": n_total,
                        "n_missing": n_missing,
                        "skipped_reason": (
                            f"{n_missing}/{n_total} ({frac_missing:.1%}) row(s) missing "
                            f"'{sname}' for backbone={bb!r} -- wholesale, not a degraded "
                            "tail (score was effectively never computed); family dropped"
                        ),
                    }
                )
                continue
            if n_missing:
                print(
                    f"run_audit: excluded {n_missing}/{n_total} row(s) with missing "
                    f"'{sname}' for backbone={bb!r} (degraded tail); recorded as "
                    "n_excluded in the audit result"
                )
            kept = [u for u in subset if not _is_missing(u.get(sname))]
            roster.append((sname, bb, kept, n_missing))

    results = []
    for sname, bb, subset, n_excluded in roster:
        err = np.array([u["cer"] for u in subset], dtype=float)
        score = np.array([u[sname] for u in subset], dtype=float)

        aurc_method = _aurc.aurc(err, score)
        aurc_random = _aurc.random_aurc(err)
        excess = _aurc.excess_aurc_gain(err, score)
        coverage, risk = _aurc.risk_coverage_curve(err, score)
        _, oracle_risk = _aurc.risk_coverage_curve(err, -err)

        perm = _matched_permutation_p(err, score, strata=None, n_perm=n_perm, seed=seed)

        results.append(
            {
                "score": sname,
                "backbone": bb,
                "n": int(len(subset)),
                "n_excluded": int(n_excluded),
                "aurc_method": aurc_method,
                "aurc_random": aurc_random,
                "excess_aurc": excess,
                "p_value": perm["p_value"],
                "abstention_fraction": perm["abstention_fraction"],
                "n_perm": perm["n_perm"],
                "curve_coverage": coverage,
                "curve_risk": risk,
                "curve_oracle_risk": oracle_risk,
            }
        )

    if results:
        pvals = [r["p_value"] for r in results]
        holm = _multiplicity.holm_bonferroni(pvals, alpha=alpha)
        for r, p_holm, rej in zip(results, holm["adjusted_p"], holm["reject"]):
            r["p_holm"] = float(p_holm)
            r["reject_holm"] = bool(rej)

    micro = _speaker_blocked_micro_cer_ci(utterances, seed=seed)
    macro = _cer.macro_cer(utterances)

    out: Dict[str, Any] = {
        "alpha": float(alpha),
        "n_perm": int(n_perm),
        "n": len(utterances),
        "holm_family_size": len(results),
        "results": results,
        "skipped": skipped,
        "micro_cer": micro,
        "macro_cer": macro,
        "clip_count": int(sum(1 for u in utterances if u.get("clipped"))),
    }
    return _provenance.stamp_result(out, script_path=__file__, seeds=[seed])
