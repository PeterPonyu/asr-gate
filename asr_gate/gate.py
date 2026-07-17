"""The conformal gate: G1 (LTT) + G2 (Mondrian split-conformal upper bound),
Mondrian strata, and the honest-uncertainty refusal rules (design §2.3-2.5).

Fit/calibrate separation (design §2.3, critical correctness requirement)
--------------------------------------------------------------------------
s5's temperature and G2's ``score -> CER`` residual map are FIT on a TUNE
pool and CONFORMALIZED/CERTIFIED on a CAL pool. :func:`calibrate_gate`
enforces this at the API boundary: pass ``tune_instances`` explicitly (a
speaker-disjoint pool) or omit it to let the tool auto-split
``cal_instances`` by SPEAKER (never by utterance -- §3.1(a)) into an
internal fit/conformalize pair. Passing the SAME table (by identity or by
speaker overlap) as both raises :class:`GateError` -- fitting and
calibrating on the same split silently invalidates the coverage guarantee,
so this is refused, not warned about.

G1 vs G2, precisely
--------------------
- G1 (primary, LTT): certifies, with probability >= 1 - delta over the
  calibration draw, that the ACCEPTED SET's macro-CER (mean of
  per-utterance CER) <= alpha. A (alpha, delta) PAC-style guarantee, not a
  per-utterance bound. See :mod:`asr_gate.ltt`.
- G2 (secondary, Mondrian conformal): a per-utterance, per-stratum upper
  bound on CER via split/Mondrian conformal on the residuals of a
  score -> CER map fit on tune. A MARGINAL, in-expectation guarantee
  (``1 - alpha`` coverage per stratum), weaker than G1 but interpretable
  per-utterance. Accept iff the upper bound <= alpha.

These are two DIFFERENT guarantee types (§2.3, T3's whole reason for
existing) -- ``calibrate_gate``'s ``guarantee`` argument picks exactly ONE
to drive the ACCEPT/DEFER decision that ``apply_gate`` reports; the other
is still computed and reported for transparency, never silently dropped,
but is not what "certified" refers to in the gate's output.

Honest-uncertainty rules (design §2.5)
----------------------------------------
- Any stratum with < ``min_stratum_n`` (default 200) calibration utterances
  is DEFER-ALWAYS at apply time, regardless of score.
- Any utterance whose 1-best hypothesis has > 20% non-CJK, non-digit
  characters is OOD-REFUSE (a validity statement, not an uncertainty
  estimate -- a distinct exit state from DEFER).
- ``apply_gate`` warns (never silently recalibrates) when the incoming
  batch's G1-score distribution KS-departs from the calibration
  fingerprint beyond a preregistered distance.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from relmetrics import conformal as _conformal
from relmetrics import provenance as _provenance

from asr_gate import cer as _cer
from asr_gate import ltt as _ltt
from asr_gate import scores as _scores

__all__ = [
    "GateError",
    "sha256_of_file",
    "split_by_speaker",
    "duration_stratum",
    "is_ood",
    "assign_stratum",
    "calibrate_gate",
    "apply_gate",
]

DEFAULT_MIN_STRATUM_N = 200
DEFAULT_OOD_NON_CJK_FRAC = 0.2
DEFAULT_KS_WARN_DISTANCE = 0.15


class GateError(ValueError):
    """Raised on any gate.py precondition violation (leakage, missing refs,
    unknown guarantee/strata) with a precise, actionable message."""


def sha256_of_file(path: Union[str, Path]) -> str:
    """SHA256 of a file's bytes, for provenance-stamping inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_by_speaker(
    instances: List[Dict[str, Any]], frac: float, seed: int = 0
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Speaker-DISJOINT split: ``frac`` of SPEAKERS (not utterances) go to
    pool A, the rest to pool B. Never splits a speaker's utterances across
    both pools (critical correctness requirement #3)."""
    if not 0.0 < frac < 1.0:
        raise GateError(f"frac must be in (0, 1), got {frac}")
    speakers = sorted({u["speaker_id"] for u in instances})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(speakers))
    n_a = max(1, int(round(frac * len(speakers))))
    n_a = min(n_a, len(speakers) - 1) if len(speakers) > 1 else n_a
    a_speakers = {speakers[i] for i in perm[:n_a]}
    pool_a = [u for u in instances if u["speaker_id"] in a_speakers]
    pool_b = [u for u in instances if u["speaker_id"] not in a_speakers]
    return pool_a, pool_b


def _check_speaker_disjoint(
    tune_instances: List[Dict[str, Any]], cal_instances: List[Dict[str, Any]]
) -> None:
    if tune_instances is cal_instances:
        raise GateError(
            "calibrate_gate: tune_instances and cal_instances are the same object -- "
            "fitting (s5 temperature, G2 residual map) and calibrating/certifying on "
            "the identical split invalidates the coverage guarantee (design §2.3). "
            "Pass a speaker-disjoint tune pool, or omit tune_instances to auto-split "
            "cal_instances by speaker."
        )
    tune_speakers = {u["speaker_id"] for u in tune_instances}
    cal_speakers = {u["speaker_id"] for u in cal_instances}
    overlap = tune_speakers & cal_speakers
    if overlap:
        raise GateError(
            "calibrate_gate: tune_instances and cal_instances share "
            f"{len(overlap)} speaker(s) ({sorted(overlap)[:5]}...) -- fit and "
            "calibrate splits must be speaker-disjoint (design §2.3 leakage rule, "
            "§3.1(a))."
        )


def _require_cer(instances: List[Dict[str, Any]], where: str) -> None:
    missing = [u["utt_id"] for u in instances if "cer" not in u]
    if missing:
        raise GateError(
            f"{where}: {len(missing)} utterance(s) have no computed 'cer' "
            f"(missing ref_text or not yet passed through cer.compute_cer_batch), "
            f"e.g. {missing[:3]}. calibrate/audit refuse to run without refs "
            "(design §2.5 no-reference-mode rule); use `asr-gate apply` for "
            "ref-free operation."
        )


def duration_stratum(duration_s: float, edges: Sequence[float]) -> int:
    """Tercile bucket index (0, 1, 2) for a duration given frozen edges
    (2 cut points, ascending)."""
    return int(np.searchsorted(edges, duration_s, side="right"))


_NON_CJK_DIGIT_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿぀-ヿ0-9]"
)


def is_ood(hyp_text: str, threshold: float = DEFAULT_OOD_NON_CJK_FRAC) -> bool:
    """OOD/non-Mandarin refusal check (design §2.5): True iff the fraction
    of non-CJK, non-digit characters in ``hyp_text`` exceeds ``threshold``.
    Empty hypotheses are NOT OOD (nothing to flag; a separate concern)."""
    text = hyp_text or ""
    if len(text) == 0:
        return False
    non_cjk = sum(1 for ch in text if not _NON_CJK_DIGIT_RE.match(ch) and not ch.isspace())
    denom = sum(1 for ch in text if not ch.isspace())
    if denom == 0:
        return False
    return (non_cjk / denom) > threshold


def assign_stratum(
    utterance: Dict[str, Any],
    duration_edges: Sequence[float],
    use_gender: bool,
    known_genders: Sequence[str],
) -> str:
    """Build the Mondrian stratum key for one utterance, matching the keys
    frozen in ``gate["strata"]["counts"]`` at calibration time."""
    d_bucket = duration_stratum(utterance["duration_s"], duration_edges)
    if not use_gender:
        return f"dur{d_bucket}"
    gender = utterance.get("gender")
    if gender not in known_genders:
        return f"dur{d_bucket}:gender_unseen"
    return f"dur{d_bucket}:{gender}"


def _fit_g2_map(scores_arr: np.ndarray, cer_arr: np.ndarray) -> Dict[str, float]:
    """Univariate OLS ``predicted_cer = slope * score + intercept``, fit on
    TUNE. A deliberate simplification of "an s -> CER map": a simple linear
    map rather than s6's multi-feature ridge regressor, keeping G2
    explainable and decoupled from s6's exploratory-only status."""
    X = np.vstack([scores_arr, np.ones_like(scores_arr)]).T
    coef, *_ = np.linalg.lstsq(X, cer_arr, rcond=None)
    return {"slope": float(coef[0]), "intercept": float(coef[1])}


def _apply_g2_map(scores_arr: np.ndarray, g2_map: Dict[str, float]) -> np.ndarray:
    return g2_map["slope"] * scores_arr + g2_map["intercept"]


def calibrate_gate(
    cal_instances: List[Dict[str, Any]],
    tune_instances: Optional[List[Dict[str, Any]]] = None,
    alpha: float = 0.02,
    delta: float = 0.1,
    g1_score: str = "s1",
    g2_score: Optional[str] = None,
    guarantee: str = "ltt",
    strata: Optional[Sequence[str]] = None,
    fit_frac: float = 0.5,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
    numeral_policy: str = "keep",
    n_grid: int = 200,
    min_accept_frac: float = 0.1,
    ltt_procedure: str = "bonferroni",
    ltt_p_value: str = "eb",
    seed: int = 0,
    input_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Calibrate the gate: fit s5/G2 on tune, certify G1 (LTT) + G2
    (Mondrian conformal) on cal. Returns the ``gate.json`` payload.

    Parameters
    ----------
    cal_instances:
        Utterance table (must carry ``ref_text``/``cer``, i.e. already
        passed through ``cer.compute_cer_batch``, and s1-s4 via
        ``scores.score_table``). If ``tune_instances`` is ``None``, this
        pool is auto-split by SPEAKER into an internal fit/conformalize
        pair (``fit_frac`` to fit, the rest to conformalize/certify).
    tune_instances:
        Optional explicit, speaker-disjoint fitting pool (e.g. the real
        pilot's separate 20-speaker "tune" carve). Raises :class:`GateError`
        if it shares any speaker with ``cal_instances`` or is the same
        object.
    alpha, delta:
        G1 target CER bound / failure probability (design §2.3).
    g1_score:
        Which score column drives the LTT accept/defer threshold.
    g2_score:
        Which score column drives the G2 residual map; defaults to
        ``g1_score``.
    guarantee:
        ``"ltt"`` (G1 drives ACCEPT/DEFER, default) or ``"mondrian-ub"``
        (G2 per-stratum upper bound drives it). Both are always computed.
    strata:
        Subset of ``{"duration_tercile", "gender"}``; default
        ``("duration_tercile",)``.
    fit_frac:
        Speaker fraction routed to the auto-split fit pool when
        ``tune_instances`` is not given.
    min_stratum_n:
        Strata with fewer cal utterances are DEFER-ALWAYS at apply time.
    ltt_procedure, ltt_p_value:
        Pass-through to :func:`asr_gate.ltt.ltt_certify`'s ``procedure`` /
        ``p_value`` -- see that function's docstring for the pilot-failure
        motivation and validity arguments. Defaults ``"bonferroni"`` /
        ``"eb"`` (the tighter, ordering-free combination; the original
        ``"fixed-sequence"`` / ``"hb"`` combination is still available).
    input_paths:
        Optional source file path(s), hashed into provenance via
        :func:`sha256_of_file`.

    Returns
    -------
    dict
        The full gate specification (JSON-serializable via
        ``io.to_jsonable``), including a provenance stamp.
    """
    if guarantee not in ("ltt", "mondrian-ub"):
        raise GateError(f"guarantee must be 'ltt' or 'mondrian-ub', got {guarantee!r}")
    strata = tuple(strata) if strata is not None else ("duration_tercile",)
    unknown_strata = set(strata) - {"duration_tercile", "gender"}
    if unknown_strata:
        raise GateError(f"unknown strata axis/axes {unknown_strata}; choose from "
                         "{'duration_tercile', 'gender'}")
    g2_score = g2_score or g1_score

    if tune_instances is None:
        fit_pool, conformalize_pool = split_by_speaker(cal_instances, fit_frac, seed=seed)
    else:
        _check_speaker_disjoint(tune_instances, cal_instances)
        fit_pool, conformalize_pool = tune_instances, cal_instances

    _require_cer(fit_pool, "calibrate_gate (fit/tune pool)")
    _require_cer(conformalize_pool, "calibrate_gate (conformalize/cal pool)")

    # Exclude-and-count utterances with missing s1 (degraded token-logp
    # extraction happens for a small tail of real decodes — verified 1/7,142
    # on the Aishell-1 dev pilot). Mirrors the G1 path's existing
    # n_dropped_missing_score convention: sparse missingness is excluded
    # LOUDLY and recorded in the gate record; WHOLESALE missingness (>1%)
    # still hard-refuses, because that means the input was never scored.
    def _split_missing_s1(pool: List[Dict[str, Any]], label: str):
        missing = [u for u in pool
                   if u.get("s1") is None or (isinstance(u.get("s1"), float) and np.isnan(u["s1"]))]
        if not missing:
            return pool, []
        frac = len(missing) / len(pool)
        if frac > 0.01:
            raise GateError(
                f"calibrate_gate: {len(missing)}/{len(pool)} {label} utterances have "
                "missing s1 — that is wholesale, not a degraded tail (run "
                "scores.score_table first)")
        print(f"calibrate_gate: excluded {len(missing)}/{len(pool)} {label} "
              "utterance(s) with missing s1 (degraded token-logp extraction); "
              "recorded in gate record")
        kept = [u for u in pool if u not in missing]
        return kept, sorted(u["utt_id"] for u in missing)

    fit_pool, fit_excluded_ids = _split_missing_s1(fit_pool, "fit/tune")
    conformalize_pool, cal_excluded_ids = _split_missing_s1(conformalize_pool, "conformalize/cal")

    # --- s5: fit temperature on tune, apply to cal. ---
    fit_s1 = np.array([u["s1"] for u in fit_pool], dtype=float)
    fit_cer = np.array([u["cer"] for u in fit_pool], dtype=float)
    if np.any(np.isnan(fit_s1)):
        raise GateError("calibrate_gate: s5 needs s1 on every fit/tune utterance "
                         "(run scores.score_table first)")
    temperature = _scores.fit_temperature(fit_s1, fit_cer)

    cal_s1 = np.array([u["s1"] for u in conformalize_pool], dtype=float)
    cal_s5 = _scores.apply_temperature(cal_s1, temperature)
    for u, v in zip(conformalize_pool, cal_s5):
        u["s5"] = float(v)

    # --- G2 map: fit on tune, conformalize residuals on cal. ---
    fit_g2_scores = np.array([u[g2_score] for u in fit_pool], dtype=float)
    g2_map = _fit_g2_map(fit_g2_scores, fit_cer)

    cal_g2_scores = np.array([u[g2_score] for u in conformalize_pool], dtype=float)
    cal_cer = np.array([u["cer"] for u in conformalize_pool], dtype=float)
    predicted_cer = _apply_g2_map(cal_g2_scores, g2_map)
    residuals = cal_cer - predicted_cer

    # --- Mondrian strata: edges frozen from the conformalize/cal pool. ---
    durations = np.array([u["duration_s"] for u in conformalize_pool], dtype=float)
    duration_edges = np.quantile(durations, [1 / 3, 2 / 3]).tolist()
    use_gender = "gender" in strata
    known_genders: List[str] = []
    if use_gender:
        known_genders = sorted(
            {u["gender"] for u in conformalize_pool if u.get("gender") is not None}
        )
    stratum_keys = np.array(
        [assign_stratum(u, duration_edges, use_gender, known_genders) for u in conformalize_pool]
    )

    counts: Dict[str, int] = {}
    for k in np.unique(stratum_keys):
        counts[str(k)] = int(np.sum(stratum_keys == k))
    defer_always = {k: (n < min_stratum_n) for k, n in counts.items()}

    mondrian = _conformal.MondrianConformal(alpha=alpha, rng=seed).fit(residuals, stratum_keys)
    thresholds = {str(k): v for k, v in mondrian.thresholds().items()}

    # --- G1: LTT certificate on the conformalize/cal pool. ---
    g1_scores_arr = np.array([u.get(g1_score) for u in conformalize_pool], dtype=object)
    valid_mask = np.array([v is not None for v in g1_scores_arr])
    n_dropped_g1 = int((~valid_mask).sum())
    ltt_losses = cal_cer[valid_mask]
    ltt_scores = g1_scores_arr[valid_mask].astype(float)
    ltt_result = _ltt.ltt_certify(
        ltt_losses, ltt_scores, alpha=alpha, delta=delta, n_grid=n_grid,
        min_accept_frac=min_accept_frac, procedure=ltt_procedure, p_value=ltt_p_value,
    )
    ltt_result["n_dropped_missing_score"] = n_dropped_g1

    # --- Domain fingerprint: cal quantiles of g1_score and duration. ---
    fp_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    fingerprint = {
        "g1_score_name": g1_score,
        "g1_score_quantiles": np.quantile(ltt_scores, fp_quantiles).tolist(),
        "duration_quantiles": np.quantile(durations, fp_quantiles).tolist(),
        "quantile_probs": fp_quantiles,
    }

    input_hashes = (
        {str(p): sha256_of_file(p) for p in input_paths} if input_paths else {}
    )

    result: Dict[str, Any] = {
        "alpha": float(alpha),
        "delta": float(delta),
        "guarantee": guarantee,
        "g1_score": g1_score,
        "g2_score": g2_score,
        "strata_axes": list(strata),
        "duration_edges": duration_edges,
        "use_gender_stratum": use_gender,
        "known_genders": known_genders,
        "min_stratum_n": int(min_stratum_n),
        "strata": {
            "counts": counts,
            "defer_always": defer_always,
            "thresholds": thresholds,
        },
        "temperature": temperature,
        "g2_map": g2_map,
        "g1": ltt_result,
        "n_fit": int(len(fit_pool)),
        "n_cal": int(len(conformalize_pool)),
        "excluded_missing_s1": {
            "fit": fit_excluded_ids,
            "cal": cal_excluded_ids,
            "n_fit_excluded": len(fit_excluded_ids),
            "n_cal_excluded": len(cal_excluded_ids),
        },
        "fingerprint": fingerprint,
        "normalizer_version": _cer.NORMALIZER_VERSION,
        "numeral_policy": numeral_policy,
        "ood_non_cjk_threshold": DEFAULT_OOD_NON_CJK_FRAC,
        "ks_warn_distance": DEFAULT_KS_WARN_DISTANCE,
        "input_sha256": input_hashes,
    }
    return _provenance.stamp_result(result, script_path=__file__, seeds=[seed])


def apply_gate(
    gate: Dict[str, Any], instances: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply a calibrated gate to new (ref-free) utterances.

    Every utterance gets exactly one of ``ACCEPT``, ``DEFER``,
    ``OOD-REFUSE``. Requires s1-s4 already computed
    (``scores.score_table``); s5 is derived here from the gate's frozen
    temperature.

    Returns
    -------
    dict
        ``decisions`` (list of per-utterance dicts), ``domain_fingerprint``
        (KS-distance warning, batch-level), echoed ``alpha``/``delta``/
        ``guarantee``, and a provenance stamp.
    """
    g1_score = gate["g1_score"]
    g2_score = gate["g2_score"]
    duration_edges = gate["duration_edges"]
    use_gender = gate["use_gender_stratum"]
    known_genders = gate["known_genders"]
    defer_always = gate["strata"]["defer_always"]
    thresholds = gate["strata"]["thresholds"]
    lambda_star = gate["g1"]["lambda_star"]
    alpha = gate["alpha"]
    guarantee = gate["guarantee"]

    decisions = []
    batch_scores = []
    for u in instances:
        if any(k not in u for k in ("s1", "s2", "s3", "s4")):
            # Keys legitimately holding None (degraded mode, e.g. no
            # token_logps) are fine -- only genuinely UNSCORED utterances
            # (scores.score_table never ran) are rejected here.
            raise GateError(
                f"apply_gate: utterance {u.get('utt_id')!r} has no s1-s4 scores "
                "(run scores.score_table before apply)"
            )
        s1 = u.get("s1")
        s5 = _scores.apply_temperature(np.array([s1]), gate["temperature"])[0] if s1 is not None else None

        if is_ood(u["hyp_text"], gate["ood_non_cjk_threshold"]):
            decisions.append(
                {"utt_id": u["utt_id"], "action": "OOD-REFUSE", "stratum": None,
                 "reason": "non-CJK hyp fraction exceeds threshold"}
            )
            continue

        stratum = assign_stratum(u, duration_edges, use_gender, known_genders)
        if defer_always.get(stratum, True):
            decisions.append(
                {"utt_id": u["utt_id"], "action": "DEFER", "stratum": stratum,
                 "reason": "stratum below min_stratum_n or unseen at calibration"}
            )
            continue

        g1_val = u.get(g1_score) if g1_score != "s5" else s5
        g2_val = u.get(g2_score) if g2_score != "s5" else s5
        batch_scores.append(g1_val)

        g2_bound = None
        if g2_val is not None and stratum in thresholds:
            predicted = gate["g2_map"]["slope"] * g2_val + gate["g2_map"]["intercept"]
            g2_bound = predicted + thresholds[stratum]

        if guarantee == "ltt":
            accept = (
                lambda_star is not None and g1_val is not None and g1_val >= lambda_star
            )
        else:  # mondrian-ub
            accept = g2_bound is not None and g2_bound <= alpha

        decisions.append(
            {
                "utt_id": u["utt_id"],
                "action": "ACCEPT" if accept else "DEFER",
                "stratum": stratum,
                "g1_score_value": g1_val,
                "g2_upper_bound": g2_bound,
                "reason": None,
            }
        )

    fp = gate["fingerprint"]
    ks_warning = None
    if batch_scores:
        # Compare the batch's G1-score distribution against the calibration
        # fingerprint's quantiles via a KS-style 2-sample distance approx
        # (quantile-grid ECDF comparison; avoids needing raw cal scores at
        # apply time -- only the frozen fingerprint quantiles).
        ref_q = np.array(fp["g1_score_quantiles"])
        probs = np.array(fp["quantile_probs"])
        batch_arr = np.sort(np.asarray(batch_scores, dtype=float))
        batch_ecdf_at_ref = np.searchsorted(batch_arr, ref_q, side="right") / len(batch_arr)
        ks_distance = float(np.max(np.abs(batch_ecdf_at_ref - probs)))
        ks_warning = {
            "ks_distance": ks_distance,
            "threshold": gate["ks_warn_distance"],
            "warn": ks_distance > gate["ks_warn_distance"],
        }

    result: Dict[str, Any] = {
        "alpha": alpha,
        "delta": gate["delta"],
        "guarantee": guarantee,
        "decisions": decisions,
        "n": len(instances),
        "n_accept": sum(1 for d in decisions if d["action"] == "ACCEPT"),
        "n_defer": sum(1 for d in decisions if d["action"] == "DEFER"),
        "n_ood_refuse": sum(1 for d in decisions if d["action"] == "OOD-REFUSE"),
        "domain_fingerprint_check": ks_warning,
    }
    return _provenance.stamp_result(result, script_path=__file__, seeds=None)
