"""Confidence / nonconformity scores s1-s6 (design §2.2).

s1-s5 are the PRIMARY score family (model-free given decode artifacts);
s6 is exploratory. All are computed FROM the canonical utterance table
(``io.py``) -- no model access is needed at score time.

Degraded mode (design §2.2): a backbone exposing only 1-best + token
posteriors gets s1/s2/s4 and a degraded-mode banner for s3 (N-best margin
unavailable -- reported ``None``, never imputed). s4 additionally needs the
optional ``token_full_posteriors`` extension field (a full per-token
vocabulary distribution, NOT part of the design doc's minimal nbest shape
but declared in ``io.py`` as the carrier s4 needs); it is ``None`` whenever
that field is absent, which will be the common case for most real decode
artifacts -- documented explicitly rather than silently omitted.

s5/s6 fit/calibrate separation
-------------------------------
s5 (temperature-scaled s1) and s6 (the learned CER-regressor) are FIT on
the tune split only (never cal -- design §2.3's leakage rule); this module
exposes ``fit_temperature``/``apply_temperature`` and
``fit_cer_regressor``/``apply_cer_regressor`` as separate fit/apply steps
so callers (``gate.py``) can enforce that separation explicitly. Neither
function is called implicitly from ``score_table`` -- s5/s6 are computed
only when a caller supplies already-fit parameters.

s5, concretely
---------------
"Temperature scaling ... fit on tune split (NLL objective)" (design §2.2)
is underspecified at the level of exact mechanics (no raw logits are
available in the canonical schema, only log-probabilities). This module's
concrete instantiation: fit a single scalar ``T > 0`` minimizing the
negative log-likelihood of a binary "clean" label
(``y_i = 1`` iff ``CER_i == 0``, computed on the TUNE split, which has
refs) under the model ``P(y_i = 1) = sigmoid(s1_i / T)``; ``s5 = s1 / T``.
This is the standard single-parameter temperature-scaling recipe
(Guo et al. 2017) applied to the closest available proxy for a raw logit
(s1, already a log-probability). Documented as a deliberate scope choice.

s6, concretely
---------------
The design's "gradient-boosted [CER regressor] on s1-s5 + duration + hyp
length" needs a GBRT implementation (e.g. scikit-learn), which is outside
this tool's pinned minimal dependency set (numpy + pandas + relmetrics
only, per the target layout). s6 is therefore a genuine STUB per the
target layout's own wording: ridge-regularized linear least squares
(closed-form, dependency-free) on the same feature set, predicting CER
directly. Flagged exploratory (BH family, never Holm) exactly as the
design intends; upgrading to a real GBRT is future work, noted in README.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "PRIMARY_SCORES",
    "compute_s1",
    "compute_s2",
    "compute_s3",
    "compute_s4",
    "score_utterance",
    "score_table",
    "fit_temperature",
    "apply_temperature",
    "S6_FEATURES",
    "fit_cer_regressor",
    "apply_cer_regressor",
]

PRIMARY_SCORES = ("s1", "s2", "s3", "s4", "s5")


def compute_s1(nbest: List[Dict[str, Any]]) -> Optional[float]:
    """s1: length-normalized log-posterior -- mean over tokens of log p(token)
    for the top-1 hypothesis. ``None`` if ``token_logps`` is unavailable."""
    token_logps = nbest[0].get("token_logps")
    if not token_logps:
        return None
    return float(np.mean(token_logps))


def compute_s2(nbest: List[Dict[str, Any]]) -> Optional[float]:
    """s2: weakest link -- min over tokens of p(token) for the top-1
    hypothesis (``exp`` of the min log-prob, since exp is monotone).
    ``None`` if ``token_logps`` is unavailable."""
    token_logps = nbest[0].get("token_logps")
    if not token_logps:
        return None
    return float(np.exp(np.min(token_logps)))


def compute_s3(nbest: List[Dict[str, Any]]) -> Optional[float]:
    """s3: N-best margin -- ``(logp(hyp1) - logp(hyp2)) / len(hyp1)``, chars
    of the top-1 hypothesis text used as the length-normalization proxy
    (Mandarin is effectively character-tokenized). ``None`` (degraded mode)
    if fewer than 2 hypotheses or either sequence-level ``logp`` is missing.
    """
    if len(nbest) < 2:
        return None
    logp1, logp2 = nbest[0].get("logp"), nbest[1].get("logp")
    if logp1 is None or logp2 is None:
        return None
    hyp1_len = len(nbest[0]["text"])
    if hyp1_len == 0:
        return None
    return float((logp1 - logp2) / hyp1_len)


def compute_s4(nbest: List[Dict[str, Any]]) -> Optional[float]:
    """s4: negative mean token entropy rate -- ``-mean_t H(posterior_t)``
    for the top-1 hypothesis, requiring the FULL per-token vocabulary
    distribution (``token_full_posteriors``, an extension field -- see
    module docstring). ``None`` whenever it is absent (the common case),
    never imputed from ``token_logps`` alone (a single chosen-token
    log-prob cannot recover the full-distribution entropy)."""
    full = nbest[0].get("token_full_posteriors")
    if not full:
        return None
    entropies = []
    for token_dist in full:
        p = np.asarray(token_dist, dtype=float)
        p = p[p > 0]
        if p.size == 0:
            entropies.append(0.0)
            continue
        entropies.append(float(-np.sum(p * np.log(p))))
    if not entropies:
        return None
    return float(-np.mean(entropies))


def score_utterance(utterance: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute s1-s4 (primary, model-free scores) for one canonical
    utterance record. s5/s6 are NOT computed here (see module docstring)."""
    nbest = utterance["nbest"]
    return {
        "s1": compute_s1(nbest),
        "s2": compute_s2(nbest),
        "s3": compute_s3(nbest),
        "s4": compute_s4(nbest),
    }


def score_table(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Augment every utterance in the table with s1-s4."""
    out = []
    for u in utterances:
        u2 = dict(u)
        u2.update(score_utterance(u))
        out.append(u2)
    return out


# ---------------------------------------------------------------------------
# s5: temperature scaling. fit_temperature MUST be called on a tune split
# only; apply_temperature is then used on cal/test s1 values.
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _nll(t: float, s1: np.ndarray, y: np.ndarray) -> float:
    if t <= 0:
        return float("inf")
    p = _sigmoid(s1 / t)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(s1: np.ndarray, cer: np.ndarray) -> float:
    """Fit the scalar temperature ``T`` on a TUNE split (needs refs, hence
    ``cer``). ``y_i = 1{cer_i == 0}``; ``T`` minimizes the NLL of
    ``sigmoid(s1_i / T)`` against ``y_i``. Raises if any ``s1`` is missing
    (``None``/NaN) -- s5 requires s1 to be available.

    Returns
    -------
    float
        The fit temperature, always > 0.
    """
    s1 = np.asarray(s1, dtype=float)
    cer = np.asarray(cer, dtype=float)
    if s1.shape != cer.shape or s1.size == 0:
        raise ValueError("s1 and cer must be equal-length, non-empty arrays")
    if np.any(np.isnan(s1)):
        raise ValueError("fit_temperature: s1 contains missing values (None/NaN)")
    y = (cer == 0.0).astype(float)
    result = minimize_scalar(
        _nll, args=(s1, y), bounds=(1e-3, 1e4), method="bounded"
    )
    return float(result.x)


def apply_temperature(s1: np.ndarray, temperature: float) -> np.ndarray:
    """s5 = s1 / T, applied to (already TUNE-fit) ``temperature``."""
    s1 = np.asarray(s1, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return s1 / temperature


# ---------------------------------------------------------------------------
# s6: exploratory CER-regressor stub (ridge closed-form; see module
# docstring for why this is not a GBRT). fit_cer_regressor MUST be called
# on a tune split only.
# ---------------------------------------------------------------------------

S6_FEATURES = ("s1", "s2", "s3", "s4", "s5", "duration_s", "hyp_len")


def _feature_matrix(utterances: List[Dict[str, Any]]) -> np.ndarray:
    rows = []
    for u in utterances:
        row = []
        for f in S6_FEATURES:
            if f == "hyp_len":
                row.append(float(len(u.get("hyp_text", ""))))
            else:
                v = u.get(f)
                row.append(float(v) if v is not None else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def fit_cer_regressor(
    utterances: List[Dict[str, Any]], cer: np.ndarray, ridge_lambda: float = 1.0
) -> Dict[str, Any]:
    """Fit s6's ridge-regularized linear CER regressor on a TUNE split.

    Missing features (``None`` scores, e.g. s3/s4 in degraded mode) are
    imputed to 0.0 after column-standardization is undone by design (ridge
    on raw features; 0-imputation is a documented simplification -- s6 is
    explicitly exploratory, never promoted to a Holm claim).

    Returns
    -------
    dict
        ``{"coef": ndarray (len(S6_FEATURES),), "intercept": float,
        "feature_means": ndarray, "feature_scales": ndarray}`` -- standardize
        -then-fit parameters needed by :func:`apply_cer_regressor`.
    """
    X = _feature_matrix(utterances)
    y = np.asarray(cer, dtype=float)
    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("utterances and cer must have the same non-zero length")

    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0
    Xs = (X - means) / scales

    n, p = Xs.shape
    Xd = np.hstack([Xs, np.ones((n, 1))])
    ridge = ridge_lambda * np.eye(p + 1)
    ridge[-1, -1] = 0.0  # do not regularize the intercept
    coef_full = np.linalg.solve(Xd.T @ Xd + ridge, Xd.T @ y)

    return {
        "coef": coef_full[:-1].tolist(),
        "intercept": float(coef_full[-1]),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
    }


def apply_cer_regressor(
    utterances: List[Dict[str, Any]], params: Dict[str, Any]
) -> np.ndarray:
    """Apply an (already TUNE-fit) s6 regressor to score new utterances."""
    X = _feature_matrix(utterances)
    means = np.asarray(params["feature_means"], dtype=float)
    scales = np.asarray(params["feature_scales"], dtype=float)
    coef = np.asarray(params["coef"], dtype=float)
    Xs = (X - means) / scales
    return Xs @ coef + params["intercept"]
