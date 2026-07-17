"""Learn-then-Test (LTT) selective-risk certificate.

NOT in ``relmetrics`` yet -- this is the single new-math module the design
doc calls for (§2.2: "the single new-math addition ... upstreamed to
relmetrics, mirrors ope-audit's CRC upstreaming"). Implements exactly the
critic-corrected statistic from design §2.3, not the naive one.

The problem
-----------
Selective risk ``R(lambda) = E[loss * 1{s >= lambda}] / E[1{s >= lambda}]``
is a RATIO of two means and is NOT monotone in ``lambda`` in general, so
plain split-conformal / CRC (which need a monotone loss) does not directly
certify "accept iff s >= lambda*". Learn-then-Test (Angelopoulos, Bates,
Candes, Jordan & Lei, arXiv:2110.01052) certifies it instead via per-lambda
hypothesis tests plus fixed-sequence testing, at the cost of a (alpha,
delta) PAC-style guarantee (not a marginal expectation).

The corrected statistic (design §2.3, post-critic-review)
-----------------------------------------------------------
The per-lambda null is ``H0(lambda): R(lambda) > alpha``. Rather than
computing a Hoeffding-Bentkus (HB) p-value on the ratio ``R(lambda)``
directly (invalid -- HB needs a BOUNDED-MEAN statistic, and a ratio of
random means is not one), the null is tested via the algebraically
equivalent bounded-mean reformulation

    H0(lambda): E[(loss - alpha) * 1{s >= lambda}] > 0

which coincides with ``R(lambda) > alpha`` whenever the acceptance
probability ``E[1{s >= lambda}]`` is positive (dividing a mean of the same
sign by a positive quantity doesn't flip its sign). The statistic
``X_i(lambda) = (loss_i - alpha) * 1{s_i >= lambda}`` is bounded in
``[-alpha, 1-alpha]`` for every ``i`` (loss in ``[0, 1]``, alpha in
``(0, 1)``) REGARDLESS of how many points are accepted at that lambda --
this fixed bound (not a per-lambda-varying one) is exactly what makes HB
well-defined here. We shift it to ``Y_i(lambda) = X_i(lambda) + alpha in
[0, 1]`` and apply the standard HB p-value (Bates et al. 2021, "Distribution
-Free, Risk-Controlling Prediction Sets", eq. 3) for testing whether a
[0, 1]-bounded mean exceeds a threshold, with threshold = alpha, using ALL
n calibration points at every lambda (points below lambda contribute
exactly ``alpha`` to ``Y``, not zero -- this keeps the sample size, and
hence the HB bound, fixed across the whole lambda grid).

Fixed-sequence testing (``procedure="fixed-sequence"``)
----------------------------------------------------------
Lambdas are tested in DECREASING order of conservatism, i.e. from the
HIGHEST lambda (smallest accepted set, safest) downward. We reject
``H0(lambda)`` (i.e. certify that lambda) at level delta as long as
``p(lambda) <= delta``; the first lambda where we FAIL to reject stops
the procedure (no smaller/more-permissive lambda is tested or certified).
This monotone stopping rule is what lets every individual test run at
level delta with no Bonferroni correction across the grid (Angelopoulos et
al. 2021, Lemma 1) -- family-wise validity of the WHOLE procedure comes
from the stopping rule, not from correcting each test.
``lambda* = the most permissive (smallest) lambda that was certified``.

PILOT FAILURE (real Aishell cal, n=3567, alpha=0.02, delta=0.1): this mode
certified NOTHING. The most-conservative grid point (accepted_fraction
0.100, empirical accepted-region risk 0.0095 -- well under alpha) got
p=0.90 under Hoeffding-Bentkus, because ``Y_i(lambda) = alpha +
(loss_i-alpha)*1{s_i>=lambda}`` dilutes the departure-from-alpha signal by
the acceptance fraction itself while HB's concentration uses the full
``[0,1]`` range regardless of how small the per-point variance actually
is; fixed-sequence then halted at that single non-rejection and never
tried a more permissive lambda. Two independent fixes address this (and
compose): a tighter, variance-adaptive p-value (below), and a selection
rule that does not die at one weak-power grid point (below).

Empirical-Bernstein p-value (``p_value="eb"``, default)
----------------------------------------------------------
:func:`hb_pvalue` uses only the FIXED ``[0, 1]`` range of ``Y``, ignoring
that ``Y``'s actual per-point variance on this data is tiny (CER is
concentrated near 0 for the vast majority of utterances). An
empirical-Bernstein (EB) bound (Maurer & Pontil 2009, Theorem 4;
Audibert-Munos-Szepesvari 2007) instead adapts to the SAMPLE variance
``V_hat`` (``ddof=1``), giving a dramatically tighter deviation bound
whenever ``V_hat`` is small:

    eps(delta) = sqrt(2 * V_hat * ln(2/delta) / n) + 7*ln(2/delta) / (3*(n-1))

with ``P(E[Y] > Ybar + eps(delta)) <= delta`` for every ``delta in (0, 1)``,
for ANY distribution on ``[0, 1]`` (not just under H0) -- this is the
defining property of ``Ybar + eps(delta)`` as a ``(1-delta)`` upper
confidence bound (UCB) on ``E[Y]``.

:func:`eb_pvalue` inverts this UCB family into a p-value for
``H0: E[Y] >= alpha`` via "UCB inversion" (the same general recipe that
underlies Bates et al. 2021's own p-value constructions): define
``p = inf{delta in (0, 1] : Ybar + eps(delta) <= alpha}`` (1.0 if no such
delta exists, e.g. because ``Ybar >= alpha`` already). ``eps`` is strictly
increasing in ``ln(1/delta)`` hence strictly decreasing in ``delta``, so
this infimum is attained at equality and is found in closed form by
solving the quadratic ``eps(delta) = alpha - Ybar`` for
``sqrt(ln(2/delta))``.

Why this is a valid p-value (``P(p <= u | H0) <= u`` for every ``u``): by
construction, ``p <= u`` holds iff ``Ybar + eps(u) <= alpha`` (monotonicity
of ``eps`` in ``delta``). Under ``H0`` (``E[Y] >= alpha``), the event
``{Ybar + eps(u) <= alpha}`` implies ``{Ybar + eps(u) <= E[Y]}``, i.e.
``{E[Y] > Ybar + eps(u)}`` up to a measure-zero boundary (``Y`` is
continuous here since ``V_hat`` is continuous a.s.). The EB UCB property
gives ``P(E[Y] > Ybar + eps(u)) <= u`` unconditionally, so
``P(p <= u | H0) <= P(E[Y] > Ybar + eps(u) | H0) <= u``. This argument uses
only ``E[Y] >= alpha`` (not equality), so it holds uniformly across the
whole (composite) null.

Bonferroni-over-grid selection (``procedure="bonferroni"``, default)
-------------------------------------------------------------------------
Instead of a single fixed-sequence walk that halts at the first
non-rejection, EVERY lambda in the grid (``K`` candidates) is tested at
level ``delta / K``. ``lambda* = the REJECTED lambda with the LARGEST
accepted_fraction`` (``None``, VACUOUS, if no lambda is rejected at that
level). Family-wise validity: a false certification occurs only if some
lambda with true ``R(lambda) > alpha`` is rejected; by a union bound over
the ``K`` tests each valid at level ``delta/K``,
``P(exists a falsely-rejected lambda in the grid) <= K * (delta/K) =
delta``, and restricting the selection rule to only ever pick among
rejected lambdas keeps that same bound for whichever one is chosen. This
is standard, ordering-free, ``FWER <= delta`` LTT (Angelopoulos et al.
2021, the Bonferroni instantiation contrasted with their fixed-sequence
one) -- it cannot be sunk by one low-power grid point the way
fixed-sequence can, at the cost of the ``ln(K)``-ish power loss from
correcting every test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import binom

__all__ = ["hb_pvalue", "eb_pvalue", "build_lambda_grid", "ltt_certify"]


def _kl_bernoulli(a: float, b: float) -> float:
    """Binary KL divergence ``h1(a, b) = a*log(a/b) + (1-a)*log((1-a)/(1-b))``
    with the convention ``0*log(0) = 0``. ``a, b`` must be in ``[0, 1]``."""
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 1e-12, 1.0 - 1e-12))
    term1 = 0.0 if a <= 0.0 else a * np.log(a / b)
    term2 = 0.0 if a >= 1.0 else (1.0 - a) * np.log((1.0 - a) / (1.0 - b))
    return float(term1 + term2)


def hb_pvalue(y: np.ndarray, alpha: float) -> float:
    """Hoeffding-Bentkus p-value testing ``H0: E[Y] >= alpha`` for a
    ``[0, 1]``-bounded random variable ``Y``, ``n = len(y)`` i.i.d. draws.

    ``p_HB = min(1, exp(-n * h1(min(ybar, alpha), alpha)),
    e * P(Binomial(n, alpha) <= ceil(n * ybar)))`` (Bates et al. 2021, eq. 3;
    the min of the Hoeffding and Bentkus bounds, each separately a valid
    super-uniform p-value under H0, and their combination proven valid in
    the same reference). Returns 1.0 when ``ybar >= alpha`` (no evidence
    against H0) automatically, via the KL term vanishing at ``a == b``.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if np.any((y < -1e-9) | (y > 1.0 + 1e-9)):
        raise ValueError("y must be bounded in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = y.size
    ybar = float(np.mean(y))

    h1 = _kl_bernoulli(min(ybar, alpha), alpha)
    p_hoeffding = float(np.exp(-n * h1))

    k = int(np.ceil(n * ybar))
    p_bentkus = float(np.e * binom.cdf(k, n, alpha))

    return float(min(1.0, p_hoeffding, p_bentkus))


def eb_pvalue(y: np.ndarray, alpha: float) -> float:
    """Empirical-Bernstein p-value testing ``H0: E[Y] >= alpha`` for a
    ``[0, 1]``-bounded random variable ``Y``, ``n = len(y)`` i.i.d. draws.

    Construction (module docstring has the full derivation and validity
    proof): invert the Maurer & Pontil (2009) empirical-Bernstein upper
    confidence bound ``UCB(delta) = ybar + sqrt(2*v_hat*ln(2/delta)/n) +
    7*ln(2/delta)/(3*(n-1))`` (``v_hat`` the sample variance, ``ddof=1``) to
    ``p = inf{delta in (0, 1] : UCB(delta) <= alpha}`` (``1.0`` if
    ``ybar >= alpha``, since then no ``delta`` drives ``UCB`` below
    ``alpha``). ``UCB`` is monotone decreasing in ``delta``, so this
    infimum is found in closed form by solving the quadratic in
    ``sqrt(ln(2/delta))``. This is a valid p-value, ``P(p <= u | H0) <= u``
    for every ``u``, because ``p <= u`` implies ``UCB(u) <= alpha <= E[Y]``
    (under H0), i.e. ``E[Y] > ybar + eps(u)``, an event the EB bound caps
    at probability ``u`` unconditionally.

    Unlike :func:`hb_pvalue` (which only uses the fixed ``[0, 1]`` range),
    this adapts to the SAMPLE VARIANCE of ``y`` -- much tighter whenever
    that variance is small, as it is for per-utterance CER on the pilot
    data. Requires ``n >= 2`` to estimate a variance; returns ``1.0`` for
    ``n == 1`` (no evidence available).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if np.any((y < -1e-9) | (y > 1.0 + 1e-9)):
        raise ValueError("y must be bounded in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = y.size
    ybar = float(np.mean(y))

    target = alpha - ybar
    if target <= 0.0 or n < 2:
        return 1.0

    v_hat = float(np.var(y, ddof=1))
    a = 7.0 / (3.0 * (n - 1))
    b = float(np.sqrt(2.0 * v_hat / n))
    # Solve a*t^2 + b*t - target == 0 for t = sqrt(ln(2/delta)) >= 0, the
    # unique non-negative root (a > 0, target > 0 guarantees a real,
    # positive root via the quadratic formula's '+' branch).
    disc = b * b + 4.0 * a * target
    t = (-b + np.sqrt(disc)) / (2.0 * a)
    x_star = t * t  # = ln(2/delta) at the crossing point
    p = 2.0 * np.exp(-x_star)
    return float(min(1.0, max(0.0, p)))


def build_lambda_grid(
    scores: np.ndarray, n_grid: int = 200, min_accept_frac: float = 0.1
) -> np.ndarray:
    """Default candidate-threshold grid: quantiles of ``scores``, ascending,
    deduplicated. Callers pass ``lambda_grid`` explicitly to override.

    The grid's TOP (most conservative, tested FIRST under fixed-sequence
    testing) is capped at the ``1 - min_accept_frac`` quantile rather than
    the sample max. This is a grid-CONSTRUCTION choice, not a change to the
    testing procedure's validity, but it matters a great deal for its
    POWER: because :func:`hb_pvalue` is evaluated on
    ``Y_i(lambda) = alpha + (loss_i - alpha) * 1{s_i >= lambda}`` over ALL
    ``n`` calibration points (not just the accepted ones -- that's what
    keeps the statistic's bound fixed across lambda, per the module
    docstring), the signal from the accepted subset is diluted by the
    ACCEPTED FRACTION itself: ``ybar - alpha`` scales roughly with
    ``accepted_fraction`` even when the accepted subset's true risk is far
    below ``alpha``. A top-of-grid lambda that accepts only a handful of
    points is therefore essentially UNFALSIFIABLE (``ybar`` stays
    numerically indistinguishable from ``alpha`` regardless of how good
    those few points are) -- and because fixed-sequence testing HALTS at
    the first non-rejection, wasting that first test on a hopeless,
    near-empty acceptance set would make the whole procedure vacuous even
    when strongly informative lambdas exist further down the grid. The
    default floor, 10%, is deliberately pinned to the same threshold the
    design doc's own K4 vacuity check uses ("no score family accepts >= 10%
    of test utterances" is reported VACUOUS-AT-TARGET) -- below that
    floor, the design already treats the result as a non-finding, so
    excluding it from the search costs nothing the design doc considers
    a real answer.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 <= min_accept_frac < 1.0:
        raise ValueError("min_accept_frac must be in [0, 1)")
    qs = np.linspace(0.0, 1.0 - min_accept_frac, n_grid)
    grid = np.unique(np.quantile(scores, qs))
    return grid


_P_VALUE_FUNCS = {"hb": hb_pvalue, "eb": eb_pvalue}


def ltt_certify(
    losses: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    lambda_grid: Optional[Sequence[float]] = None,
    n_grid: int = 200,
    min_accept_frac: float = 0.1,
    procedure: str = "bonferroni",
    p_value: str = "eb",
) -> Dict[str, Any]:
    """Learn-then-Test selective-risk certificate (design §2.3, G1).

    Parameters
    ----------
    losses:
        Per-calibration-utterance loss (CER), shape ``(n,)``, in ``[0, 1]``.
    scores:
        Per-calibration-utterance confidence score, shape ``(n,)``; HIGHER
        means accepted first (``accept iff score >= lambda``).
    alpha:
        Target certified risk bound (e.g. 0.02 for 2% CER).
    delta:
        Failure probability: with probability >= ``1 - delta`` over the
        calibration draw, the accepted set's macro-CER <= ``alpha``.
    lambda_grid:
        Explicit candidate thresholds; default: quantiles of ``scores``
        (see :func:`build_lambda_grid`).
    n_grid:
        Grid size when ``lambda_grid`` is not given.
    min_accept_frac:
        Passed to :func:`build_lambda_grid` when ``lambda_grid`` is not
        given -- caps the most-conservative candidate lambda's acceptance
        rate from below (avoids a hopeless, zero-power first test).
    procedure:
        ``"bonferroni"`` (default): test every grid lambda at level
        ``delta / K`` and select the REJECTED lambda with the largest
        accepted_fraction -- ordering-free, cannot die at one weak-power
        grid point (see module docstring). ``"fixed-sequence"``: the
        original walk from most- to least-conservative lambda, stopping at
        the first non-rejection (kept for backward compatibility / the
        pilot-failure regression test).
    p_value:
        ``"eb"`` (default): empirical-Bernstein, adapts to the sample
        variance of the per-lambda statistic -- much tighter than ``"hb"``
        when that variance is small (see :func:`eb_pvalue`). ``"hb"``:
        Hoeffding-Bentkus, uses only the fixed ``[0, 1]`` range (the
        original construction; see :func:`hb_pvalue`).

    Returns
    -------
    dict
        ``lambda_star`` (float or ``None`` if no lambda was certified --
        VACUOUS), ``accepted_fraction`` (float, fraction of calibration
        points with ``score >= lambda_star``, 0.0 if vacuous), ``certified``
        (bool), ``alpha``, ``delta``, ``n``, ``procedure``, ``p_value`` (the
        construction name used), ``K`` (grid size), ``trace`` (list of
        ``{"lambda", "p_value", "rejected", "accepted_fraction",
        "empirical_risk"}``). For ``procedure="bonferroni"`` the trace
        covers the FULL grid (ascending lambda); for
        ``procedure="fixed-sequence"`` it is in TESTING order (descending
        lambda) and stops at the first non-rejection, as before.
    """
    losses = np.asarray(losses, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if losses.shape != scores.shape or losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses and scores must be equal-length, non-empty 1-D arrays")
    if np.any((losses < -1e-9) | (losses > 1.0 + 1e-9)):
        raise ValueError("losses must be bounded in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if procedure not in ("bonferroni", "fixed-sequence"):
        raise ValueError(f"procedure must be 'bonferroni' or 'fixed-sequence', got {procedure!r}")
    if p_value not in _P_VALUE_FUNCS:
        raise ValueError(f"p_value must be one of {sorted(_P_VALUE_FUNCS)}, got {p_value!r}")

    pfunc = _P_VALUE_FUNCS[p_value]
    n = losses.size
    grid = (
        np.asarray(sorted(set(float(v) for v in lambda_grid)), dtype=float)
        if lambda_grid is not None
        else build_lambda_grid(scores, n_grid=n_grid, min_accept_frac=min_accept_frac)
    )
    K = int(grid.size)

    def _test_lambda(lam: float) -> Dict[str, Any]:
        accept_mask = scores >= lam
        y = alpha + (losses - alpha) * accept_mask  # in [0, 1], all n points
        p = pfunc(y, alpha)
        acc_frac = float(accept_mask.mean())
        emp_risk = float(losses[accept_mask].mean()) if accept_mask.any() else float("nan")
        return {
            "lambda": float(lam),
            "p_value": p,
            "accepted_fraction": acc_frac,
            "empirical_risk": emp_risk,
        }

    trace: List[Dict[str, Any]] = []
    lambda_star: Optional[float] = None

    if procedure == "fixed-sequence":
        # Most conservative (highest lambda) first; halt at first non-rejection.
        for lam in grid[::-1]:
            entry = _test_lambda(lam)
            rejected = entry["p_value"] <= delta
            entry["rejected"] = bool(rejected)
            trace.append(entry)
            if rejected:
                lambda_star = entry["lambda"]
            else:
                break
    else:  # bonferroni
        level = delta / K if K > 0 else delta
        best_acc_frac = -1.0
        for lam in grid:
            entry = _test_lambda(lam)
            rejected = entry["p_value"] <= level
            entry["rejected"] = bool(rejected)
            trace.append(entry)
            if rejected and entry["accepted_fraction"] > best_acc_frac:
                best_acc_frac = entry["accepted_fraction"]
                lambda_star = entry["lambda"]

    if lambda_star is None:
        return {
            "lambda_star": None,
            "accepted_fraction": 0.0,
            "certified": False,
            "alpha": float(alpha),
            "delta": float(delta),
            "n": int(n),
            "procedure": procedure,
            "p_value": p_value,
            "K": K,
            "trace": trace,
        }

    final_mask = scores >= lambda_star
    return {
        "lambda_star": lambda_star,
        "accepted_fraction": float(final_mask.mean()),
        "certified": True,
        "alpha": float(alpha),
        "delta": float(delta),
        "n": int(n),
        "procedure": procedure,
        "p_value": p_value,
        "K": K,
        "trace": trace,
    }
