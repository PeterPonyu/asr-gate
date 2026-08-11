"""Conformal risk control helpers (SSOT for CRC get_lhat).

Minimal implementation of the standard monotone empirical risk threshold
search used by Angelopoulos et al. conformal risk control. Imported by
003/007 packages; do not reimplement DIY crc_lambda in frontier runners.
"""
from __future__ import annotations

from typing import Callable, Sequence
import numpy as np


def get_lhat(
    losses: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> float:
    """Return the smallest lambda such that n/(n+1) * Rhat(lambda) + B/(n+1) <= alpha.

    Parameters
    ----------
    losses : array-like, shape (n, L) or callable path via precomputed grid
        If 2D, losses[i, j] = loss of example i at lambdas[j].
        If 1D of length L, treated as already-averaged Rhat(lambda) (discouraged).
    lambdas : decreasing or arbitrary grid of thresholds (will be sorted ascending)
    alpha : target risk level in (0, 1)
    B : bound on loss (default 1.0 for [0,1] losses)
    """
    lambdas = np.asarray(lambdas, dtype=float)
    order = np.argsort(lambdas)
    lambdas_s = lambdas[order]
    losses = np.asarray(losses, dtype=float)
    if losses.ndim == 1:
        rhat = losses[order]
        n = None  # unknown; use raw rhat + conservative note
        # Without n, use rhat <= alpha (weaker). Prefer 2D input.
        for lam, r in zip(lambdas_s, rhat):
            if r <= alpha:
                return float(lam)
        return float(lambdas_s[-1])
    if losses.ndim != 2:
        raise ValueError("losses must be 1D or 2D")
    # losses: (n, L) matching original lambda order
    losses_s = losses[:, order]
    n = losses_s.shape[0]
    rhat = losses_s.mean(axis=0)
    bound = (n / (n + 1.0)) * rhat + B / (n + 1.0)
    ok = np.where(bound <= alpha)[0]
    if ok.size == 0:
        return float(lambdas_s[-1])
    return float(lambdas_s[ok[0]])


def crc_select_threshold(
    scores: np.ndarray,
    labels_loss_fn: Callable[[np.ndarray, float], np.ndarray],
    alpha: float,
    lambda_grid: np.ndarray | None = None,
    B: float = 1.0,
) -> float:
    """scores: (n,), loss_fn(scores, lam) -> (n,) losses in [0,B]."""
    if lambda_grid is None:
        lambda_grid = np.linspace(scores.min(), scores.max(), 101)
    losses = np.stack([labels_loss_fn(scores, float(lam)) for lam in lambda_grid], axis=1)
    return get_lhat(losses, lambda_grid, alpha=alpha, B=B)
