import numpy as np
import pytest

from asr_gate import ltt


def test_hb_pvalue_high_mean_never_rejects():
    y = np.full(500, 0.9)
    assert ltt.hb_pvalue(y, alpha=0.1) == pytest.approx(1.0)


def test_hb_pvalue_low_mean_large_n_rejects():
    rng = np.random.default_rng(0)
    y = np.clip(rng.normal(0.01, 0.02, size=5000), 0, 1)
    p = ltt.hb_pvalue(y, alpha=0.1)
    assert p < 0.01


def test_hb_pvalue_rejects_bounds():
    with pytest.raises(ValueError):
        ltt.hb_pvalue(np.array([1.5, 0.2]), alpha=0.1)
    with pytest.raises(ValueError):
        ltt.hb_pvalue(np.array([0.5]), alpha=1.5)


def test_build_lambda_grid_caps_top_by_min_accept_frac():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=1000)
    grid = ltt.build_lambda_grid(scores, n_grid=50, min_accept_frac=0.1)
    top = grid[-1]
    accepted_frac = float((scores >= top).mean())
    assert accepted_frac >= 0.09  # approx 10%, allow quantile-estimation slack


def test_ltt_certify_basic_shapes():
    rng = np.random.default_rng(1)
    n = 3000
    score = rng.uniform(0, 1, size=n)
    base = np.where(score > 0.5, 0.01, 0.20)
    loss = np.clip(base + rng.normal(0, 0.03, size=n), 0, 1)

    result = ltt.ltt_certify(loss, score, alpha=0.05, delta=0.1, n_grid=200, min_accept_frac=0.2)
    assert result["alpha"] == 0.05
    assert result["delta"] == 0.1
    assert result["n"] == n
    assert isinstance(result["trace"], list) and len(result["trace"]) >= 1
    if result["certified"]:
        assert result["lambda_star"] is not None
        assert 0.0 < result["accepted_fraction"] <= 1.0
    else:
        assert result["lambda_star"] is None
        assert result["accepted_fraction"] == 0.0


def test_ltt_certify_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ltt.ltt_certify(np.array([0.1, 0.2]), np.array([0.1]), alpha=0.05)  # shape mismatch
    with pytest.raises(ValueError):
        ltt.ltt_certify(np.array([1.5, 0.2]), np.array([0.1, 0.2]), alpha=0.05)  # loss out of [0,1]
    with pytest.raises(ValueError):
        ltt.ltt_certify(np.array([0.1, 0.2]), np.array([0.1, 0.2]), alpha=1.5)  # bad alpha


def _gen_step(n, seed):
    r = np.random.default_rng(seed)
    score = r.uniform(0, 1, size=n)
    base = np.where(score > 0.5, 0.01, 0.20)
    loss = np.clip(base + r.normal(0, 0.03, size=n), 0, 1)
    return score, loss


def test_ltt_certificate_violation_rate_within_delta():
    """Core correctness check (critical requirement #1): over many
    calibration resamples at a fixed (alpha, delta), the fraction of
    CERTIFIED draws whose TRUE selective risk at lambda* exceeds alpha
    must be <= delta (with Monte-Carlo slack), and certified draws must
    actually occur (this is not a vacuously-always-refuses test)."""
    # Ground truth: true R(lambda) from a very large draw.
    score_gt, loss_gt = _gen_step(2_000_000, seed=12345)

    def true_risk(lam: float) -> float:
        mask = score_gt >= lam
        return float(loss_gt[mask].mean()) if mask.sum() else 0.0

    alpha, delta = 0.05, 0.1
    n_resamples = 200
    n_cal = 8000
    violations = 0
    n_certified = 0
    accepted_fracs = []

    for seed in range(n_resamples):
        score, loss = _gen_step(n_cal, seed=30_000 + seed)
        result = ltt.ltt_certify(
            loss, score, alpha=alpha, delta=delta, n_grid=200, min_accept_frac=0.2
        )
        if not result["certified"]:
            continue
        n_certified += 1
        accepted_fracs.append(result["accepted_fraction"])
        if true_risk(result["lambda_star"]) > alpha:
            violations += 1

    # The scenario must actually exercise certification, not just vacuity.
    assert n_certified >= n_resamples * 0.5
    assert np.mean(accepted_fracs) > 0.1

    # Binomial Monte-Carlo slack on top of delta, per design §4 K3.
    mc_slack = 2 * np.sqrt(delta * (1 - delta) / n_certified)
    violation_rate = violations / n_certified
    assert violation_rate <= delta + mc_slack, (
        f"violation_rate={violation_rate} exceeds delta={delta} + slack={mc_slack}"
    )


def test_ltt_certify_vacuous_when_no_headroom():
    """When score carries zero information (independent of loss) and the
    target alpha is at/below the population mean risk, no lambda should be
    certifiable -- lambda_star is None and accepted_fraction is 0."""
    rng = np.random.default_rng(7)
    n = 2000
    score = rng.uniform(0, 1, size=n)
    loss = np.clip(rng.normal(0.15, 0.03, size=n), 0, 1)  # independent of score
    result = ltt.ltt_certify(loss, score, alpha=0.05, delta=0.1, n_grid=100, min_accept_frac=0.1)
    assert result["certified"] is False
    assert result["lambda_star"] is None
    assert result["accepted_fraction"] == 0.0


# ---------------------------------------------------------------------------
# eb_pvalue: unit sanity checks (mirrors the hb_pvalue checks above)
# ---------------------------------------------------------------------------


def test_eb_pvalue_high_mean_never_rejects():
    y = np.full(500, 0.9)
    assert ltt.eb_pvalue(y, alpha=0.1) == pytest.approx(1.0)


def test_eb_pvalue_low_mean_large_n_rejects():
    rng = np.random.default_rng(0)
    y = np.clip(rng.normal(0.01, 0.02, size=5000), 0, 1)
    p = ltt.eb_pvalue(y, alpha=0.1)
    assert p < 0.01


def test_eb_pvalue_rejects_bounds():
    with pytest.raises(ValueError):
        ltt.eb_pvalue(np.array([1.5, 0.2]), alpha=0.1)
    with pytest.raises(ValueError):
        ltt.eb_pvalue(np.array([0.5]), alpha=1.5)


def test_eb_pvalue_tighter_than_hb_on_pilot_top_grid_point():
    """Deterministic reproduction of the exact real-pilot mechanism (design
    §2.3 pilot finding): the top-of-grid lambda accepted ~10% of points with
    empirical risk ~0.0095 (well under alpha=0.02), yet Hoeffding-Bentkus
    gave p~0.90 because the OTHER 90% of points contribute exactly `alpha`
    to Y (no information, no variance) while HB's bound only sees the fixed
    [0, 1] range. Empirical-Bernstein, adapting to the (tiny) actual sample
    variance of Y, should be noticeably tighter on the identical statistic
    -- though on this single heavily-diluted grid point alone it need not
    fully reject; the full fix also requires testing the rest of the grid
    (see the Bonferroni-mode tests below)."""
    rng = np.random.default_rng(0)
    n = 3567
    alpha = 0.02
    n_acc = int(round(0.10 * n))
    accepted_loss = np.clip(rng.normal(0.0095, 0.01, size=n_acc), 0, 1)
    y = np.concatenate([alpha + (accepted_loss - alpha), np.full(n - n_acc, alpha)])

    p_hb = ltt.hb_pvalue(y, alpha)
    p_eb = ltt.eb_pvalue(y, alpha)
    assert p_hb > 0.5, f"expected HB to reproduce the pilot's stuck p-value, got {p_hb}"
    assert p_eb < p_hb, f"EB ({p_eb}) should be at least as tight as HB ({p_hb}) here"


# ---------------------------------------------------------------------------
# eb_pvalue: p-value validity on synthetic H0-boundary nulls
# (P(p <= u | H0) <= u for every u, per the module docstring's proof)
# ---------------------------------------------------------------------------


def test_eb_pvalue_validity_on_synthetic_null():
    alpha = 0.3
    n = 300
    n_reps = 3000
    rng = np.random.default_rng(7)
    ps = np.empty(n_reps)
    for i in range(n_reps):
        # E[Y] == alpha exactly (worst case within H0: E[Y] >= alpha).
        y = (rng.uniform(0.0, 1.0, size=n) < alpha).astype(float)
        ps[i] = ltt.eb_pvalue(y, alpha)

    for u in (0.05, 0.1, 0.2):
        rate = float(np.mean(ps <= u))
        mc_slack = 3.0 * np.sqrt(u * (1 - u) / n_reps)
        assert rate <= u + mc_slack, (
            f"u={u}: empirical P(p<=u)={rate} exceeds u+slack={u + mc_slack}"
        )


# ---------------------------------------------------------------------------
# Bonferroni-over-grid procedure
# ---------------------------------------------------------------------------


def test_ltt_certify_bonferroni_certifies_known_region_and_only_selects_rejected():
    """Synthetic case with a known-certifiable lambda region (score > 0.5
    implies low loss): Bonferroni mode must (a) certify, (b) select the
    REJECTED lambda with the largest accepted_fraction, and (c) never
    select a lambda whose own trace entry says `rejected=False`."""
    rng = np.random.default_rng(1)
    n = 3000
    score = rng.uniform(0, 1, size=n)
    base = np.where(score > 0.5, 0.005, 0.20)
    loss = np.clip(base + rng.normal(0, 0.01, size=n), 0, 1)

    result = ltt.ltt_certify(
        loss, score, alpha=0.05, delta=0.1, n_grid=200, min_accept_frac=0.2,
        procedure="bonferroni", p_value="eb",
    )
    assert result["procedure"] == "bonferroni"
    assert result["p_value"] == "eb"
    assert result["K"] == len(result["trace"])
    assert result["certified"] is True
    assert result["lambda_star"] is not None

    rejected_entries = [e for e in result["trace"] if e["rejected"]]
    assert rejected_entries, "expected at least one rejected grid point"

    # lambda_star must be the rejected entry with the largest accepted_fraction.
    best = max(rejected_entries, key=lambda e: e["accepted_fraction"])
    assert best["lambda"] == pytest.approx(result["lambda_star"])
    assert best["accepted_fraction"] == pytest.approx(result["accepted_fraction"])

    # Never select a non-rejected lambda: the trace entry matching
    # lambda_star must itself be rejected, and no non-rejected entry has a
    # larger accepted_fraction than what was selected.
    selected_entry = next(e for e in result["trace"] if e["lambda"] == result["lambda_star"])
    assert selected_entry["rejected"] is True
    non_rejected = [e for e in result["trace"] if not e["rejected"]]
    for e in non_rejected:
        assert e["lambda"] != result["lambda_star"]


def test_ltt_certify_bonferroni_never_selects_non_rejected_across_resamples():
    """Repeat the never-selects-non-rejected invariant over many independent
    resamples (not just one lucky seed)."""
    for seed in range(30):
        score, loss = _gen_step(1500, seed=seed)
        result = ltt.ltt_certify(
            loss, score, alpha=0.05, delta=0.1, n_grid=100, min_accept_frac=0.1,
            procedure="bonferroni", p_value="eb",
        )
        if result["lambda_star"] is None:
            continue
        selected_entry = next(e for e in result["trace"] if e["lambda"] == result["lambda_star"])
        assert selected_entry["rejected"] is True


def test_ltt_certificate_violation_rate_within_delta_bonferroni_eb():
    """Core correctness check, repeated for the new default path
    (procedure='bonferroni', p_value='eb'): over many calibration
    resamples at a fixed (alpha, delta), the fraction of CERTIFIED draws
    whose TRUE selective risk at lambda* exceeds alpha must be <= delta
    (with Monte-Carlo slack), and certified draws must actually occur."""
    score_gt, loss_gt = _gen_step(500_000, seed=54321)

    def true_risk(lam: float) -> float:
        mask = score_gt >= lam
        return float(loss_gt[mask].mean()) if mask.sum() else 0.0

    alpha, delta = 0.05, 0.1
    n_resamples = 150
    n_cal = 4000
    violations = 0
    n_certified = 0
    accepted_fracs = []

    for seed in range(n_resamples):
        score, loss = _gen_step(n_cal, seed=70_000 + seed)
        result = ltt.ltt_certify(
            loss, score, alpha=alpha, delta=delta, n_grid=150, min_accept_frac=0.2,
            procedure="bonferroni", p_value="eb",
        )
        if not result["certified"]:
            continue
        n_certified += 1
        accepted_fracs.append(result["accepted_fraction"])
        if true_risk(result["lambda_star"]) > alpha:
            violations += 1

    assert n_certified >= n_resamples * 0.5
    assert np.mean(accepted_fracs) > 0.1

    mc_slack = 2 * np.sqrt(delta * (1 - delta) / n_certified)
    violation_rate = violations / n_certified
    assert violation_rate <= delta + mc_slack, (
        f"violation_rate={violation_rate} exceeds delta={delta} + slack={mc_slack}"
    )


# ---------------------------------------------------------------------------
# Regression: the old (hb, fixed-sequence) path is unchanged, and it still
# exhibits the exact pilot-failure mechanism the new default path fixes.
# ---------------------------------------------------------------------------


def test_ltt_certify_fixed_sequence_hb_regression_basic_shapes():
    """Same scenario/assertions as test_ltt_certify_basic_shapes, but with
    the old procedure/p_value passed explicitly -- confirms the new
    parameters didn't change old-path behavior."""
    rng = np.random.default_rng(1)
    n = 3000
    score = rng.uniform(0, 1, size=n)
    base = np.where(score > 0.5, 0.01, 0.20)
    loss = np.clip(base + rng.normal(0, 0.03, size=n), 0, 1)

    result = ltt.ltt_certify(
        loss, score, alpha=0.05, delta=0.1, n_grid=200, min_accept_frac=0.2,
        procedure="fixed-sequence", p_value="hb",
    )
    assert result["procedure"] == "fixed-sequence"
    assert result["p_value"] == "hb"
    assert result["alpha"] == 0.05
    assert result["delta"] == 0.1
    assert result["n"] == n
    assert isinstance(result["trace"], list) and len(result["trace"]) >= 1
    if result["certified"]:
        assert result["lambda_star"] is not None
        assert 0.0 < result["accepted_fraction"] <= 1.0
    else:
        assert result["lambda_star"] is None
        assert result["accepted_fraction"] == 0.0


def test_ltt_certify_fixed_sequence_hb_regression_vacuous():
    rng = np.random.default_rng(7)
    n = 2000
    score = rng.uniform(0, 1, size=n)
    loss = np.clip(rng.normal(0.15, 0.03, size=n), 0, 1)
    result = ltt.ltt_certify(
        loss, score, alpha=0.05, delta=0.1, n_grid=100, min_accept_frac=0.1,
        procedure="fixed-sequence", p_value="hb",
    )
    assert result["certified"] is False
    assert result["lambda_star"] is None
    assert result["accepted_fraction"] == 0.0


def test_ltt_certify_old_path_reproduces_pilot_style_failure_new_default_fixes_it():
    """End-to-end synthetic reproduction of the real pilot's failure mode
    (macro-CER < alpha, but the fixed-sequence/HB combination still gets
    stuck at the top-of-grid, most-conservative lambda and certifies
    nothing): the OLD default (fixed-sequence, hb) stays vacuous, while the
    NEW default (bonferroni, eb) certifies a nontrivial accepted_fraction
    on the identical data."""
    rng = np.random.default_rng(2)
    n = 3600
    score = rng.normal(0, 1, size=n)
    base = np.clip(0.30 - 0.02 * score, 0.0, 1.0)
    is_bad = rng.uniform(0, 1, size=n) < base * 0.06
    loss = np.where(
        is_bad,
        np.clip(rng.uniform(0.1, 0.6, size=n), 0, 1),
        np.clip(np.abs(rng.normal(0, 0.003, size=n)), 0, 1),
    )
    assert loss.mean() < 0.02  # population macro-loss below alpha, like the real pilot

    alpha, delta = 0.02, 0.1
    old = ltt.ltt_certify(
        loss, score, alpha=alpha, delta=delta, n_grid=200, min_accept_frac=0.1,
        procedure="fixed-sequence", p_value="hb",
    )
    new = ltt.ltt_certify(
        loss, score, alpha=alpha, delta=delta, n_grid=200, min_accept_frac=0.1,
        procedure="bonferroni", p_value="eb",
    )
    assert old["certified"] is False
    assert new["certified"] is True
    assert new["accepted_fraction"] > 0.0
