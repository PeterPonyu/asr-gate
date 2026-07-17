import numpy as np
import pytest

from asr_gate import cer, scores
from tests.conftest import make_synthetic_utterances


def test_s1_s2_mean_and_min():
    nbest = [{"text": "abc", "logp": -3.0, "token_logps": [-1.0, -2.0, -0.5]}]
    assert scores.compute_s1(nbest) == pytest.approx(-3.5 / 3.0)
    assert scores.compute_s2(nbest) == pytest.approx(np.exp(-2.0))


def test_s1_s2_none_without_token_logps():
    nbest = [{"text": "abc", "logp": -3.0, "token_logps": None}]
    assert scores.compute_s1(nbest) is None
    assert scores.compute_s2(nbest) is None


def test_s3_margin_needs_two_hyps():
    nbest_one = [{"text": "abc", "logp": -3.0, "token_logps": None}]
    assert scores.compute_s3(nbest_one) is None

    nbest_two = [
        {"text": "abc", "logp": -3.0, "token_logps": None},
        {"text": "abd", "logp": -5.0, "token_logps": None},
    ]
    assert scores.compute_s3(nbest_two) == pytest.approx((-3.0 - -5.0) / 3.0)


def test_s3_none_when_logp_missing():
    nbest = [
        {"text": "abc", "logp": None, "token_logps": None},
        {"text": "abd", "logp": -5.0, "token_logps": None},
    ]
    assert scores.compute_s3(nbest) is None


def test_s4_needs_full_posteriors():
    nbest_no_posteriors = [{"text": "ab", "logp": -1.0, "token_logps": [-0.5, -0.5]}]
    assert scores.compute_s4(nbest_no_posteriors) is None

    # Two tokens; token 1 is a near-certain distribution (low entropy), token
    # 2 is uniform over 4 outcomes (higher entropy) -> negative mean entropy.
    nbest = [dict(nbest_no_posteriors[0])]
    nbest[0]["token_full_posteriors"] = [
        [0.97, 0.01, 0.01, 0.01],
        [0.25, 0.25, 0.25, 0.25],
    ]
    s4 = scores.compute_s4(nbest)
    assert s4 is not None
    assert s4 < 0  # negative entropy rate


def test_score_table_degraded_mode_flags_are_per_utterance():
    utterances = [
        {
            "utt_id": "a",
            "nbest": [{"text": "x", "logp": -1.0, "token_logps": [-0.5]}],
        },
        {
            "utt_id": "b",
            "nbest": [
                {"text": "x", "logp": -1.0, "token_logps": [-0.5]},
                {"text": "y", "logp": -2.0, "token_logps": None},
            ],
        },
    ]
    scored = scores.score_table(utterances)
    assert scored[0]["s3"] is None  # degraded: only 1-best
    assert scored[1]["s3"] is not None


def test_temperature_fit_apply_recovers_reasonable_scale():
    rng = np.random.default_rng(0)
    n = 2000
    s1 = rng.normal(0, 5.0, size=n)  # deliberately mis-scaled "logit"
    true_t = 3.0
    p = 1.0 / (1.0 + np.exp(-s1 / true_t))
    y = (rng.random(n) < p).astype(float)
    cer_vals = np.where(y == 1.0, 0.0, 0.2)  # y=1 <=> cer==0, matches fit_temperature's label

    t_hat = scores.fit_temperature(s1, cer_vals)
    assert t_hat > 0
    # Recovered temperature should be in a sane range of the true one.
    assert 1.0 < t_hat < 9.0

    s5 = scores.apply_temperature(s1, t_hat)
    assert s5.shape == s1.shape
    np.testing.assert_allclose(s5, s1 / t_hat)


def test_fit_temperature_rejects_missing_s1():
    with pytest.raises(ValueError):
        scores.fit_temperature(np.array([1.0, np.nan]), np.array([0.0, 0.1]))


def test_cer_regressor_fits_and_predicts_on_synthetic_signal():
    utterances = make_synthetic_utterances(n=400, n_speakers=20, seed=3)
    scored = scores.score_table(utterances)
    scored = cer.compute_cer_batch(scored)
    for u in scored:
        u["s5"] = u["s1"]  # stand-in, s5 fit is tested separately

    cer_arr = np.array([u["cer"] for u in scored])
    params = scores.fit_cer_regressor(scored, cer_arr)
    preds = scores.apply_cer_regressor(scored, params)

    # A regressor trained on data correlated with CER should beat a
    # constant-mean baseline on its own training data.
    mse_model = np.mean((preds - cer_arr) ** 2)
    mse_baseline = np.mean((cer_arr.mean() - cer_arr) ** 2)
    assert mse_model < mse_baseline


def test_primary_scores_constant():
    assert scores.PRIMARY_SCORES == ("s1", "s2", "s3", "s4", "s5")
