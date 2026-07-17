import numpy as np
import pytest

from asr_gate import cer as _cer
from asr_gate import gate as _gate
from asr_gate import scores as _scores
from tests.conftest import make_synthetic_utterances


def _prep(n, n_speakers, seed, correlation_strength=1.0):
    utterances = make_synthetic_utterances(
        n=n, n_speakers=n_speakers, seed=seed, correlation_strength=correlation_strength
    )
    scored = _scores.score_table(utterances)
    return _cer.compute_cer_batch(scored)


def test_split_by_speaker_never_splits_a_speaker():
    utterances = _prep(n=800, n_speakers=40, seed=1)
    pool_a, pool_b = _gate.split_by_speaker(utterances, frac=0.5, seed=0)
    speakers_a = {u["speaker_id"] for u in pool_a}
    speakers_b = {u["speaker_id"] for u in pool_b}
    assert speakers_a.isdisjoint(speakers_b)
    assert len(pool_a) + len(pool_b) == len(utterances)


def test_calibrate_refuses_same_object_as_tune_and_cal():
    utterances = _prep(n=400, n_speakers=20, seed=2)
    with pytest.raises(_gate.GateError, match="same object"):
        _gate.calibrate_gate(utterances, tune_instances=utterances, alpha=0.05, delta=0.1)


def test_calibrate_refuses_overlapping_speakers_between_tune_and_cal():
    utterances = _prep(n=800, n_speakers=40, seed=3)
    pool_a, pool_b = _gate.split_by_speaker(utterances, frac=0.5, seed=0)
    # Deliberately leak: add one pool_b utterance's speaker into pool_a too.
    leaked = dict(pool_b[0])
    tainted_pool_a = pool_a + [leaked]
    with pytest.raises(_gate.GateError, match="speaker-disjoint"):
        _gate.calibrate_gate(tainted_pool_a, tune_instances=pool_b, alpha=0.05, delta=0.1)


def test_calibrate_requires_refs():
    utterances = make_synthetic_utterances(n=100, n_speakers=10, seed=4)
    scored = _scores.score_table(utterances)  # no cer computed
    with pytest.raises(_gate.GateError, match="refs"):
        _gate.calibrate_gate(scored, alpha=0.05, delta=0.1)


def test_calibrate_auto_split_speaker_disjoint():
    utterances = _prep(n=2000, n_speakers=40, seed=5)
    gate = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, fit_frac=0.5, seed=5, n_grid=150, min_accept_frac=0.15
    )
    assert gate["n_fit"] + gate["n_cal"] == len(utterances)
    assert "g1" in gate and "g2_map" in gate
    assert gate["normalizer_version"] == _cer.NORMALIZER_VERSION
    assert "provenance" in gate


def test_mondrian_small_stratum_flagged_defer_always():
    utterances = _prep(n=600, n_speakers=30, seed=6)
    gate = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, min_stratum_n=1_000_000, seed=6, n_grid=100,
    )
    # With an absurdly high floor, EVERY stratum must be defer-always.
    assert all(gate["strata"]["defer_always"].values())


def test_is_ood_flags_non_cjk_dominant_text():
    assert _gate.is_ood("hello world this is english") is True
    assert _gate.is_ood("今天天气很好") is False
    assert _gate.is_ood("今天天气很好x") is False  # 1/7 non-CJK, well under 20%


def test_apply_ood_refuse_exit_state_distinct_from_defer():
    utterances = _prep(n=1000, n_speakers=40, seed=8)
    gate = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, seed=8, n_grid=150, min_accept_frac=0.15
    )
    test_utterances = _prep(n=100, n_speakers=10, seed=9)
    test_utterances = _scores.score_table(test_utterances)
    ood_row = dict(test_utterances[0])
    ood_row["hyp_text"] = "this is completely english text with no chinese"
    test_utterances[0] = ood_row

    result = _gate.apply_gate(gate, test_utterances)
    actions = {d["utt_id"]: d["action"] for d in result["decisions"]}
    assert actions[ood_row["utt_id"]] == "OOD-REFUSE"
    assert set(actions.values()) <= {"ACCEPT", "DEFER", "OOD-REFUSE"}
    assert result["n_ood_refuse"] >= 1


def test_calibrate_apply_certificate_coverage_on_held_out_synthetic():
    """End-to-end: calibrate G1 on one synthetic draw, apply to a FRESH
    held-out synthetic draw from the same generator, and check the
    accepted set's true (ref-based) macro-CER respects alpha (informative
    score, strong correlation -> should hold comfortably)."""
    cal_utterances = _prep(n=6000, n_speakers=40, seed=10, correlation_strength=1.0)
    gate = _gate.calibrate_gate(
        cal_utterances, alpha=0.08, delta=0.1, seed=10, n_grid=200, min_accept_frac=0.2,
    )
    assert gate["g1"]["certified"] is True

    held_out = _prep(n=3000, n_speakers=40, seed=11, correlation_strength=1.0)
    result = _gate.apply_gate(gate, [dict(u) for u in held_out])
    accepted_ids = {d["utt_id"] for d in result["decisions"] if d["action"] == "ACCEPT"}
    assert len(accepted_ids) > 0

    accepted_cer = [u["cer"] for u in held_out if u["utt_id"] in accepted_ids]
    empirical_macro_cer = float(np.mean(accepted_cer))
    # Not a per-draw guarantee (that's the (alpha, delta) PAC statement,
    # checked statistically in test_ltt.py) -- a sanity bound with slack.
    assert empirical_macro_cer <= gate["alpha"] + 0.05


def test_guarantee_mondrian_ub_drives_different_decisions_than_ltt():
    utterances = _prep(n=3000, n_speakers=40, seed=12)
    gate_ltt = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, guarantee="ltt", seed=12, n_grid=150,
        min_accept_frac=0.15,
    )
    gate_ub = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, guarantee="mondrian-ub", seed=12, n_grid=150,
        min_accept_frac=0.15,
    )
    test_utterances = _scores.score_table(_prep(n=500, n_speakers=20, seed=13))
    res_ltt = _gate.apply_gate(gate_ltt, [dict(u) for u in test_utterances])
    res_ub = _gate.apply_gate(gate_ub, [dict(u) for u in test_utterances])
    assert res_ltt["guarantee"] == "ltt"
    assert res_ub["guarantee"] == "mondrian-ub"


def test_domain_fingerprint_ks_warning_fires_on_shifted_batch():
    utterances = _prep(n=3000, n_speakers=40, seed=14)
    gate = _gate.calibrate_gate(
        utterances, alpha=0.05, delta=0.1, seed=14, n_grid=150, min_accept_frac=0.15,
    )
    shifted = _prep(n=300, n_speakers=20, seed=15, correlation_strength=0.0)
    for u in shifted:
        # Force an extreme, obviously-shifted score distribution.
        u["duration_s"] = u["duration_s"]
    shifted = _scores.score_table(shifted)
    for u in shifted:
        if u.get("s1") is not None:
            u["s1"] = u["s1"] - 20.0  # large synthetic shift

    result = _gate.apply_gate(gate, shifted)
    check = result["domain_fingerprint_check"]
    assert check is not None
    assert check["warn"] is True


def test_calibrate_invalid_guarantee_rejected():
    utterances = _prep(n=200, n_speakers=20, seed=16)
    with pytest.raises(_gate.GateError, match="guarantee"):
        _gate.calibrate_gate(utterances, guarantee="bogus")


def test_calibrate_excludes_sparse_missing_s1_and_records_it():
    """A degraded-tail missing s1 (verified 1/7,142 on the real Aishell dev
    pilot) is excluded-and-counted, never a hard refusal; the exclusion is
    recorded in the gate record (mirrors G1's n_dropped_missing_score)."""
    utterances = _prep(n=800, n_speakers=40, seed=7)
    utterances[3]["s1"] = None  # one degraded utterance
    victim = utterances[3]["utt_id"]
    gate_record = _gate.calibrate_gate(utterances, alpha=0.05, delta=0.1, seed=0)
    exc = gate_record["excluded_missing_s1"]
    assert exc["n_fit_excluded"] + exc["n_cal_excluded"] == 1
    assert victim in exc["fit"] + exc["cal"]


def test_calibrate_still_refuses_wholesale_missing_s1():
    """>1% missing s1 means the input was never scored — hard refusal stays."""
    utterances = _prep(n=800, n_speakers=40, seed=8)
    for u in utterances[: len(utterances) // 2]:
        u["s1"] = None
    with pytest.raises(_gate.GateError, match="wholesale"):
        _gate.calibrate_gate(utterances, alpha=0.05, delta=0.1, seed=0)
