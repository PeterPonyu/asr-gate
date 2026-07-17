import pytest

from asr_gate import audit as _audit
from asr_gate import cer as _cer
from asr_gate import scores as _scores
from tests.conftest import make_synthetic_utterances


def _prep(n, n_speakers, seed, correlation_strength=1.0, degraded_frac=0.1):
    utterances = make_synthetic_utterances(
        n=n, n_speakers=n_speakers, seed=seed, correlation_strength=correlation_strength,
        degraded_frac=degraded_frac,
    )
    scored = _scores.score_table(utterances)
    return _cer.compute_cer_batch(scored)


def test_audit_flags_informative_score_not_noise_score():
    """s1 (derived from quality-correlated token_logps) must show positive
    excess-AURC and reject Holm; s3 (independent N-best margin, built-in
    noise comparator) must not."""
    utterances = _prep(n=2000, n_speakers=40, seed=20, degraded_frac=0.0)
    result = _audit.run_audit(utterances, score_names=["s1", "s3"], n_perm=1000, alpha=0.05, seed=20)

    by_score = {r["score"]: r for r in result["results"]}
    s1 = by_score["s1"]
    assert s1["excess_aurc"] > 0
    assert s1["p_value"] < 0.05
    assert s1["reject_holm"] is True

    s3 = by_score["s3"]
    assert s3["reject_holm"] is False


def test_holm_family_size_is_never_hardcoded():
    utterances = _prep(n=1500, n_speakers=30, seed=21)
    result_two = _audit.run_audit(utterances, score_names=["s1", "s2"], n_perm=200, seed=21)
    assert result_two["holm_family_size"] == 2

    result_all = _audit.run_audit(utterances, n_perm=200, seed=21)  # s1,s2,s4,s5 excluded (no s5); s3 present too depending on availability
    assert result_all["holm_family_size"] == len(result_all["results"])
    assert result_all["holm_family_size"] != 10  # never hardcoded


def test_holm_family_size_scales_with_backbone_roster():
    a = _prep(n=800, n_speakers=20, seed=22)
    b = _prep(n=800, n_speakers=20, seed=23)
    for u in a:
        u["backbone"] = "B1"
    for u in b:
        u["backbone"] = "B2"
    combined = a + b
    result = _audit.run_audit(
        combined, score_names=["s1", "s2"], backbone_field="backbone", n_perm=200, seed=22
    )
    # 2 scores x 2 backbones = 4, computed dynamically from the roster.
    assert result["holm_family_size"] == 4
    assert {r["backbone"] for r in result["results"]} == {"B1", "B2"}


def test_audit_requires_refs():
    utterances = make_synthetic_utterances(n=100, n_speakers=10, seed=24)
    scored = _scores.score_table(utterances)  # no cer
    with pytest.raises(_audit.AuditError, match="refs"):
        _audit.run_audit(scored)


def test_micro_cer_speaker_blocked_ci_present_and_sane():
    utterances = _prep(n=1200, n_speakers=30, seed=25)
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=200, seed=25)
    micro = result["micro_cer"]
    assert micro["n_blocks"] == 30
    lo, hi = micro["ci"]
    assert 0.0 <= lo <= micro["point"] <= hi <= 1.0


def test_macro_and_micro_cer_reported_separately_and_distinctly_named():
    utterances = _prep(n=1200, n_speakers=30, seed=26)
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=200, seed=26)
    assert "macro_cer" in result
    assert "micro_cer" in result
    assert isinstance(result["macro_cer"], float)
    assert isinstance(result["micro_cer"], dict)


def test_excess_aurc_matches_direct_relmetrics_calls():
    """Zero new math: excess-AURC must equal a direct relmetrics call on
    the same transformed arrays."""
    import numpy as np
    from relmetrics import aurc as _aurc_mod

    utterances = _prep(n=600, n_speakers=20, seed=27)
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=100, seed=27)
    r = result["results"][0]

    err = np.array([u["cer"] for u in utterances])
    score = np.array([u["s1"] for u in utterances])
    expected_excess = _aurc_mod.excess_aurc_gain(err, score)
    assert r["excess_aurc"] == pytest.approx(expected_excess, rel=1e-9)


def test_provenance_stamped():
    utterances = _prep(n=300, n_speakers=15, seed=28)
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=100, seed=28)
    assert "provenance" in result
    assert result["provenance"]["script"].endswith("audit.py")


def test_sparse_null_score_keeps_family_with_exclusion_count():
    """A degraded-tail missing score (e.g. 4/7,176 on the real Aishell-1
    test run) is excluded-and-counted, never a hard family drop -- mirrors
    gate.py's excluded_missing_s1 convention (critical correctness fix:
    previously ANY null in a score column dropped holm_family_size to 0
    for that score)."""
    utterances = _prep(n=500, n_speakers=20, seed=30, degraded_frac=0.0)
    for u in utterances[:3]:
        u["s1"] = None
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=200, seed=30)

    assert result["holm_family_size"] == 1
    r = result["results"][0]
    assert r["score"] == "s1"
    assert r["n_excluded"] == 3
    assert r["n"] == len(utterances) - 3
    assert result["skipped"] == []


def test_all_null_score_is_skipped_with_reason_not_dropped_silently():
    """A score that is None for EVERY row of a backbone (e.g. s4 with no
    token_full_posteriors, or s3 in single-hypothesis-only decodes) must be
    recorded in ``skipped`` with an explicit reason, never silently
    vanish from the roster while leaving holm_family_size unexplained."""
    utterances = _prep(n=300, n_speakers=15, seed=31, degraded_frac=0.0)
    # s4 needs token_full_posteriors, which the synthetic fixture never sets.
    result = _audit.run_audit(utterances, score_names=["s1", "s4"], n_perm=200, seed=31)

    assert result["holm_family_size"] == 1
    assert {r["score"] for r in result["results"]} == {"s1"}
    assert len(result["skipped"]) == 1
    skip = result["skipped"][0]
    assert skip["score"] == "s4"
    assert skip["n_missing"] == skip["n_total"] == len(utterances)
    assert "skipped_reason" in skip and skip["skipped_reason"]


def test_wholesale_missing_score_drops_family_with_reason():
    """>1% missing for a score is wholesale (the column was effectively
    never computed), not a degraded tail -- the family is still dropped,
    but with an explicit, non-silent reason recorded in ``skipped``."""
    utterances = _prep(n=500, n_speakers=20, seed=32, degraded_frac=0.0)
    for u in utterances[: len(utterances) // 2]:  # 50% missing >> 1% threshold
        u["s1"] = None
    result = _audit.run_audit(utterances, score_names=["s1"], n_perm=200, seed=32)

    assert result["holm_family_size"] == 0
    assert result["results"] == []
    assert len(result["skipped"]) == 1
    skip = result["skipped"][0]
    assert skip["score"] == "s1"
    assert skip["n_missing"] == len(utterances) // 2
    assert skip["n_missing"] < skip["n_total"]  # not all-null, but still wholesale
    assert "wholesale" in skip["skipped_reason"]
