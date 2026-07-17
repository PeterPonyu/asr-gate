import pytest

from asr_gate import cer


def test_hand_computed_example():
    """ref '今天天气很好' vs hyp '今天天汽很好' -> 1 substitution / 6 chars = 1/6."""
    result = cer.compute_cer("今天天汽很好", "今天天气很好")
    assert result["edits"] == 1
    assert result["ref_len"] == 6
    assert result["cer"] == pytest.approx(1.0 / 6.0)
    assert result["clipped"] is False


def test_identical_strings_zero_cer():
    result = cer.compute_cer("今天天气很好", "今天天气很好")
    assert result["cer"] == 0.0
    assert result["edits"] == 0


def test_completely_different_strings():
    result = cer.compute_cer("我们明天去学校", "今天天气很好")
    assert result["ref_len"] == 6
    assert 0.0 < result["cer"] <= 1.0


def test_clipping_on_heavy_insertion():
    # hyp much longer than ref -> raw ratio can exceed 1, clipped.
    ref = "好"
    hyp = "今天天气很好啊呀呢吧"
    result = cer.compute_cer(hyp, ref)
    assert result["cer_raw"] > 1.0
    assert result["cer"] == 1.0
    assert result["clipped"] is True


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        cer.compute_cer("你好", "")


def test_normalizer_strips_punctuation_and_whitespace():
    assert cer.normalize_text("今天，天气 很好！") == "今天天气很好"


def test_normalizer_nfkc():
    # full-width digits normalize to half-width via NFKC.
    assert cer.normalize_text("１２３") == "123"


def test_numeral_policy_digits_to_cjk():
    assert cer.normalize_text("2023年", numeral_policy="digits-to-cjk") == "二〇二三年"


def test_numeral_policy_cjk_to_digits():
    assert cer.normalize_text("二〇二三年", numeral_policy="cjk-to-digits") == "2023年"


def test_numeral_policy_roundtrips_only_digit_substitution_not_reading():
    # Deliberately NOT a full Chinese-number reading: "234" != "二百三十四".
    assert cer.normalize_text("234", numeral_policy="digits-to-cjk") == "二三四"


def test_char_edit_distance_basic():
    assert cer.char_edit_distance("", "") == 0
    assert cer.char_edit_distance("abc", "") == 3
    assert cer.char_edit_distance("", "abc") == 3
    assert cer.char_edit_distance("kitten", "sitting") == 3


def test_compute_cer_batch_skips_missing_ref():
    utterances = [
        {"utt_id": "a", "hyp_text": "今天天汽很好", "ref_text": "今天天气很好"},
        {"utt_id": "b", "hyp_text": "你好", "ref_text": None},
    ]
    out = cer.compute_cer_batch(utterances)
    assert "cer" in out[0]
    assert "cer" not in out[1]


def test_micro_vs_macro_cer_are_named_apart_and_differ_on_unequal_lengths():
    utterances = [
        {"utt_id": "a", "hyp_text": "今", "ref_text": "今"},  # cer 0, ref_len 1
        {"utt_id": "b", "hyp_text": "错错错错错错错错错错", "ref_text": "今天天气很好呀呢吧啊"},  # cer 1, ref_len 10
    ]
    scored = cer.compute_cer_batch(utterances)
    macro = cer.macro_cer(scored)
    micro = cer.micro_cer(scored)
    # macro = mean(0, 1) = 0.5; micro = (0 + 10) / (1 + 10) = 10/11.
    assert macro == pytest.approx(0.5)
    assert micro == pytest.approx(10.0 / 11.0)
    assert macro != micro


def test_micro_macro_none_when_unscored():
    assert cer.micro_cer([{"utt_id": "a"}]) is None
    assert cer.macro_cer([{"utt_id": "a"}]) is None
