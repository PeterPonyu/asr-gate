"""Unit tests for ``orchestration/decode_sherpa_onnx.py``'s sherpa-onnx result
-> canonical nbest mapping and per-token ``ys_log_probs`` alignment logic.

No sherpa_onnx/soundfile dependency: the ``OfflineRecognizerResult`` is stood
in for with a tiny fake object exposing exactly the attributes the mapping
reads (``.text``, ``.tokens``, ``.ys_log_probs`` / ``.ys_probs``). The real
``ys_log_probs`` population under a live transducer decode is verified on-box in
the ``--probe`` smoke (FREEZE-AMENDMENT-2026-07-13.md §1); these tests pin the
alignment LOGIC on a synthetic fixture only, exactly as
``tests/test_decode_paraformer.py`` pins the paraformer hook logic without
funasr.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "decode_sherpa_onnx.py"
_spec = importlib.util.spec_from_file_location("decode_sherpa_onnx", _SCRIPT_PATH)
decode_sherpa_onnx = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(decode_sherpa_onnx)


class _FakeResult:
    """Stands in for a sherpa-onnx OfflineRecognizerResult: only the attributes
    the mapping reads. ``ys_attr`` selects which log-prob attribute name is
    populated (``ys_log_probs`` on current master, ``ys_probs`` on alternate
    builds) -- mirrors the getattr fallback in the real result object."""

    def __init__(self, text, tokens, ys=None, ys_attr="ys_log_probs"):
        self.text = text
        self.tokens = tokens
        if ys is not None:
            setattr(self, ys_attr, ys)


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------


def test_clean_text_strips_word_boundary_marker_and_whitespace():
    # k2/icefall word-boundary marker U+2581 "▁" + inter-token spacing.
    assert decode_sherpa_onnx._clean_text("广 州 市") == "广州市"
    assert decode_sherpa_onnx._clean_text("▁THE ▁END") == "THEEND"


def test_clean_text_empty():
    assert decode_sherpa_onnx._clean_text("") == ""


# ---------------------------------------------------------------------------
# _extract_ys_log_probs
# ---------------------------------------------------------------------------


def test_extract_ys_log_probs_reads_ys_log_probs():
    r = _FakeResult("广州", ["广", "州"], ys=[-0.1, -0.2], ys_attr="ys_log_probs")
    assert decode_sherpa_onnx._extract_ys_log_probs(r) == pytest.approx([-0.1, -0.2])


def test_extract_ys_log_probs_falls_back_to_ys_probs():
    r = _FakeResult("广州", ["广", "州"], ys=[-0.3, -0.4], ys_attr="ys_probs")
    assert decode_sherpa_onnx._extract_ys_log_probs(r) == pytest.approx([-0.3, -0.4])


def test_extract_ys_log_probs_none_when_absent():
    r = _FakeResult("广州", ["广", "州"], ys=None)
    assert decode_sherpa_onnx._extract_ys_log_probs(r) is None


def test_extract_ys_log_probs_none_when_empty_list():
    r = _FakeResult("", [], ys=[], ys_attr="ys_log_probs")
    assert decode_sherpa_onnx._extract_ys_log_probs(r) is None


# ---------------------------------------------------------------------------
# _result_to_nbest -- the core ys_probs -> token -> char alignment
# ---------------------------------------------------------------------------


def test_result_to_nbest_pure_char_tokens_aligns_one_to_one():
    # Chinese char-level transducer: tokens ARE chars, ys_log_probs 1:1.
    r = _FakeResult("广州市", ["广", "州", "市"], ys=[-0.1, -0.2, -0.3])
    nb = decode_sherpa_onnx._result_to_nbest(r)
    assert nb["text"] == "广州市"
    assert nb["token_logps"] == pytest.approx([-0.1, -0.2, -0.3])
    assert nb["logp"] == pytest.approx(-0.6)


def test_result_to_nbest_builds_text_from_tokens_stripping_markers():
    # Mixed script: an English word carries the U+2581 word-start marker; the
    # marker is stripped for hyp_text, and token_logps stays 1:1 with tokens.
    r = _FakeResult("ignored", ["你", "好", "▁OK"], ys=[-0.1, -0.2, -0.9])
    nb = decode_sherpa_onnx._result_to_nbest(r)
    assert nb["text"] == "你好OK"
    assert nb["token_logps"] == pytest.approx([-0.1, -0.2, -0.9])


def test_result_to_nbest_falls_back_to_result_text_when_no_tokens():
    r = _FakeResult("广 州", tokens=[], ys=None)
    nb = decode_sherpa_onnx._result_to_nbest(r)
    assert nb["text"] == "广州"
    assert nb["token_logps"] is None
    assert nb["logp"] is None


def test_result_to_nbest_degrades_on_length_mismatch(capsys):
    # ys_log_probs count != tokens count -> token_logps None, loud warning,
    # never a silent misalignment of scores to the wrong characters.
    r = _FakeResult("广州市", ["广", "州", "市"], ys=[-0.1, -0.2])
    nb = decode_sherpa_onnx._result_to_nbest(r)
    assert nb["text"] == "广州市"  # text still recovered
    assert nb["token_logps"] is None
    assert nb["logp"] is None
    assert "ys_log_probs length" in capsys.readouterr().err


def test_result_to_nbest_text_without_posteriors_still_decodes():
    # A build that exposes tokens but no ys_log_probs: text is emitted, scores
    # degrade to None (s1/s2 become None downstream) rather than crashing.
    r = _FakeResult("广州", ["广", "州"], ys=None)
    nb = decode_sherpa_onnx._result_to_nbest(r)
    assert nb["text"] == "广州"
    assert nb["token_logps"] is None


# ---------------------------------------------------------------------------
# --help works with no sherpa_onnx installed (lazy-import contract)
# ---------------------------------------------------------------------------


def test_help_runs_without_sherpa_onnx():
    # argparse --help exits 0 BEFORE _load_recognizer's lazy sherpa_onnx import
    # is ever reached -- the module must be importable and --help usable in an
    # environment with neither sherpa_onnx nor soundfile.
    with pytest.raises(SystemExit) as ei:
        decode_sherpa_onnx.main(["--help"])
    assert ei.value.code == 0
