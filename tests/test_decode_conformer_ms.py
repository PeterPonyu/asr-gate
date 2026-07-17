"""Unit + integration tests for
``orchestration/decode_conformer_ms.py``'s tolerant FunASR-result
extraction, best-effort N-best beam_size fallback, and end-to-end canonical-
record shape.

No funasr/torch dependency: the end-to-end test injects a fake ``funasr``
module into ``sys.modules`` (mirroring
``tests/test_decode_paraformer.py``'s technique).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "decode_conformer_ms.py"
_spec = importlib.util.spec_from_file_location("decode_conformer_ms", _SCRIPT_PATH)
decode_conformer_ms = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(decode_conformer_ms)


# ---------------------------------------------------------------------------
# _strip_char_spacing / _extract_result (tolerant probing)
# ---------------------------------------------------------------------------


def test_strip_char_spacing_removes_spaces():
    assert decode_conformer_ms._strip_char_spacing("广 州 市") == "广州市"


def test_extract_result_text_only_shape():
    item = {"key": "U1", "text": "广 州 市", "timestamp": [[0, 1]]}
    result = decode_conformer_ms._extract_result(item)
    assert result == {"text": "广州市", "logp": None, "token_logps": None}


def test_extract_result_probes_logp_alias_keys():
    item = {"key": "U1", "text": "你 好", "avg_logprob": -1.5}
    result = decode_conformer_ms._extract_result(item)
    assert result["logp"] == pytest.approx(-1.5)


def test_extract_result_probes_token_logps_alias_keys():
    item = {"key": "U1", "text": "你 好", "token_score": [-0.1, -0.2]}
    result = decode_conformer_ms._extract_result(item)
    assert result["token_logps"] == pytest.approx([-0.1, -0.2])


def test_extract_result_warns_on_empty_text(capsys):
    result = decode_conformer_ms._extract_result({"key": "x"})
    assert result["text"] == ""
    assert "warning" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _extract_result: NESTED shapes (added 2026-07-09 after usable=0/3 on the
# box -- real result item has no flat score keys the original tolerant
# probe recognized; extend to search plausible nested structures too).
# ---------------------------------------------------------------------------


def test_extract_result_nested_nbest_list():
    item = {
        "key": "U1", "text": "广 州 市",
        "nbest": [{"text": "广 州 市", "avg_logprob": -0.9, "token_score": [-0.1, -0.2, -0.3]}],
    }
    result = decode_conformer_ms._extract_result(item)
    assert result["logp"] == pytest.approx(-0.9)
    assert result["token_logps"] == pytest.approx([-0.1, -0.2, -0.3])


def test_extract_result_nested_sentence_info_per_token_dicts():
    item = {
        "key": "U1", "text": "你 好",
        "sentence_info": [{"token": "你", "score": -0.4}, {"token": "好", "avg_logprob": -0.6}],
    }
    result = decode_conformer_ms._extract_result(item)
    assert result["token_logps"] == pytest.approx([-0.4, -0.6])


def test_extract_result_nested_sentence_info_incomplete_degrades_to_none():
    # one token dict carries no recognized score key -- must not half-guess
    # a misaligned sequence.
    item = {
        "key": "U1", "text": "你 好",
        "sentence_info": [{"token": "你", "score": -0.4}, {"token": "好"}],
    }
    result = decode_conformer_ms._extract_result(item)
    assert result["token_logps"] is None


def test_extract_result_nested_ct_us_score_array():
    item = {"key": "U1", "text": "甲 乙", "ctc_score": [-0.2, -0.3]}
    result = decode_conformer_ms._extract_result(item)
    assert result["token_logps"] == pytest.approx([-0.2, -0.3])


def test_extract_result_flat_keys_take_priority_over_nested():
    item = {
        "key": "U1", "text": "甲",
        "avg_logprob": -1.0,  # flat -- should win
        "nbest": [{"avg_logprob": -99.0}],
    }
    result = decode_conformer_ms._extract_result(item)
    assert result["logp"] == pytest.approx(-1.0)


def test_extract_result_no_recognizable_shape_stays_degraded():
    item = {"key": "U1", "text": "甲", "some_unrelated_field": 42}
    result = decode_conformer_ms._extract_result(item)
    assert result["logp"] is None
    assert result["token_logps"] is None


# ---------------------------------------------------------------------------
# _generate: best-effort beam_size N-best + fallback
# ---------------------------------------------------------------------------


class _FakeModelBeamOK:
    def generate(self, input, batch_size=1, beam_size=None):
        if beam_size:
            return [{"key": "u", "text": "甲"}, {"key": "u", "text": "乙"}]
        return [{"key": "u", "text": "甲"}]


def test_generate_requests_nbest_when_supported(tmp_path):
    warned = {}
    raw = decode_conformer_ms._generate(_FakeModelBeamOK(), tmp_path / "u.wav", 2, warned)
    assert len(raw) == 2
    assert warned == {}


class _FakeModelBeamRejected:
    def generate(self, input, batch_size=1):
        return [{"key": "u", "text": "甲"}]
    # no beam_size kwarg accepted at all -> TypeError on the beam_size call


def test_generate_falls_back_and_warns_once(tmp_path, capsys):
    warned = {}
    model = _FakeModelBeamRejected()
    raw1 = decode_conformer_ms._generate(model, tmp_path / "u1.wav", 3, warned)
    raw2 = decode_conformer_ms._generate(model, tmp_path / "u2.wav", 3, warned)
    assert len(raw1) == 1
    assert len(raw2) == 1
    err = capsys.readouterr().err
    assert err.count("warning: model.generate() rejected beam_size=") == 1
    assert warned == {"beam_size": True}


def test_generate_nbest_1_never_requests_beam_size(tmp_path):
    class _StrictModel:
        def generate(self, input, batch_size=1):
            return [{"key": "u", "text": "甲"}]

    warned = {}
    raw = decode_conformer_ms._generate(_StrictModel(), tmp_path / "u.wav", 1, warned)
    assert len(raw) == 1


def test_generate_wraps_dict_result_in_list(tmp_path):
    class _DictReturningModel:
        def generate(self, input, batch_size=1):
            return {"key": "u", "text": "甲"}

    warned = {}
    raw = decode_conformer_ms._generate(_DictReturningModel(), tmp_path / "u.wav", 1, warned)
    assert raw == [{"key": "u", "text": "甲"}]


# ---------------------------------------------------------------------------
# end-to-end: main() with fully mocked funasr module (no real funasr/torch)
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 1.0, rate: int = 16000) -> None:
    n_frames = int(duration_s * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


def test_main_end_to_end_thchs30_degraded_mode(tmp_path, monkeypatch):
    """Covers the THCHS-30 corpus path (decode_whisper.py's tests cover the
    Aishell path) and the degraded-mode (no token_logps) case, which is the
    realistic default for this unverified backbone (see module docstring)."""

    class _E2EModel:
        def __init__(self, model=None, device=None):
            pass

        def generate(self, input, batch_size=1):
            utt_id = Path(input).stem
            return [{"key": utt_id, "text": "你 好", "timestamp": [[0, 1], [1, 2]]}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    _make_wav(tmp_path / "test" / "A11_101.wav")
    (tmp_path / "test" / "A11_101.wav.trn").write_text("你好\n", encoding="utf-8")

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_conformer_ms.main(
        ["--split", "test", "--corpus", "thchs30", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["hyp_text"] == "你好"
    assert record["speaker_id"] == "A11"
    assert record["ref_text"] == "你好"
    assert record["nbest"][0]["token_logps"] is None  # degraded mode, per module docstring

    # Caller-shaped real-path check: validate against the canonical schema.
    from asr_gate.io import validate_utterances

    validate_utterances([record])


def test_main_end_to_end_aishell_nbest(tmp_path, monkeypatch):
    class _E2EModel:
        def __init__(self, model=None, device=None):
            pass

        def generate(self, input, batch_size=1, beam_size=None):
            utt_id = Path(input).stem
            if beam_size:
                return [
                    {"key": utt_id, "text": "广 州 市", "logp": -1.0},
                    {"key": utt_id, "text": "光 州 市", "logp": -2.0},
                ]
            return [{"key": utt_id, "text": "广 州 市"}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_conformer_ms.main(
        [
            "--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path),
            "--out", str(out_path), "--nbest", "2",
        ]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["hyp_text"] == "广州市"
    assert len(record["nbest"]) == 2
    assert record["nbest"][1]["text"] == "光州市"


def test_main_prints_machine_readable_score_counts(tmp_path, monkeypatch, capsys):
    """Final summary line must expose n_with_logp=/n_with_token_logps= so
    chain gates can read decoder-reported numbers (rather than re-deriving
    them by re-parsing the JSONL)."""

    class _E2EModel:
        def __init__(self, model=None, device=None):
            pass

        def generate(self, input, batch_size=1):
            utt_id = Path(input).stem
            if utt_id == "UTT0001":
                return [{"key": utt_id, "text": "广 州 市", "avg_logprob": -0.5}]
            return [{"key": utt_id, "text": "光 州 市"}]  # no score keys -- degraded

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    _make_wav(wav_dir / "UTT0002.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\nUTT0002 光州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_conformer_ms.main(
        ["--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_with_logp=1" in out
    assert "n_with_token_logps=0" in out


# ---------------------------------------------------------------------------
# --probe mode: dumps the raw funasr result structure for live shape
# verification, no canonical JSONL, no _extract_result applied.
# ---------------------------------------------------------------------------


def test_probe_mode_dumps_raw_structure_no_canonical_jsonl(tmp_path, monkeypatch, capsys):
    class _E2EModel:
        def __init__(self, model=None, device=None):
            pass

        def generate(self, input, batch_size=1):
            utt_id = Path(input).stem
            return [{"key": utt_id, "text": "广 州 市", "weird_nested": {"a": [1, 2, 3]}}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    _make_wav(wav_dir / "UTT0002.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\nUTT0002 广州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_conformer_ms.main(
        [
            "--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path),
            "--out", str(out_path), "--probe", "1",
        ]
    )
    assert rc == 0
    assert not out_path.exists()  # probe mode never writes the canonical JSONL

    probe_path = out_path.with_suffix(".probe.json")
    assert probe_path.exists()
    probed = json.loads(probe_path.read_text(encoding="utf-8"))
    assert len(probed) == 1
    assert probed[0]["utt_id"] == "UTT0001"
    assert "weird_nested" in probed[0]["raw_repr"][0]

    stdout = capsys.readouterr().out
    assert "PROBE mode" in stdout
    assert "weird_nested" in stdout


def test_probe_mode_truncates_long_repr_and_records_generate_errors(tmp_path, monkeypatch):
    class _E2EModelOneBad:
        def __init__(self, model=None, device=None):
            pass

        def generate(self, input, batch_size=1):
            utt_id = Path(input).stem
            if utt_id == "UTT0001":
                raise RuntimeError("boom")
            return [{"key": utt_id, "text": "x" * 5000}]  # forces repr truncation

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModelOneBad
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    _make_wav(wav_dir / "UTT0002.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\nUTT0002 广州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_conformer_ms.main(
        [
            "--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path),
            "--out", str(out_path), "--probe", "2",
        ]
    )
    assert rc == 0

    probe_path = out_path.with_suffix(".probe.json")
    probed = json.loads(probe_path.read_text(encoding="utf-8"))
    assert len(probed) == 2
    assert "error" in probed[0]
    assert "boom" in probed[0]["error"]
    assert len(probed[1]["raw_repr"][0]) <= 2000


def test_decode_conformer_ms_help_runs_without_funasr():
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"], capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
