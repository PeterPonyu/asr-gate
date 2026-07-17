"""Unit + integration tests for ``orchestration/decode_paraformer.py``'s
FunASR-output mapping (hyp_text/whitespace-stripping) and token-logp-hook
alignment logic (bug fixes discovered by the real-model box smoke test).

No funasr/torch dependency: the hook's captured tensor is stood in for
with a plain numpy array matching the documented post-hook contract
(``capture["log_probs"]`` is a numpy array after ``.detach().cpu().numpy()``
-- see ``_install_token_logp_hook``'s docstring), and the end-to-end test
injects a fake ``funasr`` module into ``sys.modules`` rather than
requiring the real package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "decode_paraformer.py"
_spec = importlib.util.spec_from_file_location("decode_paraformer", _SCRIPT_PATH)
decode_paraformer = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(decode_paraformer)


# ---------------------------------------------------------------------------
# hyp_text mapping / whitespace-stripping
# ---------------------------------------------------------------------------


def test_strip_char_spacing_removes_funasr_inter_char_spaces():
    assert decode_paraformer._strip_char_spacing("广 州 市 房 地 产") == "广州市房地产"


def test_strip_char_spacing_handles_empty_string():
    assert decode_paraformer._strip_char_spacing("") == ""


def test_extract_result_maps_hyp_text_from_verified_funasr_shape():
    # Verified real shape (funasr 1.3.14, model="paraformer-zh"): only
    # key/text/timestamp, text has FunASR's inter-character spacing.
    item = {"key": "BAC009S0724W0121", "text": "广 州 市 房 地 产", "timestamp": [[0, 1]]}
    result = decode_paraformer._extract_result(item)
    assert result["text"] == "广州市房地产"
    assert result["logp"] is None
    assert result["token_logps"] is None


def test_extract_result_warns_on_empty_text(capsys):
    result = decode_paraformer._extract_result({"key": "x"})
    assert result["text"] == ""
    assert "warning" in capsys.readouterr().err


def test_adapt_funasr_strips_inter_character_spacing():
    from asr_gate.io import adapt_funasr

    records = [
        {"key": "U1", "text": "广 州 市 房 地 产", "timestamp": [[0, 1]]},
    ]
    out = adapt_funasr(records)
    assert out[0]["hyp_text"] == "广州市房地产"
    assert out[0]["nbest"][0]["text"] == "广州市房地产"


# ---------------------------------------------------------------------------
# token-logp hook alignment
# ---------------------------------------------------------------------------


def _fake_vocab_logprobs(chosen_ids, vocab_size=10):
    """Build a ``[1, T, vocab]`` numpy array where each row's argmax is
    ``chosen_ids[t]`` with a distinct, unambiguous log-prob value."""
    t = len(chosen_ids)
    arr = np.full((1, t, vocab_size), -50.0, dtype=float)
    for i, tid in enumerate(chosen_ids):
        arr[0, i, tid] = -0.1 * (i + 1)
    return arr


def test_extract_hooked_token_logps_aligns_after_filtering_sos_eos_blank():
    # sos=1, eos=2, blank=0 (matches the real funasr SeaCo-Paraformer ids
    # verified on the decode box). 4 real characters (ids 5,6,7,8) plus a
    # trailing eos row -- the exact T == n_chars + 1 pattern verified
    # empirically against real Aishell-1 dev utterances.
    chosen_ids = [5, 6, 7, 8, 2]
    log_probs = _fake_vocab_logprobs(chosen_ids)
    capture = {"log_probs": log_probs, "_sos": 1, "_eos": 2, "_blank_id": 0}
    result = decode_paraformer._extract_hooked_token_logps(capture, n_chars=4)
    assert result == pytest.approx([-0.1, -0.2, -0.3, -0.4])


def test_extract_hooked_token_logps_filters_blank_too():
    chosen_ids = [5, 0, 6, 7, 2]  # a blank (id 0) row in the middle
    log_probs = _fake_vocab_logprobs(chosen_ids)
    capture = {"log_probs": log_probs, "_sos": 1, "_eos": 2, "_blank_id": 0}
    result = decode_paraformer._extract_hooked_token_logps(capture, n_chars=3)
    assert result == pytest.approx([-0.1, -0.3, -0.4])


def test_extract_hooked_token_logps_none_when_capture_empty():
    assert decode_paraformer._extract_hooked_token_logps({}, n_chars=3) is None


def test_extract_hooked_token_logps_none_on_alignment_mismatch(capsys):
    chosen_ids = [5, 6, 7]  # no eos row -- won't match n_chars=5
    log_probs = _fake_vocab_logprobs(chosen_ids)
    capture = {"log_probs": log_probs, "_sos": 1, "_eos": 2, "_blank_id": 0}
    result = decode_paraformer._extract_hooked_token_logps(capture, n_chars=5)
    assert result is None
    assert "alignment mismatch" in capsys.readouterr().err


class _FakeTensor:
    """Stands in for a torch.Tensor: only the methods the hook calls."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeInner:
    sos, eos, blank_id = 1, 2, 0

    def _seaco_decode_with_ASF(self, *args, **kwargs):
        return _FakeTensor(np.zeros((1, 3, 10)))


class _FakeModel:
    def __init__(self, model=None, device=None):
        self.model = _FakeInner()


def test_install_token_logp_hook_captures_tensor_on_call():
    fake_model = _FakeModel()
    capture = decode_paraformer._install_token_logp_hook(fake_model)
    assert "log_probs" not in capture

    fake_model.model._seaco_decode_with_ASF()
    assert "log_probs" in capture
    assert capture["log_probs"].shape == (1, 3, 10)
    assert (capture["_sos"], capture["_eos"], capture["_blank_id"]) == (1, 2, 0)


def test_install_token_logp_hook_degrades_when_method_missing(capsys):
    class _FakeModelNoHook:
        model = object()

    capture = decode_paraformer._install_token_logp_hook(_FakeModelNoHook())
    assert capture == {}
    assert "warning" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# end-to-end: main() with a fully mocked funasr module (no real funasr/torch)
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 1.0, rate: int = 16000) -> None:
    n_frames = int(duration_s * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


# ---------------------------------------------------------------------------
# --wav-list input mode (added 2026-07-09 for the noise-stratified arm --
# decode an arbitrary flat wav list, e.g. mix_musan.py's mixed-audio
# output directory, without disturbing the default Aishell-wav-tree path
# exercised above).
# ---------------------------------------------------------------------------


def test_wavlist_speaker_id_extracts_aishell_pattern():
    assert decode_paraformer._wavlist_speaker_id("BAC009S0724W0121") == "S0724"


def test_wavlist_speaker_id_falls_back_to_whole_utt_id_when_no_match():
    assert decode_paraformer._wavlist_speaker_id("noisy_utt_42") == "noisy_utt_42"


def test_load_wav_list_entries_parses_and_derives_speaker(tmp_path):
    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(
        "BAC009S0724W0121 /root/mixed/BAC009S0724W0121.wav\n"
        "BAC009S0725W0001 /root/mixed/BAC009S0725W0001.wav\n",
        encoding="utf-8",
    )
    entries = decode_paraformer._load_wav_list_entries(wav_list, limit=None)
    assert entries == [
        ("BAC009S0724W0121", "S0724", Path("/root/mixed/BAC009S0724W0121.wav")),
        ("BAC009S0725W0001", "S0725", Path("/root/mixed/BAC009S0725W0001.wav")),
    ]


def test_load_wav_list_entries_respects_limit(tmp_path):
    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text("u1 /a/u1.wav\nu2 /a/u2.wav\nu3 /a/u3.wav\n", encoding="utf-8")
    entries = decode_paraformer._load_wav_list_entries(wav_list, limit=2)
    assert len(entries) == 2


def test_load_wav_list_entries_skips_malformed_lines(tmp_path):
    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text("u1 /a/u1.wav\nmalformed_no_path\n\nu2 /a/u2.wav\n", encoding="utf-8")
    entries = decode_paraformer._load_wav_list_entries(wav_list, limit=None)
    assert [e[0] for e in entries] == ["u1", "u2"]


def test_main_end_to_end_wav_list_mode_with_aishell_transcript_source(tmp_path, monkeypatch):
    class _E2EInner:
        sos, eos, blank_id = 1, 2, 0

        def _seaco_decode_with_ASF(self, *a, **k):
            arr = np.full((1, 3, 10), -50.0, dtype=float)
            for i, tid in enumerate([5, 6, 2]):
                arr[0, i, tid] = -0.1 * (i + 1)
            return _FakeTensor(arr)

    class _E2EModel:
        def __init__(self, model=None, device=None):
            self.model = _E2EInner()

        def generate(self, input, batch_size=1):
            self.model._seaco_decode_with_ASF()
            utt_id = Path(input).stem
            return [{"key": utt_id, "text": "你 好", "timestamp": [[0, 1], [1, 2]]}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "mixed_5db"
    wav_dir.mkdir()
    _make_wav(wav_dir / "BAC009S0724W0121.wav")

    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "BAC009S0724W0121 你好\n", encoding="utf-8"
    )

    wav_list = tmp_path / "wavlist.txt"
    wav_list.write_text(f"BAC009S0724W0121 {wav_dir / 'BAC009S0724W0121.wav'}\n", encoding="utf-8")

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_paraformer.main(
        [
            "--split", "test",  # unused in --wav-list mode, still required by argparse
            "--data-root", str(tmp_path),
            "--wav-list", str(wav_list),
            "--out", str(out_path),
        ]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["utt_id"] == "BAC009S0724W0121"
    assert record["speaker_id"] == "S0724"
    assert record["hyp_text"] == "你好"
    assert record["ref_text"] == "你好"  # resolved via --transcript-source aishell (default)

    from asr_gate.io import validate_utterances

    validate_utterances([record])


def test_main_wav_list_mode_transcript_source_none_leaves_ref_null(tmp_path, monkeypatch):
    class _E2EModel:
        def __init__(self, model=None, device=None):
            self.model = type("I", (), {"sos": 1, "eos": 2, "blank_id": 0})()

        def generate(self, input, batch_size=1):
            utt_id = Path(input).stem
            return [{"key": utt_id, "text": "你 好"}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "mixed_5db"
    wav_dir.mkdir()
    _make_wav(wav_dir / "noisy_utt_1.wav")

    wav_list = tmp_path / "wavlist.txt"
    wav_list.write_text(f"noisy_utt_1 {wav_dir / 'noisy_utt_1.wav'}\n", encoding="utf-8")

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_paraformer.main(
        [
            "--split", "test",
            "--data-root", str(tmp_path),
            "--wav-list", str(wav_list),
            "--transcript-source", "none",
            "--out", str(out_path),
        ]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["ref_text"] is None
    assert record["speaker_id"] == "noisy_utt_1"  # no S#### match -- falls back to whole utt_id


def test_main_wav_list_mode_missing_transcript_file_aborts_loudly(tmp_path):
    wav_list = tmp_path / "wavlist.txt"
    wav_list.write_text(f"u1 {tmp_path / 'u1.wav'}\n", encoding="utf-8")
    out_path = tmp_path / "decode_out.jsonl"
    with pytest.raises(SystemExit):
        decode_paraformer.main(
            [
                "--split", "test",
                "--data-root", str(tmp_path),  # no transcript/ dir under here
                "--wav-list", str(wav_list),
                "--out", str(out_path),
            ]
        )


def test_main_end_to_end_hyp_text_and_token_logps(tmp_path, monkeypatch):
    class _E2EInner:
        sos, eos, blank_id = 1, 2, 0

        def _seaco_decode_with_ASF(self, *a, **k):
            # 3 real characters (ids 5,6,7) + trailing eos row (id 2).
            arr = np.full((1, 4, 10), -50.0, dtype=float)
            for i, tid in enumerate([5, 6, 7, 2]):
                arr[0, i, tid] = -0.1 * (i + 1)
            return _FakeTensor(arr)

    class _E2EModel:
        def __init__(self, model=None, device=None):
            self.model = _E2EInner()

        def generate(self, input, batch_size=1):
            # Mimic inference() internally calling the (hooked) method.
            self.model._seaco_decode_with_ASF()
            utt_id = Path(input).stem
            return [{"key": utt_id, "text": "广 州 市", "timestamp": [[0, 1], [1, 2], [2, 3]]}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    wav_dir.mkdir(parents=True)
    _make_wav(wav_dir / "UTT0001.wav")

    transcript_dir = tmp_path / "transcript"
    transcript_dir.mkdir()
    (transcript_dir / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_paraformer.main(
        ["--split", "dev", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["hyp_text"] == "广州市"
    assert record["nbest"][0]["text"] == "广州市"
    assert record["nbest"][0]["token_logps"] == pytest.approx([-0.1, -0.2, -0.3])
    assert record["nbest"][0]["logp"] == pytest.approx(-0.6)
    assert record["ref_text"] == "广州市"


# ---------------------------------------------------------------------------
# --corpus switch (added 2026-07-13): a NON-Aishell corpus routes through
# asr_gate.corpora.discover_corpus. The DEFAULT --corpus aishell path is
# unchanged -- verified by every other e2e test above (none pass --corpus, so
# they exercise the untouched inlined Aishell discovery; the diff-guard).
# ---------------------------------------------------------------------------


def test_main_corpus_switch_routes_thchs30_through_discoverer(tmp_path, monkeypatch):
    class _E2EInner:
        sos, eos, blank_id = 1, 2, 0

        def _seaco_decode_with_ASF(self, *a, **k):
            arr = np.full((1, 3, 10), -50.0, dtype=float)
            for i, tid in enumerate([5, 6, 2]):  # 2 chars + eos
                arr[0, i, tid] = -0.1 * (i + 1)
            return _FakeTensor(arr)

    class _E2EModel:
        def __init__(self, model=None, device=None):
            self.model = _E2EInner()

        def generate(self, input, batch_size=1):
            self.model._seaco_decode_with_ASF()
            return [{"key": Path(input).stem, "text": "你 好", "timestamp": [[0, 1], [1, 2]]}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = _E2EModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    # THCHS-30 layout: split dir wav + data/ .wav.trn sidecar (discover_thchs30).
    (tmp_path / "test").mkdir()
    _make_wav(tmp_path / "test" / "A11_101.wav")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "A11_101.wav.trn").write_text("你 好\n", encoding="utf-8")

    out_path = tmp_path / "dec_para_thchs30.jsonl"
    rc = decode_paraformer.main(
        ["--split", "test", "--corpus", "thchs30",
         "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["utt_id"] == "A11_101"
    assert record["speaker_id"] == "A11"       # from discover_thchs30
    assert record["hyp_text"] == "你好"
    assert record["ref_text"] == "你好"        # ref carried by the discoverer

    from asr_gate.io import validate_utterances

    validate_utterances([record])
