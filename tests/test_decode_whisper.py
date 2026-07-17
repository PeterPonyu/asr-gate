"""Unit + integration tests for ``orchestration/decode_whisper.py``'s
token-logp -> character alignment (:func:`_align_token_logps_to_chars`) and
end-to-end canonical-record shape.

No transformers/torch/soundfile dependency: the end-to-end test injects
fake ``torch``/``transformers`` modules into ``sys.modules`` (mirroring
``tests/test_decode_paraformer.py``'s fake-``funasr``-module technique) and
monkeypatches ``_load_audio`` directly rather than faking ``soundfile``.
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

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "decode_whisper.py"
_spec = importlib.util.spec_from_file_location("decode_whisper", _SCRIPT_PATH)
decode_whisper = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(decode_whisper)


# ---------------------------------------------------------------------------
# _align_token_logps_to_chars: the alignment convention documented in the
# module docstring.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """``decode(ids)`` = concatenation of each id's fixed textual
    contribution -- a deterministic stand-in for real (context-dependent)
    BPE decoding, sufficient to exercise the alignment algorithm's control
    flow (equal-split on multi-char tokens, carry-forward on zero-char
    tokens)."""

    def __init__(self, char_map):
        self.char_map = char_map

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.char_map.get(i, "") for i in ids)


def test_align_one_char_per_token():
    tok = _FakeTokenizer({101: "广", 102: "州", 103: "市"})
    text, logps = decode_whisper._align_token_logps_to_chars(
        tok, [101, 102, 103], [-0.1, -0.2, -0.3]
    )
    assert text == "广州市"
    assert logps == pytest.approx([-0.1, -0.2, -0.3])


def test_align_multi_char_token_splits_logp_equally():
    tok = _FakeTokenizer({1: "你好", 2: "吗"})
    text, logps = decode_whisper._align_token_logps_to_chars(tok, [1, 2], [-0.4, -0.2])
    assert text == "你好吗"
    # -0.4 split equally over 2 new chars ("你","好"), -0.2 over 1 ("吗")
    assert logps == pytest.approx([-0.2, -0.2, -0.2])
    assert sum(logps) == pytest.approx(-0.6)  # sum-preserving


def test_align_zero_char_token_carries_forward_to_next_char():
    tok = _FakeTokenizer({1: "", 2: "好"})
    text, logps = decode_whisper._align_token_logps_to_chars(tok, [1, 2], [-0.05, -0.2])
    assert text == "好"
    assert logps == pytest.approx([-0.25])  # -0.05 folded into the one real char


def test_align_trailing_zero_char_token_folds_into_last_char():
    tok = _FakeTokenizer({1: "市", 2: ""})  # e.g. a trailing eos-like token
    text, logps = decode_whisper._align_token_logps_to_chars(tok, [1, 2], [-0.3, -0.05])
    assert text == "市"
    assert logps == pytest.approx([-0.35])


def test_align_empty_result_when_all_tokens_contribute_nothing():
    tok = _FakeTokenizer({1: "", 2: ""})
    text, logps = decode_whisper._align_token_logps_to_chars(tok, [1, 2], [-0.1, -0.2])
    assert text == ""
    assert logps is None


# ---------------------------------------------------------------------------
# end-to-end: main() with fully mocked torch/transformers (no real deps)
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 1.0, rate: int = 16000) -> None:
    n_frames = int(duration_s * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


class _FakeTensor:
    """Stands in for a torch.Tensor: only ``.to``/``.tolist`` are used by
    decode_whisper.py's calling code."""

    def __init__(self, data):
        self.data = data

    def to(self, device=None, dtype=None):
        return self

    def tolist(self):
        return self.data


class _FakeSeqList(list):
    """A list that also exposes ``.shape`` like a batched tensor's leading
    dimension -- exactly what ``out.sequences.shape[0]`` needs."""

    @property
    def shape(self):
        return (len(self),)


def test_main_end_to_end_hyp_text_and_token_logps(tmp_path, monkeypatch):
    char_map = {101: "广", 102: "州", 103: "市", 2: ""}  # 2 = trailing eos-like token

    class _FakeTokenizerReal:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join(char_map.get(i, "") for i in ids)

    class _FakeProcessorOutput:
        def __init__(self, input_features):
            self.input_features = input_features

    class _FakeProcessor:
        tokenizer = _FakeTokenizerReal()

        def __call__(self, audio, sampling_rate, return_tensors):
            return _FakeProcessorOutput(_FakeTensor([[0.0]]))

        def get_decoder_prompt_ids(self, language, task):
            return []

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeGenOut:
        sequences = _FakeSeqList([_FakeTensor([101, 102, 103, 2])])
        scores = ("nonempty-sentinel",)  # must be truthy: real out.scores usability check
        sequences_scores = None

    class _FakeModel:
        dtype = None
        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def generate(self, *args, **kwargs):
            return _FakeGenOut()

        def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits=True):
            return _FakeSeqList([_FakeTensor([-0.1, -0.2, -0.3, -0.05])])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperForConditionalGeneration = _FakeModel
    fake_transformers.WhisperProcessor = _FakeProcessor

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        decode_whisper, "_load_audio",
        lambda wav_path: (np.zeros(1600, dtype=np.float32), 16000),
    )

    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\n", encoding="utf-8"
    )

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_whisper.main(
        ["--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["hyp_text"] == "广州市"
    assert record["nbest"][0]["text"] == "广州市"
    assert record["nbest"][0]["token_logps"] == pytest.approx([-0.1, -0.2, -0.35])
    assert record["nbest"][0]["logp"] == pytest.approx(-0.65)
    assert record["ref_text"] == "广州市"
    assert record["speaker_id"] == "SPK1"

    # Caller-shaped real-path check: the emitted record must validate
    # against the canonical schema the rest of asr-gate consumes.
    from asr_gate.io import validate_utterances

    validate_utterances([record])


def test_main_end_to_end_thchs30_corpus(tmp_path, monkeypatch):
    """Same wiring, but through the THCHS-30 corpus adapter -- covers the
    second corpus path (Aishell is covered by the test above)."""
    char_map = {5: "你", 6: "好"}

    class _FakeTokenizerReal:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join(char_map.get(i, "") for i in ids)

    class _FakeProcessorOutput:
        def __init__(self, input_features):
            self.input_features = input_features

    class _FakeProcessor:
        tokenizer = _FakeTokenizerReal()

        def __call__(self, audio, sampling_rate, return_tensors):
            return _FakeProcessorOutput(_FakeTensor([[0.0]]))

        def get_decoder_prompt_ids(self, language, task):
            return []

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeGenOut:
        sequences = _FakeSeqList([_FakeTensor([5, 6])])
        scores = ("nonempty-sentinel",)  # must be truthy: real out.scores usability check
        sequences_scores = None

    class _FakeModel:
        dtype = None
        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def generate(self, *args, **kwargs):
            return _FakeGenOut()

        def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits=True):
            return _FakeSeqList([_FakeTensor([-0.05, -0.1])])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperForConditionalGeneration = _FakeModel
    fake_transformers.WhisperProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        decode_whisper, "_load_audio",
        lambda wav_path: (np.zeros(800, dtype=np.float32), 16000),
    )

    _make_wav(tmp_path / "test" / "A11_101.wav")
    (tmp_path / "test" / "A11_101.wav.trn").write_text("你好\n", encoding="utf-8")

    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_whisper.main(
        ["--split", "test", "--corpus", "thchs30", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["hyp_text"] == "你好"
    assert record["speaker_id"] == "A11"
    assert record["ref_text"] == "你好"


# ---------------------------------------------------------------------------
# _generate_with_scores: transformers-version-adaptive regression coverage
# (2026-07-09 box smoke test: SMOKE_whisper usable=0/3 -- output_scores=/
# return_dict_in_generate= silently ignored as direct generate() kwargs by
# the installed transformers). Both scenarios must end with non-null
# token_logps in the final record; a permanently-broken model must abort
# loudly instead of emitting logp=None rows.
# ---------------------------------------------------------------------------


class _FakeGenerationConfig:
    """Stand-in for ``transformers.GenerationConfig`` -- plain mutable
    attribute bag, deepcopy-able, sufficient for
    ``_generate_with_scores``'s usage (construct/copy, set a few attrs)."""

    def __init__(self):
        self.output_scores = False
        self.return_dict_in_generate = False
        self.num_beams = 1
        self.num_return_sequences = 1
        self.max_new_tokens = None


def _fake_whisper_processor(char_map):
    class _FakeTokenizerReal:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join(char_map.get(i, "") for i in ids)

    class _FakeProcessorOutput:
        def __init__(self, input_features):
            self.input_features = input_features

    class _FakeProcessor:
        tokenizer = _FakeTokenizerReal()

        def __call__(self, audio, sampling_rate, return_tensors):
            return _FakeProcessorOutput(_FakeTensor([[0.0]]))

        def get_decoder_prompt_ids(self, language, task):
            return [(1, 50259)]

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    return _FakeProcessor


def _write_aishell_fixture(tmp_path):
    wav_dir = tmp_path / "wav" / "dev" / "SPK1"
    _make_wav(wav_dir / "UTT0001.wav")
    (tmp_path / "transcript").mkdir()
    (tmp_path / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "UTT0001 广州市\n", encoding="utf-8"
    )


def test_decode_whisper_old_api_direct_kwargs_honored(tmp_path, monkeypatch):
    """Old transformers stand-in: no ``generation_config=``/``language=``/
    ``task=`` params on ``generate()`` at all (predates them) -- passing
    them raises a genuine ``TypeError``, forcing the adaptive fallback all
    the way down to the legacy-direct-kwargs attempt, which this model DOES
    honor (mirrors the script's original, pre-fix behavior)."""
    char_map = {101: "广", 102: "州", 103: "市"}
    _FakeProcessor = _fake_whisper_processor(char_map)

    class _FakeGenOut:
        sequences = _FakeSeqList([_FakeTensor([101, 102, 103])])
        scores = ("nonempty-sentinel",)
        sequences_scores = None

    class _OldAPIModel:
        dtype = None
        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def generate(
            self, input_features, forced_decoder_ids=None, num_beams=1,
            num_return_sequences=1, output_scores=False,
            return_dict_in_generate=False, max_new_tokens=None,
        ):
            assert forced_decoder_ids is not None
            assert output_scores is True
            assert return_dict_in_generate is True
            return _FakeGenOut()

        def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits=True):
            return _FakeSeqList([_FakeTensor([-0.1, -0.2, -0.3])])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperForConditionalGeneration = _OldAPIModel
    fake_transformers.WhisperProcessor = _FakeProcessor
    fake_transformers.GenerationConfig = _FakeGenerationConfig
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        decode_whisper, "_load_audio",
        lambda wav_path: (np.zeros(1600, dtype=np.float32), 16000),
    )

    _write_aishell_fixture(tmp_path)
    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_whisper.main(
        ["--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["hyp_text"] == "广州市"
    assert record["nbest"][0]["token_logps"] is not None
    assert all(v is not None for v in record["nbest"][0]["token_logps"])
    assert record["nbest"][0]["logp"] is not None


def test_decode_whisper_new_api_scores_only_via_generation_config(tmp_path, monkeypatch):
    """New transformers stand-in: direct ``output_scores=``/
    ``return_dict_in_generate=`` kwargs are silently IGNORED (returns
    ``scores=None``, exactly the box's real failure mode); scores are only
    honored when set on the ``generation_config=`` object passed in."""
    char_map = {101: "广", 102: "州", 103: "市"}
    _FakeProcessor = _fake_whisper_processor(char_map)

    class _FakeGenOutNoScores:
        sequences = _FakeSeqList([_FakeTensor([101, 102, 103])])
        scores = None
        sequences_scores = None

    class _FakeGenOutWithScores:
        sequences = _FakeSeqList([_FakeTensor([101, 102, 103])])
        scores = ("nonempty-sentinel",)
        sequences_scores = None

    class _NewAPIModel:
        dtype = None
        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def generate(
            self, input_features, generation_config=None, language=None, task=None,
            forced_decoder_ids=None, num_beams=1, num_return_sequences=1,
            output_scores=False, return_dict_in_generate=False, max_new_tokens=None,
        ):
            # Real-world failure mode: direct kwargs are accepted
            # syntactically but ignored -- only generation_config sticks.
            if generation_config is not None and getattr(generation_config, "output_scores", False):
                return _FakeGenOutWithScores()
            return _FakeGenOutNoScores()

        def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits=True):
            return _FakeSeqList([_FakeTensor([-0.1, -0.2, -0.3])])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperForConditionalGeneration = _NewAPIModel
    fake_transformers.WhisperProcessor = _FakeProcessor
    fake_transformers.GenerationConfig = _FakeGenerationConfig
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        decode_whisper, "_load_audio",
        lambda wav_path: (np.zeros(1600, dtype=np.float32), 16000),
    )

    _write_aishell_fixture(tmp_path)
    out_path = tmp_path / "decode_out.jsonl"
    rc = decode_whisper.main(
        ["--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path), "--out", str(out_path)]
    )
    assert rc == 0

    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["hyp_text"] == "广州市"
    assert record["nbest"][0]["token_logps"] is not None
    assert all(v is not None for v in record["nbest"][0]["token_logps"])
    assert record["nbest"][0]["logp"] is not None


def test_decode_whisper_aborts_loudly_when_scores_permanently_unavailable(tmp_path, monkeypatch):
    """No adaptive attempt ever produces usable scores -- must FAIL LOUDLY
    (abort the run) rather than silently emit logp=None rows."""
    char_map = {101: "广", 102: "州", 103: "市"}
    _FakeProcessor = _fake_whisper_processor(char_map)

    class _FakeGenOutNoScores:
        sequences = _FakeSeqList([_FakeTensor([101, 102, 103])])
        scores = None
        sequences_scores = None

    class _AlwaysBrokenModel:
        dtype = None
        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def generate(self, input_features, **kwargs):
            return _FakeGenOutNoScores()

        def compute_transition_scores(self, *a, **k):
            return _FakeSeqList([_FakeTensor([-0.1, -0.2, -0.3])])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _FakeNoGrad()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperForConditionalGeneration = _AlwaysBrokenModel
    fake_transformers.WhisperProcessor = _FakeProcessor
    fake_transformers.GenerationConfig = _FakeGenerationConfig
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        decode_whisper, "_load_audio",
        lambda wav_path: (np.zeros(1600, dtype=np.float32), 16000),
    )

    _write_aishell_fixture(tmp_path)
    out_path = tmp_path / "decode_out.jsonl"
    with pytest.raises(SystemExit):
        decode_whisper.main(
            ["--split", "dev", "--corpus", "aishell", "--data-root", str(tmp_path), "--out", str(out_path)]
        )


def test_decode_whisper_help_runs_without_transformers():
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"], capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
