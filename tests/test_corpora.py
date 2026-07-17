"""Tests for asr_gate.corpora: Aishell-1 + THCHS-30 layout discovery and
transcript parsing. No network/GPU -- fixtures build tiny fake corpus trees
on disk (tmp_path)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from asr_gate import corpora


def _write_wav(path: Path, n_frames: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * n_frames)


# ---------------------------------------------------------------------------
# Aishell-1
# ---------------------------------------------------------------------------


def test_discover_aishell_layout(tmp_path):
    root = tmp_path
    _write_wav(root / "wav" / "dev" / "S0002" / "BAC009S0002W0122.wav")
    _write_wav(root / "wav" / "dev" / "S0002" / "BAC009S0002W0123.wav")
    (root / "transcript").mkdir()
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "BAC009S0002W0122 广 州 市 房 地 产\n", encoding="utf-8"
    )

    entries = corpora.discover_aishell(root, "dev")

    assert len(entries) == 2
    by_id = {e.utt_id: e for e in entries}
    assert by_id["BAC009S0002W0122"].speaker_id == "S0002"
    assert by_id["BAC009S0002W0122"].ref_text == "广州市房地产"
    assert by_id["BAC009S0002W0123"].ref_text is None
    assert isinstance(by_id["BAC009S0002W0122"].wav_path, Path)


def test_discover_aishell_missing_wav_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpora.discover_aishell(tmp_path, "dev")


def test_discover_aishell_respects_limit(tmp_path):
    root = tmp_path
    for i in range(5):
        _write_wav(root / "wav" / "test" / "S0001" / f"UTT{i}.wav")
    (root / "transcript").mkdir()
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text("", encoding="utf-8")

    entries = corpora.discover_aishell(root, "test", limit=3)

    assert len(entries) == 3


def test_discover_aishell_no_transcript_file_degrades_to_all_missing_refs(tmp_path):
    root = tmp_path
    _write_wav(root / "wav" / "test" / "S0001" / "UTT0.wav")

    entries = corpora.discover_aishell(root, "test")

    assert len(entries) == 1
    assert entries[0].ref_text is None


# ---------------------------------------------------------------------------
# THCHS-30
# ---------------------------------------------------------------------------


def test_load_thchs30_transcript_strips_spacing_and_keeps_first_line_only():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        trn = Path(d) / "A11_101.wav.trn"
        trn.write_text("绿 是 阳春 烟 景\nlv4 shi4 yang2 chun1\nl v4 sh ...\n", encoding="utf-8")
        assert corpora.load_thchs30_transcript(trn) == "绿是阳春烟景"


def test_load_thchs30_transcript_missing_file_returns_none(tmp_path):
    assert corpora.load_thchs30_transcript(tmp_path / "nope.wav.trn") is None


def test_load_thchs30_transcript_empty_first_line_returns_none(tmp_path):
    trn = tmp_path / "empty.wav.trn"
    trn.write_text("\nlv4\n", encoding="utf-8")
    assert corpora.load_thchs30_transcript(trn) is None


def test_thchs_speaker_id_extraction():
    assert corpora._thchs_speaker_id("A11_101") == "A11"
    assert corpora._thchs_speaker_id("D8_2033") == "D8"
    assert corpora._thchs_speaker_id("weird") == "weird"


def test_discover_thchs30_uses_split_dir_and_data_dir_trn(tmp_path):
    root = tmp_path
    _write_wav(root / "data" / "A11_101.wav")
    (root / "data" / "A11_101.wav.trn").write_text("绿 是 阳春\n", encoding="utf-8")
    _write_wav(root / "test" / "A11_101.wav")

    entries = corpora.discover_thchs30(root, "test")

    assert len(entries) == 1
    e = entries[0]
    assert e.utt_id == "A11_101"
    assert e.speaker_id == "A11"
    assert e.ref_text == "绿是阳春"
    assert e.wav_path == root / "test" / "A11_101.wav"


def test_discover_thchs30_falls_back_to_flat_data_dir(tmp_path, capsys):
    root = tmp_path
    _write_wav(root / "data" / "B8_1.wav")
    (root / "data" / "B8_1.wav.trn").write_text("你好\n", encoding="utf-8")

    entries = corpora.discover_thchs30(root, "test")

    assert len(entries) == 1
    assert entries[0].utt_id == "B8_1"
    assert "warning" in capsys.readouterr().err


def test_discover_thchs30_missing_everything_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpora.discover_thchs30(tmp_path, "test")


def test_discover_thchs30_trn_fallback_colocated_with_split_dir(tmp_path):
    root = tmp_path
    _write_wav(root / "test" / "C1_5.wav")
    (root / "test" / "C1_5.wav.trn").write_text("再见\n", encoding="utf-8")

    entries = corpora.discover_thchs30(root, "test")

    assert entries[0].ref_text == "再见"


def test_discover_thchs30_respects_limit(tmp_path):
    root = tmp_path
    for i in range(4):
        _write_wav(root / "test" / f"A1_{i}.wav")

    entries = corpora.discover_thchs30(root, "test", limit=2)

    assert len(entries) == 2


# ---------------------------------------------------------------------------
# aidatatang_200zh (openslr-62)
# ---------------------------------------------------------------------------


def test_discover_aidatatang_layout(tmp_path):
    root = tmp_path
    _write_wav(root / "corpus" / "test" / "G0002" / "T0055G0002S0001.wav")
    _write_wav(root / "corpus" / "test" / "G0002" / "T0055G0002S0002.wav")
    (root / "transcript").mkdir()
    (root / "transcript" / "aidatatang_200_zh_transcript.txt").write_text(
        "T0055G0002S0001 广 州 市 房 地 产\n", encoding="utf-8"
    )

    entries = corpora.discover_aidatatang(root, "test")

    assert len(entries) == 2
    by_id = {e.utt_id: e for e in entries}
    assert by_id["T0055G0002S0001"].speaker_id == "G0002"
    assert by_id["T0055G0002S0001"].ref_text == "广州市房地产"
    assert by_id["T0055G0002S0002"].ref_text is None  # not in transcript


def test_discover_aidatatang_missing_wav_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpora.discover_aidatatang(tmp_path, "test")


def test_discover_aidatatang_respects_limit(tmp_path):
    root = tmp_path
    for i in range(5):
        _write_wav(root / "corpus" / "dev" / "G0001" / f"T0055G0001S000{i}.wav")
    entries = corpora.discover_aidatatang(root, "dev", limit=3)
    assert len(entries) == 3


def test_discover_aidatatang_no_transcript_degrades_to_missing_refs(tmp_path):
    root = tmp_path
    _write_wav(root / "corpus" / "test" / "G0009" / "T0055G0009S0001.wav")
    entries = corpora.discover_aidatatang(root, "test")
    assert len(entries) == 1
    assert entries[0].ref_text is None


# ---------------------------------------------------------------------------
# MAGICDATA (openslr-68)
# ---------------------------------------------------------------------------


def test_load_magicdata_trans_skips_header_strips_ext_and_spaces(tmp_path):
    trans = tmp_path / "TRANS.txt"
    trans.write_text(
        "UtteranceID\tSpeakerID\tTranscription\n"
        "16_4013_20170819121429.wav\t16_4013\t然后 呢 到 时候\n",
        encoding="utf-8",
    )
    m = corpora.load_magicdata_trans(trans)
    assert "16_4013_20170819121429" in m  # .wav stripped -> matches wav stem
    spk, ref = m["16_4013_20170819121429"]
    assert spk == "16_4013"
    assert ref == "然后呢到时候"  # inter-word spaces stripped


def test_load_magicdata_trans_missing_file_returns_empty(tmp_path):
    assert corpora.load_magicdata_trans(tmp_path / "nope.txt") == {}


def test_discover_magicdata_layout_uses_trans_speaker_and_ref(tmp_path):
    root = tmp_path
    _write_wav(root / "test" / "14_5223" / "14_5223_20170923171105.wav")
    (root / "test" / "TRANS.txt").write_text(
        "UtteranceID\tSpeakerID\tTranscription\n"
        "14_5223_20170923171105.wav\t14_5223\t你 好 世界\n",
        encoding="utf-8",
    )

    entries = corpora.discover_magicdata(root, "test")

    assert len(entries) == 1
    e = entries[0]
    assert e.utt_id == "14_5223_20170923171105"
    assert e.speaker_id == "14_5223"  # authoritative from TRANS.txt
    assert e.ref_text == "你好世界"
    assert e.wav_path == root / "test" / "14_5223" / "14_5223_20170923171105.wav"


def test_discover_magicdata_falls_back_to_parent_dir_speaker_when_absent_from_trans(tmp_path):
    root = tmp_path
    _write_wav(root / "dev" / "22_9911" / "22_9911_xyz.wav")
    (root / "dev" / "TRANS.txt").write_text(
        "UtteranceID\tSpeakerID\tTranscription\n", encoding="utf-8"
    )  # header only -- utt not listed

    entries = corpora.discover_magicdata(root, "dev")

    assert len(entries) == 1
    assert entries[0].speaker_id == "22_9911"  # fallback = parent dir name
    assert entries[0].ref_text is None


def test_discover_magicdata_missing_split_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpora.discover_magicdata(tmp_path, "test")


def test_discover_magicdata_respects_limit(tmp_path):
    root = tmp_path
    for i in range(4):
        _write_wav(root / "test" / "33_0001" / f"33_0001_{i}.wav")
    entries = corpora.discover_magicdata(root, "test", limit=2)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_discover_corpus_dispatches_to_aishell(tmp_path):
    root = tmp_path
    _write_wav(root / "wav" / "dev" / "S0001" / "U1.wav")
    (root / "transcript").mkdir()
    (root / "transcript" / "aishell_transcript_v0.8.txt").write_text("", encoding="utf-8")

    entries = corpora.discover_corpus("aishell", root, "dev")

    assert len(entries) == 1
    assert entries[0].speaker_id == "S0001"


def test_discover_corpus_dispatches_to_thchs30(tmp_path):
    root = tmp_path
    _write_wav(root / "test" / "A1_1.wav")

    entries = corpora.discover_corpus("thchs30", root, "test")

    assert len(entries) == 1
    assert entries[0].speaker_id == "A1"


def test_discover_corpus_dispatches_to_aidatatang(tmp_path):
    root = tmp_path
    _write_wav(root / "corpus" / "test" / "G0002" / "T0055G0002S0001.wav")
    entries = corpora.discover_corpus("aidatatang", root, "test")
    assert len(entries) == 1
    assert entries[0].speaker_id == "G0002"


def test_discover_corpus_dispatches_to_magicdata(tmp_path):
    root = tmp_path
    _write_wav(root / "test" / "14_5223" / "14_5223_1.wav")
    (root / "test" / "TRANS.txt").write_text(
        "UtteranceID\tSpeakerID\tTranscription\n"
        "14_5223_1.wav\t14_5223\t你 好\n",
        encoding="utf-8",
    )
    entries = corpora.discover_corpus("magicdata", root, "test")
    assert len(entries) == 1
    assert entries[0].speaker_id == "14_5223"
    assert entries[0].ref_text == "你好"


def test_discover_corpus_unknown_raises():
    with pytest.raises(ValueError):
        corpora.discover_corpus("bogus", Path("."), "dev")
