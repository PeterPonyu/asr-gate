"""Tests for orchestration/mix_musan.py's pure numpy mixing math (SNR
accuracy, determinism-by-seed) and file-level manifest/output behavior.

The pure-math functions (``mix_at_snr``/``fit_noise_length``) need no
audio library at all and are exercised directly on synthetic numpy arrays
-- this is the "exercise the exact shapes the caller passes" real-path
check (``mix_wav_list`` calls exactly these two functions, in exactly this
order, on real-length arrays). The end-to-end file-level tests need
``soundfile`` (a box-side/test-only dependency, not part of asr-gate's
pinned deps) and are skipped if it isn't installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "orchestration" / "mix_musan.py"
_spec = importlib.util.spec_from_file_location("mix_musan", _SCRIPT_PATH)
mix_musan = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mix_musan)


# ---------------------------------------------------------------------------
# mix_at_snr / achieved_snr_db round-trip accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 5.0, 15.0, 25.0])
def test_mix_at_snr_achieves_target_within_half_db(snr_db):
    rng = np.random.default_rng(42)
    n = 16000
    t = np.arange(n) / 16000.0
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.normal(0.0, 1.0, size=n)

    mixed = mix_musan.mix_at_snr(signal, noise, snr_db)
    scaled_noise = mixed - signal  # recover the noise component actually added
    achieved = mix_musan.achieved_snr_db(signal, scaled_noise)

    assert achieved == pytest.approx(snr_db, abs=0.5)


def test_mix_at_snr_silent_noise_is_noop():
    signal = np.array([0.1, -0.2, 0.3, -0.4])
    noise = np.zeros(4)
    mixed = mix_musan.mix_at_snr(signal, noise, 10.0)
    assert mixed == pytest.approx(signal, abs=1e-6)


def test_mix_at_snr_length_mismatch_raises():
    with pytest.raises(ValueError):
        mix_musan.mix_at_snr(np.zeros(4), np.zeros(5), 10.0)


def test_mix_at_snr_returns_float32():
    signal = np.ones(100, dtype=np.float64)
    noise = np.ones(100, dtype=np.float64) * 0.5
    mixed = mix_musan.mix_at_snr(signal, noise, 5.0)
    assert mixed.dtype == np.float32


# ---------------------------------------------------------------------------
# fit_noise_length: tiling + seeded crop, determinism
# ---------------------------------------------------------------------------


def test_fit_noise_length_tiles_short_noise_exact_multiple():
    # target_len == 3 * len(noise): tiling alone reaches target_len exactly,
    # so no crop step follows -- the result is deterministic without
    # needing to reason about the crop RNG draw.
    noise = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    out = mix_musan.fit_noise_length(noise, 9, rng)
    assert out.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_fit_noise_length_tiles_then_crops_short_noise():
    # target_len falls strictly between len(noise) and an exact tile
    # multiple: tiling overshoots, then the crop step trims to target_len
    # (a contiguous window of the tiled sequence, not necessarily starting
    # at index 0).
    noise = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    out = mix_musan.fit_noise_length(noise, 8, rng)
    assert len(out) == 8
    tiled = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    # the result must be SOME contiguous length-8 window of the tiled array
    windows = [tiled[i : i + 8] for i in range(len(tiled) - 8 + 1)]
    assert out.tolist() in windows


def test_fit_noise_length_crops_long_noise_deterministically_by_seed():
    noise = np.arange(1000.0)
    out_a = mix_musan.fit_noise_length(noise, 100, np.random.default_rng(7))
    out_b = mix_musan.fit_noise_length(noise, 100, np.random.default_rng(7))
    assert np.array_equal(out_a, out_b)


def test_fit_noise_length_different_seeds_usually_differ():
    noise = np.arange(10000.0)
    out_a = mix_musan.fit_noise_length(noise, 100, np.random.default_rng(1))
    out_b = mix_musan.fit_noise_length(noise, 100, np.random.default_rng(2))
    assert not np.array_equal(out_a, out_b)


def test_fit_noise_length_empty_noise_returns_zeros():
    out = mix_musan.fit_noise_length(np.array([]), 5, np.random.default_rng(0))
    assert np.array_equal(out, np.zeros(5))


def test_fit_noise_length_exact_length_passthrough():
    noise = np.array([1.0, 2.0, 3.0])
    out = mix_musan.fit_noise_length(noise, 3, np.random.default_rng(0))
    assert np.array_equal(out, noise)


# ---------------------------------------------------------------------------
# resample_linear
# ---------------------------------------------------------------------------


def test_resample_linear_noop_when_rates_match():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = mix_musan.resample_linear(data, 16000, 16000)
    assert np.array_equal(out, data)


def test_resample_linear_changes_length():
    data = np.linspace(-1, 1, 8000, dtype=np.float32)
    out = mix_musan.resample_linear(data, 8000, 16000)
    assert len(out) == pytest.approx(16000, abs=2)


# ---------------------------------------------------------------------------
# discover_musan_noise
# ---------------------------------------------------------------------------


def test_discover_musan_noise_sorted(tmp_path):
    (tmp_path / "noise" / "free-sound").mkdir(parents=True)
    (tmp_path / "noise" / "free-sound" / "b.wav").write_bytes(b"")
    (tmp_path / "noise" / "free-sound" / "a.wav").write_bytes(b"")

    files = mix_musan.discover_musan_noise(tmp_path, "noise")

    assert [f.name for f in files] == ["a.wav", "b.wav"]


def test_discover_musan_noise_missing_category_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mix_musan.discover_musan_noise(tmp_path, "noise")


def test_discover_musan_noise_empty_category_raises(tmp_path):
    (tmp_path / "noise").mkdir()
    with pytest.raises(FileNotFoundError):
        mix_musan.discover_musan_noise(tmp_path, "noise")


# ---------------------------------------------------------------------------
# discover_noise_files: the source-agnostic entry point mix_wav_list calls
# -- category="noise" (default) dispatches to discover_musan_noise;
# category=None is the NEUTRAL mode for non-MUSAN-shaped sources (ESC-50,
# EXPANSION-AMENDMENT-2026-07-09.md's disclosed substitution).
# ---------------------------------------------------------------------------


def test_discover_noise_files_default_dispatches_to_musan_layout(tmp_path):
    (tmp_path / "noise" / "free-sound").mkdir(parents=True)
    (tmp_path / "noise" / "free-sound" / "b.wav").write_bytes(b"")
    (tmp_path / "noise" / "free-sound" / "a.wav").write_bytes(b"")

    files = mix_musan.discover_noise_files(tmp_path)

    assert [f.name for f in files] == ["a.wav", "b.wav"]


def test_discover_noise_files_neutral_mode_searches_flat_tree(tmp_path):
    # ESC-50-like layout: no noise/music/speech split, just category folders
    # (or a flat pile) of .wav files directly under the root.
    (tmp_path / "1-100032-A-0").mkdir(parents=True)
    (tmp_path / "1-100032-A-0" / "dog.wav").write_bytes(b"")
    (tmp_path / "2-100038-A-14").mkdir(parents=True)
    (tmp_path / "2-100038-A-14" / "chirping_birds.wav").write_bytes(b"")

    files = mix_musan.discover_noise_files(tmp_path, category=None)

    assert {f.name for f in files} == {"dog.wav", "chirping_birds.wav"}


def test_discover_noise_files_neutral_mode_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mix_musan.discover_noise_files(tmp_path / "does_not_exist", category=None)


def test_discover_noise_files_neutral_mode_empty_dir_raises(tmp_path):
    (tmp_path / "esc50").mkdir()
    with pytest.raises(FileNotFoundError):
        mix_musan.discover_noise_files(tmp_path / "esc50", category=None)


# ---------------------------------------------------------------------------
# CLI --help works without soundfile at all (argparse-only path).
# ---------------------------------------------------------------------------


def test_mix_musan_help_runs_without_soundfile():
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"], capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# end-to-end mix_wav_list: real WAV files on disk via soundfile.
# ---------------------------------------------------------------------------

soundfile = pytest.importorskip("soundfile")


def _write_pcm_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), samples.astype(np.float32), sr, subtype="PCM_16")


def test_mix_wav_list_end_to_end_writes_manifest_and_wavs(tmp_path):
    n = 16000
    t = np.arange(n) / 16000.0
    sig1 = 0.4 * np.sin(2 * np.pi * 300 * t)
    sig2 = 0.4 * np.sin(2 * np.pi * 600 * t)
    _write_pcm_wav(tmp_path / "u1.wav", sig1)
    _write_pcm_wav(tmp_path / "u2.wav", sig2)

    noise = np.random.default_rng(0).normal(0, 0.3, size=n).astype(np.float32)
    _write_pcm_wav(tmp_path / "musan" / "noise" / "n1.wav", noise)

    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(
        f"u1 {tmp_path / 'u1.wav'}\nu2 {tmp_path / 'u2.wav'}\n", encoding="utf-8"
    )

    out_dir = tmp_path / "mixed"
    result = mix_musan.mix_wav_list(
        wav_list_path=wav_list,
        musan_dir=tmp_path / "musan",
        snr_db=10.0,
        out_dir=out_dir,
        seed=0,
    )

    assert result == {"n_written": 2, "n_skipped": 0}
    assert (out_dir / "u1.wav").exists()
    assert (out_dir / "u2.wav").exists()

    manifest_lines = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 2
    rec = json.loads(manifest_lines[0])
    assert rec["utt_id"] == "u1"
    assert rec["snr"] == 10.0
    assert rec["seed"] == 0
    assert "noise_file" in rec


def test_mix_wav_list_deterministic_by_seed(tmp_path):
    n = 8000
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000.0)).astype(np.float32)
    _write_pcm_wav(tmp_path / "u1.wav", sig)
    noise1 = np.random.default_rng(1).normal(0, 0.3, size=20000).astype(np.float32)
    noise2 = np.random.default_rng(2).normal(0, 0.3, size=20000).astype(np.float32)
    _write_pcm_wav(tmp_path / "musan" / "noise" / "n1.wav", noise1)
    _write_pcm_wav(tmp_path / "musan" / "noise" / "n2.wav", noise2)

    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(f"u1 {tmp_path / 'u1.wav'}\n", encoding="utf-8")

    out_a = tmp_path / "mixed_a"
    out_b = tmp_path / "mixed_b"
    mix_musan.mix_wav_list(wav_list, tmp_path / "musan", 10.0, out_a, seed=5)
    mix_musan.mix_wav_list(wav_list, tmp_path / "musan", 10.0, out_b, seed=5)

    a = soundfile.read(str(out_a / "u1.wav"))[0]
    b = soundfile.read(str(out_b / "u1.wav"))[0]
    assert np.array_equal(a, b)


def test_mix_wav_list_neutral_noise_dir_mode_end_to_end(tmp_path):
    """category=None: mixes against a flat ESC-50-shaped noise tree with no
    MUSAN noise/music/speech subdirectory."""
    n = 16000
    t = np.arange(n) / 16000.0
    sig = 0.4 * np.sin(2 * np.pi * 300 * t)
    _write_pcm_wav(tmp_path / "u1.wav", sig)

    noise = np.random.default_rng(0).normal(0, 0.3, size=n).astype(np.float32)
    _write_pcm_wav(tmp_path / "esc50" / "1-cat" / "n1.wav", noise)

    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(f"u1 {tmp_path / 'u1.wav'}\n", encoding="utf-8")

    out_dir = tmp_path / "mixed"
    result = mix_musan.mix_wav_list(
        wav_list_path=wav_list,
        musan_dir=tmp_path / "esc50",
        snr_db=10.0,
        out_dir=out_dir,
        seed=0,
        category=None,
    )

    assert result == {"n_written": 1, "n_skipped": 0}
    assert (out_dir / "u1.wav").exists()


def test_mix_musan_cli_noise_dir_mode_flag(tmp_path):
    n = 8000
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000.0)).astype(np.float32)
    _write_pcm_wav(tmp_path / "u1.wav", sig)
    noise = np.random.default_rng(0).normal(0, 0.3, size=n).astype(np.float32)
    _write_pcm_wav(tmp_path / "esc50" / "1-cat" / "n1.wav", noise)

    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(f"u1 {tmp_path / 'u1.wav'}\n", encoding="utf-8")

    out_dir = tmp_path / "mixed"
    import subprocess

    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT_PATH),
            "--wav-list", str(wav_list),
            "--musan-dir", str(tmp_path / "esc50"),
            "--noise-dir-mode",
            "--snr-db", "10",
            "--out-dir", str(out_dir),
            "--seed", "0",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "u1.wav").exists()


def test_mix_wav_list_skips_unreadable_wav_and_counts_it(tmp_path, capsys):
    (tmp_path / "u1.wav").write_bytes(b"not a real wav")
    _write_pcm_wav(tmp_path / "musan" / "noise" / "n1.wav", np.ones(16000, dtype=np.float32))

    wav_list = tmp_path / "wavs.txt"
    wav_list.write_text(f"u1 {tmp_path / 'u1.wav'}\n", encoding="utf-8")

    result = mix_musan.mix_wav_list(
        wav_list, tmp_path / "musan", 10.0, tmp_path / "mixed", seed=0
    )

    assert result == {"n_written": 0, "n_skipped": 1}
    assert "warning" in capsys.readouterr().err
