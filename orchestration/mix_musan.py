#!/usr/bin/env python3
"""MUSAN noise mixing: additive-noise augmentation of a wav list at a target
SNR, seeded and deterministic, for the expansion's noise-stratified arm
(EXPANSION-PLAN-2026-07-09.md §2.1's D2/MUSAN axis, reinstating SNR
Mondrian strata per design §2.4). Pure numpy + soundfile (no funasr/torch
dependency -- runs on any box, or locally for testing).

Usage::

    mix_musan.py --wav-list wavs.txt --musan-dir /root/autodl-tmp/musan \\
        --snr-db 15 --out-dir mixed_15db --seed 0

``wavs.txt``: ``utt_id wav_path`` per line (whitespace-separated, the same
convention ``orchestration/decode_paraformer.py`` uses for its transcript
file).

Output: ``{out_dir}/{utt_id}.wav`` (16kHz mono PCM16) for every input
utterance, plus a manifest JSONL (default ``{out_dir}/manifest.jsonl``) with
one ``{"utt_id", "snr", "noise_file", "seed"}`` record per mixed utterance.

Determinism: utterances are processed in SORTED ``utt_id`` order; a single
``numpy.random.default_rng(seed)`` stream drives BOTH noise-file selection
and crop-offset choice, in that fixed order -- so the same ``--seed`` always
produces the same mix for the same wav list + musan dir contents (verified
by ``tests/test_mix_musan.py::test_mix_wav_list_deterministic_by_seed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "mix_at_snr",
    "achieved_snr_db",
    "fit_noise_length",
    "discover_musan_noise",
    "discover_noise_files",
    "resample_linear",
    "load_wav_mono",
    "mix_wav_list",
]


# ---------------------------------------------------------------------------
# Pure-numpy mixing math -- no soundfile/file I/O, unit-testable directly
# (this IS the "exercise the exact shapes the caller passes" real-path
# surface: :func:`mix_wav_list` calls exactly these two functions in
# exactly this order on real-length arrays).
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` (assumed already the same length as ``signal``, via
    :func:`fit_noise_length`) so the mixture achieves
    ``20*log10(rms(signal) / rms(scaled_noise)) == snr_db``, then return
    ``signal + scaled_noise`` (float32, NOT clipped -- callers clip
    themselves before writing 16-bit PCM). RMS is computed over the RAW
    segment (no VAD/silence trimming -- a documented simplification:
    MUSAN's ``noise``/``music`` files are largely continuous, so
    silence-weighted RMS bias is expected to be small; not verified against
    real MUSAN content at build time, since the box download hadn't
    completed). If ``noise`` is silent (rms == 0), returns ``signal``
    unchanged (adding scaled silence is a no-op; avoids a divide-by-zero)."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if signal.shape != noise.shape:
        raise ValueError(
            f"mix_at_snr: signal/noise length mismatch ({signal.shape} vs {noise.shape})"
        )
    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    if noise_rms == 0.0:
        return signal.astype(np.float32)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    return (signal + noise * scale).astype(np.float32)


def achieved_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """The SNR a given (unscaled) ``signal``/``noise`` pair already
    represents -- the inverse quantity :func:`mix_at_snr` targets; used by
    tests to verify its scaling is accurate."""
    sig_rms = _rms(np.asarray(signal, dtype=np.float64))
    noise_rms = _rms(np.asarray(noise, dtype=np.float64))
    if noise_rms == 0.0:
        return float("inf")
    return float(20.0 * np.log10(sig_rms / noise_rms))


def fit_noise_length(
    noise: np.ndarray, target_len: int, rng: np.random.Generator
) -> np.ndarray:
    """Loop (tile) ``noise`` if shorter than ``target_len``; take a random
    contiguous crop of exactly ``target_len`` samples if longer (crop
    offset drawn from ``rng``, so determinism is entirely a function of the
    RNG's state at call time -- callers control reproducibility by seeding
    ``rng`` once and reusing it in a fixed processing order, see
    :func:`mix_wav_list`). Empty (or non-positive ``target_len``) ``noise``
    returns ``target_len`` zeros."""
    noise = np.asarray(noise, dtype=np.float64)
    if noise.size == 0 or target_len <= 0:
        return np.zeros(max(target_len, 0), dtype=np.float64)
    if len(noise) < target_len:
        reps = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > target_len:
        max_start = len(noise) - target_len
        start = int(rng.integers(0, max_start + 1))
        noise = noise[start : start + target_len]
    return noise[:target_len]


# ---------------------------------------------------------------------------
# File I/O -- soundfile is lazily imported so this module (and --help) load
# without it installed, matching decode_paraformer.py's box-side-only
# dependency convention.
# ---------------------------------------------------------------------------


def resample_linear(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Plain linear-interpolation resample (numpy only, no extra
    dependency). Approximate -- fine for the expected common case (MUSAN
    and Aishell/THCHS-30 are both natively 16kHz, so this is a rare
    fallback, not the mixing pipeline's normal path); documented, not a
    claim of high-fidelity resampling."""
    data = np.asarray(data, dtype=np.float32)
    if sr == target_sr or len(data) == 0:
        return data
    duration = len(data) / sr
    n_target = max(int(round(duration * target_sr)), 1)
    x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


def load_wav_mono(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Mono float32 waveform at ``target_sr`` via ``soundfile`` (lazy
    import -- box-side/test-only dependency, not part of asr-gate's pinned
    deps)."""
    import soundfile as sf  # type: ignore  # lazy

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample_linear(data, sr, target_sr)
    return data


def discover_musan_noise(musan_dir: Path, category: str = "noise") -> List[Path]:
    """Sorted list of ``.wav`` files under ``{musan_dir}/{category}/**``
    (MUSAN's standard layout: ``noise/``, ``music/``, ``speech/`` top-level
    categories, each with nested subdirectories per source)."""
    cat_dir = Path(musan_dir) / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"discover_musan_noise: {cat_dir} not found")
    files = sorted(cat_dir.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"discover_musan_noise: no .wav files found under {cat_dir}")
    return files


def discover_noise_files(noise_dir: Path, category: Optional[str] = "noise") -> List[Path]:
    """Sorted list of ``.wav`` files to use as additive noise -- the
    source-agnostic entry point :func:`mix_wav_list` actually calls.

    If ``category`` is given (the default, MUSAN mode), dispatches to
    :func:`discover_musan_noise` (``{noise_dir}/{category}/**``). If
    ``category`` is ``None`` (the NEUTRAL ``--noise-dir-mode``, added
    2026-07-09 for the ESC-50 disclosed substitution --
    ``EXPANSION-AMENDMENT-2026-07-09.md``: ESC-50 has no MUSAN-style
    ``noise``/``music``/``speech`` top-level split), every ``.wav`` file
    under ``noise_dir`` is searched directly, recursively -- works for
    ESC-50's flat-or-per-category layout as much as any other plain
    wav-file tree, without assuming which."""
    if category is not None:
        return discover_musan_noise(noise_dir, category)
    noise_dir = Path(noise_dir)
    if not noise_dir.exists():
        raise FileNotFoundError(f"discover_noise_files: {noise_dir} not found")
    files = sorted(noise_dir.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"discover_noise_files: no .wav files found under {noise_dir}")
    return files


def _load_wav_list(path: Path) -> List[Tuple[str, Path]]:
    """``utt_id wav_path`` per line (same convention as
    ``decode_paraformer.py``'s transcript-file parsing), returned SORTED by
    ``utt_id`` (the determinism contract -- see module docstring)."""
    entries: List[Tuple[str, Path]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, wav_path = parts
            entries.append((utt_id, Path(wav_path)))
    entries.sort(key=lambda e: e[0])
    return entries


def mix_wav_list(
    wav_list_path: Path,
    musan_dir: Path,
    snr_db: float,
    out_dir: Path,
    seed: int = 0,
    category: Optional[str] = "noise",
    limit: Optional[int] = None,
    manifest_out: Optional[Path] = None,
) -> Dict[str, int]:
    """Mix every utterance in ``wav_list_path`` with a seeded, randomly
    chosen noise clip at ``snr_db``, writing ``{out_dir}/{utt_id}.wav`` plus
    a manifest JSONL. One bad utterance (unreadable wav, etc.) is
    skipped-and-counted, never fatal to the whole run -- matches
    ``decode_paraformer.py``'s per-utterance robustness convention.

    ``category``: MUSAN-style top-level split (default ``"noise"``) --
    passed through to :func:`discover_noise_files`, which dispatches to
    :func:`discover_musan_noise`. Pass ``category=None`` for the NEUTRAL
    ``--noise-dir-mode`` (source-agnostic: no ``noise``/``music``/``speech``
    subdirectory assumed under ``musan_dir`` -- see
    :func:`discover_noise_files`'s docstring for the ESC-50 motivation).

    Returns ``{"n_written": int, "n_skipped": int}``."""
    import soundfile as sf  # type: ignore  # lazy

    entries = _load_wav_list(wav_list_path)
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise SystemExit(f"mix_musan: no entries in {wav_list_path}")

    noise_files = discover_noise_files(musan_dir, category)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_out) if manifest_out else out_dir / "manifest.jsonl"

    rng = np.random.default_rng(seed)
    n_written = 0
    n_skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for utt_id, wav_path in entries:
            try:
                signal = load_wav_mono(wav_path)
                noise_idx = int(rng.integers(0, len(noise_files)))
                noise_path = noise_files[noise_idx]
                noise_raw = load_wav_mono(noise_path)
                noise = fit_noise_length(noise_raw, len(signal), rng)
                mixed = mix_at_snr(signal, noise, snr_db)
            except Exception as exc:  # noqa: BLE001 - one bad utterance must not kill the run
                print(
                    f"warning: mix_musan: skipping {utt_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue
            mixed = np.clip(mixed, -1.0, 1.0)
            out_wav = out_dir / f"{utt_id}.wav"
            sf.write(str(out_wav), mixed, 16000, subtype="PCM_16")
            mf.write(
                json.dumps(
                    {"utt_id": utt_id, "snr": snr_db, "noise_file": str(noise_path), "seed": seed},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"mix_musan: n_written={n_written} n_skipped={n_skipped} snr_db={snr_db} -> {out_dir}")
    return {"n_written": n_written, "n_skipped": n_skipped}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mix_musan.py",
        description="Additive MUSAN noise mixing at a target SNR, seeded and deterministic.",
    )
    parser.add_argument("--wav-list", required=True, help="utt_id wav_path per line")
    parser.add_argument(
        "--musan-dir", required=True,
        help="noise source root; despite the flag name this is source-agnostic -- see "
             "--noise-dir-mode for non-MUSAN sources (e.g. ESC-50)",
    )
    parser.add_argument("--musan-category", default="noise", choices=["noise", "music", "speech"])
    parser.add_argument(
        "--noise-dir-mode", action="store_true",
        help="treat --musan-dir as a NEUTRAL flat/arbitrary wav-file tree (no MUSAN "
             "noise/music/speech subdirectory assumed) -- e.g. for the ESC-50 disclosed "
             "substitution (EXPANSION-AMENDMENT-2026-07-09.md); overrides --musan-category",
    )
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    mix_wav_list(
        wav_list_path=Path(args.wav_list),
        musan_dir=Path(args.musan_dir),
        snr_db=args.snr_db,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        category=None if args.noise_dir_mode else args.musan_category,
        limit=args.limit,
        manifest_out=Path(args.manifest_out) if args.manifest_out else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
