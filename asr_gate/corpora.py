"""Corpus adapters: map an on-disk corpus layout to
``(utt_id, speaker_id, wav_path, ref_text)`` records, plus reference-
transcript loading. This is the ONLY corpus-layout-aware code shared across
decode scripts -- ``orchestration/decode_whisper.py`` and
``orchestration/decode_conformer_ms.py`` both import :func:`discover_corpus`
from here rather than duplicating layout assumptions.
(``orchestration/decode_paraformer.py`` predates this module and inlines its
own Aishell-only version of the same logic; it is left untouched to avoid
destabilizing its already real-box-verified behavior -- see its module
docstring.)

Layout assumptions below are UNVERIFIED against a real on-box extraction
until Phase 0 confirms them (see each function's docstring); every
discoverer degrades loudly (a printed warning, never a silent wrong answer)
on an unrecognized shape where a reasonable fallback exists, and raises
``FileNotFoundError`` with an actionable message otherwise -- matching
``orchestration/decode_paraformer.py``'s own convention.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

__all__ = [
    "CorpusUtterance",
    "load_aishell_transcripts",
    "discover_aishell",
    "load_thchs30_transcript",
    "discover_thchs30",
    "discover_aidatatang",
    "load_magicdata_trans",
    "discover_magicdata",
    "discover_corpus",
    "CORPUS_DISCOVERERS",
]


class CorpusUtterance(NamedTuple):
    """One decodable utterance: enough to drive any ``decode_*.py`` script."""

    utt_id: str
    speaker_id: str
    wav_path: Path
    ref_text: Optional[str]


# ---------------------------------------------------------------------------
# Aishell-1 (openslr-33)
# ---------------------------------------------------------------------------

_AISHELL_SPEAKER_RE = re.compile(r"(S\d{4})")


def load_aishell_transcripts(transcript_file: Path) -> Dict[str, str]:
    """``{utt_id: ref_text}`` from Aishell's
    ``transcript/aishell_transcript_v0.8.txt`` (``"utt_id ref_text"`` per
    line, ``ref_text`` space-segmented by character -- spaces stripped
    here, matching ``orchestration/decode_paraformer.py``'s identical
    helper)."""
    refs: Dict[str, str] = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, ref_text = parts
            refs[utt_id] = ref_text.replace(" ", "")
    return refs


def discover_aishell(
    data_root: Path, split: str, limit: Optional[int] = None
) -> List[CorpusUtterance]:
    """Aishell-1 (openslr-33 layout, per design §3.1/§7 Phase 0; VERIFY
    against the actual box before trusting)::

        {data_root}/wav/{split}/{speaker_id}/{utt_id}.wav
        {data_root}/transcript/aishell_transcript_v0.8.txt

    Speaker id is the wav's parent directory name (the authoritative
    Aishell-1 convention -- more reliable than regex-parsing ``utt_id``,
    unlike ``asr_gate.io``'s ``_aishell_speaker_id``, which only has
    ``utt_id`` to work with when adapting a standalone decode dump with no
    directory structure)."""
    data_root = Path(data_root)
    wav_dir = data_root / "wav" / split
    transcript_file = data_root / "transcript" / "aishell_transcript_v0.8.txt"
    if not wav_dir.exists():
        raise FileNotFoundError(f"discover_aishell: wav dir not found: {wav_dir}")
    refs = load_aishell_transcripts(transcript_file) if transcript_file.exists() else {}

    entries: List[CorpusUtterance] = []
    for speaker_dir in sorted(p for p in wav_dir.iterdir() if p.is_dir()):
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            utt_id = wav_path.stem
            entries.append(
                CorpusUtterance(utt_id, speaker_dir.name, wav_path, refs.get(utt_id))
            )
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


# ---------------------------------------------------------------------------
# THCHS-30 (openslr-18) -- layout NOT verified against the real box
# extraction at build time: the box's no-card download of THCHS-30 FAILED
# (n_wav=0, /root/stage_expand.log STEP1) rather than merely being pending;
# see the expansion report for the retry status. Assumed tarball layout, per
# the standard openslr-18 distribution and the task brief:
#
#     {data_root}/data/*.wav          -- ALL utterances, flat, every split
#     {data_root}/data/*.wav.trn       -- transcript sidecar per wav, 3 lines:
#                                         (1) characters, SPACE-segmented
#                                         (2) pinyin with tone numbers
#                                         (3) pinyin without tones
#                                         only line (1) is used here
#     {data_root}/{split}/*.wav        -- per-split membership (train/dev/test),
#                                         real files OR symlinks into data/,
#                                         same basenames as their data/ counterpart
#
# Speaker id: the filename prefix up to (not including) the first '_', e.g.
# "A11_101.wav" -> speaker "A11" (the THCHS-30/Kaldi-recipe naming
# convention). Falls back to the whole stem (never raises) if a filename
# doesn't match -- exactly like ``asr_gate.io``'s ``_aishell_speaker_id``.
# ---------------------------------------------------------------------------

_THCHS_SPEAKER_RE = re.compile(r"^([A-Za-z]+\d+)_")


def _thchs_speaker_id(stem: str) -> str:
    m = _THCHS_SPEAKER_RE.match(stem)
    return m.group(1) if m else stem


def load_thchs30_transcript(trn_path: Path) -> Optional[str]:
    """First line of a THCHS-30 ``.wav.trn`` sidecar (space-segmented
    characters), spaces stripped to match Aishell's ``ref_text`` convention
    (and ``asr_gate.cer``'s char-level view). ``None`` (never raises) if the
    file is missing or its first line is empty -- callers count missing
    refs, matching ``orchestration/decode_paraformer.py``'s
    ``n_missing_ref`` convention."""
    try:
        with open(trn_path, "r", encoding="utf-8") as f:
            first_line = f.readline().rstrip("\n")
    except OSError:
        return None
    if not first_line.strip():
        return None
    return first_line.replace(" ", "")


def discover_thchs30(
    data_root: Path, split: str, limit: Optional[int] = None
) -> List[CorpusUtterance]:
    """Discover THCHS-30 utterances for ``split`` (``"train"``, ``"dev"``,
    or ``"test"``). Prefers ``{data_root}/{split}/*.wav`` (per-split
    membership); falls back to scanning ``{data_root}/data/*.wav`` directly
    (DEGRADED mode -- no split membership available, every utterance under
    ``data/`` is returned regardless of ``split``, with a caller-visible
    warning) if the split subdirectory doesn't exist, e.g. if the extracted
    tree turns out to be flatter than assumed.

    Transcript lookup always tries ``{data_root}/data/{stem}.wav.trn``
    first (the canonical sidecar location), falling back to a ``.wav.trn``
    colocated with the split-dir wav itself if that canonical path is
    absent (covers the case the split dirs are copies, not symlinks, and
    carry their own sidecars)."""
    data_root = Path(data_root)
    split_dir = data_root / split
    data_dir = data_root / "data"

    if split_dir.exists():
        wav_paths = sorted(split_dir.glob("*.wav"))
    elif data_dir.exists():
        print(
            f"warning: discover_thchs30: split dir {split_dir} not found; "
            f"falling back to {data_dir} (degraded: no split membership, "
            f"every utterance under data/ is returned for every split)",
            file=sys.stderr,
        )
        wav_paths = sorted(data_dir.glob("*.wav"))
    else:
        raise FileNotFoundError(
            f"discover_thchs30: neither {split_dir} nor {data_dir} exists"
        )

    entries: List[CorpusUtterance] = []
    for wav_path in wav_paths:
        stem = wav_path.stem
        trn_path = data_dir / f"{stem}.wav.trn"
        if not trn_path.exists():
            trn_path = wav_path.with_suffix(wav_path.suffix + ".trn")
        ref_text = load_thchs30_transcript(trn_path)
        entries.append(
            CorpusUtterance(stem, _thchs_speaker_id(stem), wav_path, ref_text)
        )
        if limit is not None and len(entries) >= limit:
            break
    return entries


# ---------------------------------------------------------------------------
# aidatatang_200zh (openslr-62) -- layout per the Kaldi recipe's data_prep.sh
# (egs/aidatatang_200zh/s5/local/data_prep.sh), VERIFY against the real box
# extraction (this is a Phase-0 task, per FREEZE-AMENDMENT-2026-07-13.md §2):
#
#     {data_root}/corpus/{split}/{speaker_id}/{utt_id}.wav
#     {data_root}/transcript/aidatatang_200_zh_transcript.txt
#
# The wav tree is `corpus/{train,dev,test}/{speaker}/*.wav` (Kaldi:
# `grep -i "corpus/train" wav.flist`; speaker id = `awk -F/ '{print $(NF-1)}'`
# i.e. the wav's PARENT directory name -- identical to Aishell-1's convention).
# The transcript file has EXACTLY Aishell's "utt_id space-segmented-chars" per
# line (Kaldi: filtered on column 1 by utt_id), so :func:`load_aishell_transcripts`
# parses it verbatim -- reused here rather than duplicated.
# ---------------------------------------------------------------------------


def discover_aidatatang(
    data_root: Path, split: str, limit: Optional[int] = None
) -> List[CorpusUtterance]:
    """aidatatang_200zh (openslr-62). Structurally identical to
    :func:`discover_aishell` except the wav tree lives under ``corpus/{split}/``
    (not ``wav/{split}/``) and the transcript is
    ``transcript/aidatatang_200_zh_transcript.txt`` (same space-segmented format,
    parsed by :func:`load_aishell_transcripts`). Speaker id is the wav's parent
    directory name (the authoritative Kaldi-recipe convention). Raises
    ``FileNotFoundError`` with an actionable message if the split's wav dir is
    absent; degrades to all-missing-refs (never raises) if the transcript file
    is absent, matching :func:`discover_aishell`.

    NB: the FREEZE-AMENDMENT-2026-07-13 speaker-disjoint ~4-5k-utt test CAP is a
    decode-time SELECTION applied by the orchestration layer, not by this
    discoverer -- this function enumerates the whole split deterministically
    (speaker dir sorted, then filename sorted), and ``limit`` truncates in that
    order exactly as the other discoverers do."""
    data_root = Path(data_root)
    wav_dir = data_root / "corpus" / split
    transcript_file = data_root / "transcript" / "aidatatang_200_zh_transcript.txt"
    if not wav_dir.exists():
        raise FileNotFoundError(f"discover_aidatatang: wav dir not found: {wav_dir}")
    refs = load_aishell_transcripts(transcript_file) if transcript_file.exists() else {}

    entries: List[CorpusUtterance] = []
    for speaker_dir in sorted(p for p in wav_dir.iterdir() if p.is_dir()):
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            utt_id = wav_path.stem
            entries.append(
                CorpusUtterance(utt_id, speaker_dir.name, wav_path, refs.get(utt_id))
            )
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


# ---------------------------------------------------------------------------
# MAGICDATA (openslr-68) -- layout per the lhotse recipe
# (lhotse/recipes/magicdata.py), VERIFY against the real box extraction:
#
#     {data_root}/{split}/{speaker_id}/{utt_id}.wav
#     {data_root}/{split}/TRANS.txt
#
# TRANS.txt is whitespace-separated with a HEADER line to skip
# (``UtteranceID  SpeakerID  Transcription``); the UtteranceID column carries
# the ``.wav`` extension (e.g. ``16_4013_20170819121429.wav``), SpeakerID is the
# authoritative speaker (also the wav's parent dir name), and columns [2:] are
# the transcription (word-segmented; whitespace stripped here to match the
# char-level ref convention). TRANS.txt is treated as the source of truth for
# BOTH speaker id and ref text (lhotse uses it directly); the wav's parent dir
# name is a fallback speaker id for wavs absent from TRANS.txt.
# ---------------------------------------------------------------------------


def load_magicdata_trans(trans_path: Path) -> Dict[str, tuple]:
    """Parse a MAGICDATA ``TRANS.txt`` into ``{utt_id: (speaker_id, ref_text)}``,
    keyed by the utterance STEM (the ``.wav`` extension in the UtteranceID column
    is stripped so it matches ``wav_path.stem``). The header line (starting with
    ``"UtteranceID"``) is skipped; ``ref_text`` has all whitespace stripped to
    match the char-level ref convention Aishell/THCHS use. Returns ``{}`` (never
    raises) if the file is missing."""
    mapping: Dict[str, tuple] = {}
    try:
        f = open(trans_path, "r", encoding="utf-8")
    except OSError:
        return mapping
    with f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("UtteranceID"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            utt_col, speaker_id = parts[0], parts[1]
            utt_id = utt_col[:-4] if utt_col.endswith(".wav") else utt_col
            ref_text = "".join(parts[2:]) if len(parts) > 2 else None
            mapping[utt_id] = (speaker_id, ref_text)
    return mapping


def discover_magicdata(
    data_root: Path, split: str, limit: Optional[int] = None
) -> List[CorpusUtterance]:
    """MAGICDATA (openslr-68). Wavs under ``{data_root}/{split}/{speaker}/*.wav``;
    transcript + authoritative speaker id from ``{data_root}/{split}/TRANS.txt``
    (:func:`load_magicdata_trans`). Speaker id prefers the TRANS.txt SpeakerID,
    falling back to the wav's parent directory name for utterances absent from
    TRANS.txt (never raises on a missing/rogue utt). Raises ``FileNotFoundError``
    if the split directory itself is absent. Enumerated deterministically
    (speaker dir sorted, then filename); ``limit`` truncates in that order.

    The FREEZE-AMENDMENT speaker-disjoint test CAP is applied by the
    orchestration layer, not here (see :func:`discover_aidatatang`)."""
    data_root = Path(data_root)
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"discover_magicdata: split dir not found: {split_dir}")
    trans = load_magicdata_trans(split_dir / "TRANS.txt")

    entries: List[CorpusUtterance] = []
    for wav_path in sorted(split_dir.glob("*/*.wav")):
        utt_id = wav_path.stem
        spk_from_trans, ref_text = trans.get(utt_id, (None, None))
        speaker_id = spk_from_trans or wav_path.parent.name
        entries.append(CorpusUtterance(utt_id, speaker_id, wav_path, ref_text))
        if limit is not None and len(entries) >= limit:
            return entries
    return entries


CORPUS_DISCOVERERS: Dict[str, Callable[..., List[CorpusUtterance]]] = {
    "aishell": discover_aishell,
    "thchs30": discover_thchs30,
    "aidatatang": discover_aidatatang,
    "magicdata": discover_magicdata,
}


def discover_corpus(
    corpus: str, data_root: Path, split: str, limit: Optional[int] = None
) -> List[CorpusUtterance]:
    """Dispatch to the right ``discover_*`` by corpus name (one of
    ``CORPUS_DISCOVERERS``: ``"aishell"``, ``"thchs30"``, ``"aidatatang"``,
    ``"magicdata"``); used by ``decode_whisper.py`` / ``decode_conformer_ms.py``
    / ``decode_sherpa_onnx.py`` (and ``decode_paraformer.py`` in ``--corpus``
    mode) so they share one ``--corpus`` switch."""
    if corpus not in CORPUS_DISCOVERERS:
        raise ValueError(f"unknown corpus {corpus!r}; choose one of {sorted(CORPUS_DISCOVERERS)}")
    return CORPUS_DISCOVERERS[corpus](Path(data_root), split, limit)
