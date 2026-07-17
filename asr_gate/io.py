"""Canonical utterance-table I/O and format adapters.

Every ``asr-gate`` command consumes the same normalized record shape,
regardless of the decode format it came from (FunASR, Whisper, WeNet, a
bespoke schema, ...). Adapters (``adapt_*``) are the ONLY format-aware code
in the tool -- everything downstream (``scores.py``, ``gate.py``,
``audit.py``) targets this schema exclusively.

Canonical fields (per utterance)
---------------------------------
Required:
  - ``utt_id`` (str): unique utterance id.
  - ``speaker_id`` (str): speaker id (the Mondrian/blocking unit).
  - ``duration_s`` (float > 0): audio duration in seconds.
  - ``hyp_text`` (str): 1-best hypothesis text (== ``nbest[0]["text"]``).
  - ``nbest`` (list[dict], non-empty): ranked hypotheses, each
    ``{"text": str, "logp": float, "token_logps": list[float] | None,
    "token_full_posteriors": list[list[float]] | None}``. ``token_logps``
    is the per-token log-probability of the CHOSEN token; the optional
    ``token_full_posteriors`` extension (full per-token distribution over
    the vocabulary) is what :func:`asr_gate.scores.compute_s4` needs and is
    rarely available -- absence degrades s4 to ``None``, never imputed.

Optional:
  - ``ref_text`` (str | None): reference transcript; required for
    ``calibrate``/``audit`` (refused otherwise, per the no-ground-truth
    honesty rule), not required for ``apply``.
  - ``gender`` (str | None): speaker gender, for the optional Mondrian axis.
  - ``region`` (str | None): accent-region metadata (exploratory only).

A table with utterances that have inconsistent ``nbest`` shapes (e.g. some
1-best-only, some multi-best) is allowed -- degraded-mode flags are computed
per utterance in ``scores.py``, never assumed uniform across the table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "SchemaError",
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "load_jsonl",
    "write_jsonl",
    "validate_utterances",
    "load_utterances",
    "to_jsonable",
    "adapt_funasr",
    "adapt_whisper",
    "adapt_wenet",
    "adapt_custom_schema",
    "FORMAT_ADAPTERS",
    "ingest",
]

REQUIRED_FIELDS = ("utt_id", "speaker_id", "duration_s", "hyp_text", "nbest")
OPTIONAL_FIELDS = ("ref_text", "gender", "region")


class SchemaError(ValueError):
    """Raised on any utterance-table validation failure, with a precise,
    actionable message (which record, which field, what was expected)."""


def _is_real_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


_AISHELL_SPEAKER_RE = re.compile(r"(S\d{4})")


def _aishell_speaker_id(utt_id: str) -> str:
    """Best-effort Aishell-1 speaker id extraction from a
    ``BAC009S0724W0121``-style utt_id (openslr-33 convention: a fixed
    6-char prefix, then ``S`` + 4 digits speaker id, then ``W`` + 4 digits
    utterance number). Falls back to the whole utt_id if the pattern isn't
    found (e.g. a non-Aishell corpus fed through this adapter)."""
    m = _AISHELL_SPEAKER_RE.search(utt_id)
    return m.group(1) if m else utt_id


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSON-Lines file (one JSON object per non-blank line)."""
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{lineno}: invalid JSON ({e})") from e
    if not records:
        raise SchemaError(f"{path}: contains no records")
    return records


def write_jsonl(path: Union[str, Path], records: List[Dict[str, Any]]) -> None:
    """Write a list of JSON-serializable records as JSON-Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain Python types."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep elsewhere
        np = None  # type: ignore[assignment]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [to_jsonable(v) for v in obj.tolist()]
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    return obj


def validate_utterances(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate + normalize a list of raw records into canonical utterances.

    Raises :class:`SchemaError` identifying the offending record (position
    and, if available, ``utt_id``) on any problem.
    """
    if not isinstance(records, list) or len(records) == 0:
        raise SchemaError("utterance table must be a non-empty list of records")

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()

    for i, rec in enumerate(records):
        tag = f"record[{i}]"
        if not isinstance(rec, dict):
            raise SchemaError(f"{tag}: expected an object, got {type(rec).__name__}")
        utt_id = rec.get("utt_id")
        if utt_id is not None:
            tag = f"record[{i}] (utt_id={utt_id!r})"

        for field in REQUIRED_FIELDS:
            if field not in rec or rec[field] is None:
                raise SchemaError(f"{tag}: missing required field '{field}'")

        if not isinstance(rec["utt_id"], (str, int)):
            raise SchemaError(f"{tag}: 'utt_id' must be str or int")
        utt_id_norm = str(rec["utt_id"])
        if utt_id_norm in seen_ids:
            raise SchemaError(f"{tag}: duplicate utt_id {utt_id_norm!r}")
        seen_ids.add(utt_id_norm)

        if not isinstance(rec["speaker_id"], (str, int)):
            raise SchemaError(f"{tag}: 'speaker_id' must be str or int")
        if not _is_real_number(rec["duration_s"]) or rec["duration_s"] <= 0:
            raise SchemaError(f"{tag}: 'duration_s' must be a positive number")
        if not isinstance(rec["hyp_text"], str):
            raise SchemaError(f"{tag}: 'hyp_text' must be a string")

        nbest = rec["nbest"]
        if not isinstance(nbest, list) or len(nbest) == 0:
            raise SchemaError(f"{tag}: 'nbest' must be a non-empty list")
        nbest_norm = []
        for j, hyp in enumerate(nbest):
            if not isinstance(hyp, dict) or "text" not in hyp:
                raise SchemaError(f"{tag}: nbest[{j}] must be an object with a 'text' field")
            logp = hyp.get("logp")
            if logp is not None and not _is_real_number(logp):
                raise SchemaError(f"{tag}: nbest[{j}]['logp'] must be a number or null")
            token_logps = hyp.get("token_logps")
            if token_logps is not None:
                if not isinstance(token_logps, list) or not all(
                    _is_real_number(v) for v in token_logps
                ):
                    raise SchemaError(
                        f"{tag}: nbest[{j}]['token_logps'] must be a list of numbers or null"
                    )
            token_full_posteriors = hyp.get("token_full_posteriors")
            nbest_norm.append(
                {
                    "text": hyp["text"],
                    "logp": None if logp is None else float(logp),
                    "token_logps": (
                        None if token_logps is None else [float(v) for v in token_logps]
                    ),
                    "token_full_posteriors": token_full_posteriors,
                }
            )

        ref_text = rec.get("ref_text")
        if ref_text is not None and not isinstance(ref_text, str):
            raise SchemaError(f"{tag}: 'ref_text' must be a string or null")
        gender = rec.get("gender")
        if gender is not None and not isinstance(gender, str):
            raise SchemaError(f"{tag}: 'gender' must be a string or null")
        region = rec.get("region")
        if region is not None and not isinstance(region, str):
            raise SchemaError(f"{tag}: 'region' must be a string or null")

        normalized.append(
            {
                "utt_id": utt_id_norm,
                "speaker_id": str(rec["speaker_id"]),
                "duration_s": float(rec["duration_s"]),
                "hyp_text": rec["hyp_text"],
                "nbest": nbest_norm,
                "ref_text": ref_text,
                "gender": gender,
                "region": region,
            }
        )

    return normalized


def load_utterances(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load + validate a canonical utterance table from a ``.jsonl`` file."""
    path = Path(path)
    if path.suffix != ".jsonl":
        raise SchemaError(
            f"{path}: unrecognized extension {path.suffix!r}; expected .jsonl "
            "(the canonical asr-gate utterance-table format)"
        )
    return validate_utterances(load_jsonl(path))


# ---------------------------------------------------------------------------
# Format adapters. Real FunASR/Whisper/WeNet decode-output shapes vary across
# versions and CLI options; each adapter below documents the exact input
# shape it assumes and fails loudly (KeyError/SchemaError) rather than
# silently guessing on an unrecognized shape. See the top-level report for
# the documented deviation this implies.
# ---------------------------------------------------------------------------


def adapt_funasr(
    records: List[Dict[str, Any]],
    speaker_map: Optional[Dict[str, str]] = None,
    duration_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Adapt raw FunASR ``AutoModel.generate()`` output records.

    Verified real shape (funasr 1.3.14, ``model="paraformer-zh"``, no
    ``hotword``/beam-search kwargs -- checked against actual box output)::

        {"key": "BAC009S0764W0121", "text": "今 天 天 气 很 好",
         "timestamp": [[...], ...]}

    The high-level ``generate()`` API genuinely does NOT expose
    ``logp``/``token_logps``/``nbest`` -- this adapter's aliases below are
    a best-effort probe for other FunASR model families / future
    versions, not something ever observed for paraformer-zh. Two things
    this adapter DOES fix for the verified shape:

    - ``text`` has FunASR's own literal ASCII space inserted between
      every emitted character; stripped here (via the same logic as
      ``orchestration/decode_paraformer.py``'s ``_strip_char_spacing``)
      so ``hyp_text``/``nbest[*]["text"]`` store clean text rather than
      relying on ``asr_gate.cer.normalize_text`` to paper over it
      downstream.
    - Real per-token log-probabilities are NOT recoverable from a
      serialized FunASR JSON dump at all -- they exist only as an
      internal decoder tensor discarded before ``generate()`` returns.
      Recovering them requires live model access at decode time (a
      monkeypatch hook on the model's internals), which is what
      ``orchestration/decode_paraformer.py`` does directly, bypassing
      this adapter entirely (see that script's module docstring). This
      adapter is for the case a raw FunASR JSON dump is ingested
      standalone after the fact, where ``token_logps`` will correctly
      stay ``None`` (degraded mode) unless the dump already carries a
      ``token_logps`` field from some other source.

    ``speaker_map``/``duration_map`` (utt_id -> value) supply metadata
    FunASR's decode output does not carry (speaker id is usually recovered
    from the Aishell wav-path convention upstream, in
    ``orchestration/decode_paraformer.py``, which emits canonical JSONL
    directly -- this adapter exists for the case a raw FunASR dump is
    ingested standalone).
    """
    out = []
    for rec in records:
        utt_id = rec.get("key") or rec.get("utt_id") or rec.get("uttid")
        if utt_id is None:
            raise SchemaError("adapt_funasr: record missing 'key'/'utt_id'/'uttid'")
        text = "".join(rec.get("text", "").split())
        nbest_raw = rec.get("nbest")
        if nbest_raw:
            nbest = [
                {
                    "text": "".join(h.get("text", text).split()),
                    "logp": h.get("logp"),
                    "token_logps": h.get("token_logps"),
                }
                for h in nbest_raw
            ]
        else:
            nbest = [
                {
                    "text": text,
                    "logp": rec.get("logp"),
                    "token_logps": rec.get("token_logps"),
                }
            ]
        speaker_id = (
            rec.get("speaker_id")
            or (speaker_map or {}).get(str(utt_id))
            or _aishell_speaker_id(str(utt_id))
        )
        duration_s = rec.get("duration_s") or (duration_map or {}).get(str(utt_id)) or 1.0
        out.append(
            {
                "utt_id": str(utt_id),
                "speaker_id": str(speaker_id),
                "duration_s": float(duration_s),
                "hyp_text": text,
                "nbest": nbest,
                "ref_text": rec.get("ref_text"),
                "gender": rec.get("gender"),
                "region": rec.get("region"),
            }
        )
    return validate_utterances(out)


def adapt_whisper(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt raw Whisper ``transcribe()`` output records.

    Assumed input shape (Whisper's Python API returns ``avg_logprob`` as a
    SEQUENCE-level average, not per-token log-probs; there is no N-best and
    no per-token posterior in the standard API, so ``token_logps`` is left
    ``None`` -- s1 must be recovered from ``avg_logprob`` directly, s2/s3/s4
    degrade to ``None``, documented in the top-level report)::

        {"utt_id": "...", "speaker_id": "...", "duration_s": 4.2,
         "text": "今天天气很好", "avg_logprob": -0.31,
         "ref_text": "...", "gender": "...", "lid": "zh"}

    A non-``zh`` ``lid`` (language-id) field is preserved for the OOD/LID
    refusal check in ``gate.py`` (§2.5) via the ``region`` passthrough field
    reused here as a carrier -- callers relying on LID-based OOD refusal
    should keep ``lid`` alongside the canonical fields upstream.
    """
    out = []
    for rec in records:
        utt_id = rec.get("utt_id")
        if utt_id is None:
            raise SchemaError("adapt_whisper: record missing 'utt_id'")
        text = rec.get("text", "")
        avg_logprob = rec.get("avg_logprob")
        out.append(
            {
                "utt_id": str(utt_id),
                "speaker_id": str(rec.get("speaker_id", utt_id)),
                "duration_s": float(rec.get("duration_s", 1.0)),
                "hyp_text": text,
                "nbest": [
                    {
                        "text": text,
                        "logp": avg_logprob,
                        "token_logps": rec.get("token_logps"),
                    }
                ],
                "ref_text": rec.get("ref_text"),
                "gender": rec.get("gender"),
                "region": rec.get("lid") or rec.get("region"),
            }
        )
    return validate_utterances(out)


def adapt_wenet(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt raw WeNet CTC/attention-rescoring decode output records.

    Assumed input shape (WeNet's ``recognize.py``/``--result_file`` output,
    one JSON object per line with a beam of hypotheses under ``nbest``)::

        {"key": "BAC009S0764W0121", "nbest": [
            {"sentence": "今天天气很好", "score": -8.1}, ...]}

    WeNet does not expose per-token log-posteriors in its standard decode
    output, so ``token_logps`` is left ``None`` here (s1/s2/s4 degrade to
    ``None`` for WeNet-adapted rows; s3's N-best margin remains available).
    """
    out = []
    for rec in records:
        utt_id = rec.get("key") or rec.get("utt_id")
        if utt_id is None:
            raise SchemaError("adapt_wenet: record missing 'key'/'utt_id'")
        nbest_raw = rec.get("nbest", [])
        if not nbest_raw:
            raise SchemaError(f"adapt_wenet: record {utt_id!r} has empty 'nbest'")
        nbest = [
            {
                "text": h.get("sentence", h.get("text", "")),
                "logp": h.get("score", h.get("logp")),
                "token_logps": None,
            }
            for h in nbest_raw
        ]
        out.append(
            {
                "utt_id": str(utt_id),
                "speaker_id": str(rec.get("speaker_id", _aishell_speaker_id(str(utt_id)))),
                "duration_s": float(rec.get("duration_s", 1.0)),
                "hyp_text": nbest[0]["text"],
                "nbest": nbest,
                "ref_text": rec.get("ref_text"),
                "gender": rec.get("gender"),
                "region": rec.get("region"),
            }
        )
    return validate_utterances(out)


def adapt_custom_schema(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass-through adapter: ``records`` are already (or nearly) canonical.

    Runs :func:`validate_utterances` directly -- for pipelines that already
    emit the canonical shape (e.g. ``orchestration/decode_paraformer.py``).
    """
    return validate_utterances(records)


FORMAT_ADAPTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "funasr": adapt_funasr,
    "whisper": adapt_whisper,
    "wenet": adapt_wenet,
    "custom-schema": adapt_custom_schema,
}


def ingest(
    hyps_path: Union[str, Path],
    fmt: str,
    refs_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Ingest a raw decode-output JSONL file into the canonical schema.

    Parameters
    ----------
    hyps_path:
        Raw decode-output JSONL (format given by ``fmt``).
    fmt:
        One of ``"funasr"``, ``"whisper"``, ``"wenet"``, ``"custom-schema"``.
    refs_path:
        Optional plain-text references file, one line per utterance in the
        Kaldi/Aishell ``utt_id ref_text`` convention (``utt_id`` then
        whitespace then the reference, tab or space separated); merged into
        ``ref_text`` by ``utt_id``. Utterances without a matching line get
        ``ref_text = None``.
    """
    if fmt not in FORMAT_ADAPTERS:
        raise SchemaError(f"unknown format {fmt!r}; choose one of {sorted(FORMAT_ADAPTERS)}")
    raw = load_jsonl(hyps_path)
    utterances = FORMAT_ADAPTERS[fmt](raw)

    if refs_path is not None:
        refs: Dict[str, str] = {}
        with open(refs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                refs[parts[0]] = parts[1]
        for u in utterances:
            if u["utt_id"] in refs:
                u["ref_text"] = refs[u["utt_id"]]

    return utterances
