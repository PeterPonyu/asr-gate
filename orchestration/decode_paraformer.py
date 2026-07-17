#!/usr/bin/env python3
"""Box-side decode script: FunASR Paraformer-zh over an Aishell-1 split,
emitting the canonical ``asr_gate`` decode JSONL directly (bypassing the
``ingest --format funasr`` adapter -- see ``asr_gate/io.py``'s
``adapt_funasr`` docstring for when that adapter is still useful).

Runs on the AutoDL box, NOT in CI/tests: ``funasr``/``torch`` are imported
lazily inside :func:`_load_model` so ``--help`` (and argument parsing)
works in ANY environment, including one with neither installed -- this is
exercised by ``tests/test_cli_e2e.py::test_decode_paraformer_help_runs_without_funasr``.

Assumed local layout (openslr-33 convention, per design §3.1/§7 Phase 0;
VERIFY against the actual box before a real run -- this is a Phase-0 task,
not something this script can check for you)::

    {data_root}/wav/{split}/{speaker_id}/{utt_id}.wav
    {data_root}/transcript/aishell_transcript_v0.8.txt   # "utt_id ref_text" per line

Result-shape tolerance
-----------------------
FunASR's ``AutoModel.generate()`` return shape has varied across releases
-- sometimes a list of ``{"key": ..., "text": ...}`` dicts (text only),
sometimes richer per-token detail under version-specific keys. This script
extracts what it can via a small tolerant probe (:func:`_extract_result`)
and degrades to text-only (``token_logps=None``, ``s1/s2/s4`` will be
``None`` downstream) rather than crashing on an unrecognized shape.

Verified real output shape (funasr 1.3.14, torch 2.8+cu128, on the AutoDL
decode box, ``model="paraformer-zh"``)
--------------------------------------------------------------------------
``AutoModel(model="paraformer-zh").generate(input=wav, batch_size=1)``
returns exactly::

    [{"key": "BAC009S0724W0121", "text": "广 州 市 房 地 产 中 介 协 会 分 析",
      "timestamp": [[410, 650], [690, 930], ...]}]

No ``logp``/``token_logps``/``nbest`` key is present at this level -- the
high-level API genuinely does not expose per-token scores. Two concrete
consequences fixed here:

1. ``text`` has a literal ASCII space inserted between every emitted
   character (funasr's own postprocessing) -- :func:`_strip_char_spacing`
   removes it so ``hyp_text``/``nbest[*]["text"]`` store clean text,
   consistent with (rather than dependent on) ``asr_gate.cer``'s own
   whitespace-stripping normalizer.
2. Per-token log-probabilities ARE computed internally by the model (a
   full log-softmax over the ~8404-token vocabulary per emitted
   character) but are discarded by ``AutoModel``/``inference()`` before
   the public result is built. :func:`_install_token_logp_hook` recovers
   them via a monkeypatch (no funasr fork) on
   ``model.model._seaco_decode_with_ASF`` (funasr 1.3.14,
   ``funasr/models/seaco_paraformer/model.py`` -- "paraformer-zh" resolves
   to the SeaCo-Paraformer variant). With no ``hotword`` kwarg (the
   default used here), that method returns
   ``torch.log_softmax(decoder_out, dim=-1)`` directly, shape
   ``[batch, T, vocab]``. ``inference()`` then does
   ``am_scores.argmax(dim=-1)`` / ``.max(dim=-1)`` over exactly this
   tensor to pick emitted tokens and scores, but never returns the
   scores. :func:`_extract_hooked_token_logps` replays that same argmax
   plus funasr's own sos/eos/blank-token filtering to recover a
   token_logps sequence aligned 1:1 with the emitted text characters --
   verified empirically against real Aishell-1 dev utterances (filtered
   length exactly matches character count in every probe case; Mandarin
   is effectively character-tokenized in this vocab, matching the
   assumption already documented in ``asr_gate/scores.py``'s s3
   docstring). The sequence-level ``logp`` written to the canonical
   record is ``sum(token_logps)`` over that same aligned set.

   This gives ``s1``/``s2`` (and the ``logp`` s3 needs, though a
   multi-hypothesis ``nbest`` never materializes here since
   ``model.model.beam_search`` is ``None`` for plain paraformer-zh
   decoding -- s3 stays degraded/``None``, as documented in scores.py).
   ``token_full_posteriors`` (full per-token vocabulary distribution,
   needed for s4) is technically available from the SAME captured tensor
   (each row already IS the full log-softmax) but is deliberately NOT
   persisted: for a real utterance count, ``O(tokens x ~8404-vocab)``
   floats per utterance would blow up decode JSONL size by ~3 orders of
   magnitude (a back-of-envelope full-dev-split run: ~7GB+). s4 therefore
   stays ``None`` (degraded mode) -- explicitly permitted by the design
   ("if only chosen-token logps are available, s4 stays None").

   If the hook can't attach (unexpected funasr version/model without
   ``_seaco_decode_with_ASF``) or the captured tensor's shape doesn't
   line up with the emitted text (alignment sanity check fails),
   extraction degrades to ``token_logps=None`` for that utterance with a
   stderr warning -- never crashes the decode run.

``--wav-list`` input mode (added 2026-07-09 for the noise-stratified arm,
EXPANSION-AMENDMENT-2026-07-09.md's MUSAN/ESC-50 noise axis)
--------------------------------------------------------------------------
The default path above discovers utterances from an Aishell-layout wav
tree (``{data_root}/wav/{split}/{speaker_id}/{utt_id}.wav``) -- this stays
UNTOUCHED. ``--wav-list FILE`` is an ALTERNATE input mode for decoding an
arbitrary flat list of wavs (e.g. ``orchestration/mix_musan.py``'s
noise-mixed output directory, which has no Aishell speaker-dir structure):
``utt_id wav_path`` per line, same convention as ``mix_musan.py``'s own
wav-list parsing. Speaker id is recovered from ``utt_id`` via a regex
(``S####``, the Aishell/Kaldi convention -- same pattern
``asr_gate.corpora._AISHELL_SPEAKER_RE`` matches, duplicated here rather
than imported per this script's own "predates corpora.py, inlines its own
logic" convention, see the module docstring above), falling back to the
whole ``utt_id`` (never raises) if no match.

``--transcript-source {aishell,none}`` governs how ``ref_text`` is
resolved in this mode: ``aishell`` (default) looks ``utt_id`` up in
``--transcript-file``/the usual ``{data-root}/transcript/...`` default --
correct for noise-mixed Aishell audio, since mixing preserves ``utt_id``;
``none`` leaves every ``ref_text`` as ``None`` (e.g. a wav-list with no
known clean-text mapping).

``--corpus`` switch (added 2026-07-13 for the landscape expansion,
FREEZE-AMENDMENT-2026-07-13.md)
--------------------------------------------------------------------------
The DEFAULT ``--corpus aishell`` path is the original Aishell-inlined
discovery above and is **byte-for-byte UNTOUCHED** -- the default
invocation (no ``--corpus``, no ``--wav-list``) hits exactly the same
``_discover_wavs`` + ``_load_transcripts`` code as before (this script
"predates corpora.py and inlines its own Aishell-only logic", per this
module's design intent, and that logic stays real-box-verified and
unchanged). Any OTHER ``--corpus`` value (``thchs30``/``aidatatang``/
``magicdata``) routes through the shared ``asr_gate.corpora.discover_corpus``
adapter instead -- letting Paraformer-zh decode THCHS-30 / aidatatang /
MagicData for the cross-corpus certificate cells, exactly as
``decode_whisper.py``/``decode_conformer_ms.py`` already do. ``--wav-list``
mode takes precedence over ``--corpus`` (it is Aishell-utt-id specific).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/asr-gate/ root
from asr_gate import corpora  # noqa: E402  (only used by the non-Aishell --corpus branch)


def _wav_duration_s(path: Path) -> float:
    """Duration via the stdlib ``wave`` module (no extra dependency). Falls
    back to 1.0 with a stderr warning if the file can't be read as PCM WAV
    (canonical schema requires ``duration_s > 0``)."""
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception as e:  # noqa: BLE001 - best-effort, never fatal
        print(f"warning: could not read duration of {path}: {e}", file=sys.stderr)
    return 1.0


def _load_transcripts(transcript_file: Path) -> Dict[str, str]:
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
            refs[utt_id] = ref_text.replace(" ", "")  # Aishell transcript is space-segmented
    return refs


def _discover_wavs(wav_dir: Path, limit: Optional[int]) -> List[Tuple[str, str, Path]]:
    """Returns ``[(utt_id, speaker_id, wav_path), ...]`` from
    ``wav_dir/{speaker_id}/{utt_id}.wav`` (openslr-33 layout)."""
    entries = []
    for speaker_dir in sorted(p for p in wav_dir.iterdir() if p.is_dir()):
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            utt_id = wav_path.stem
            entries.append((utt_id, speaker_dir.name, wav_path))
            if limit is not None and len(entries) >= limit:
                return entries
    return entries


_WAVLIST_SPEAKER_RE = re.compile(r"(S\d{4})")


def _wavlist_speaker_id(utt_id: str) -> str:
    """Best-effort Aishell speaker id from a bare ``utt_id`` -- no wav_path
    directory structure to read it from in ``--wav-list`` mode, unlike
    :func:`_discover_wavs`'s default path. Same ``S####`` convention
    ``asr_gate.corpora._AISHELL_SPEAKER_RE`` matches. Falls back to the
    whole ``utt_id`` (never raises) if no match, mirroring
    ``asr_gate.corpora._thchs_speaker_id``'s identical fallback."""
    m = _WAVLIST_SPEAKER_RE.search(utt_id)
    return m.group(1) if m else utt_id


def _load_wav_list_entries(
    wav_list_path: Path, limit: Optional[int]
) -> List[Tuple[str, str, Path]]:
    """``--wav-list`` input mode (see module docstring): ``utt_id wav_path``
    per line, same convention as ``orchestration/mix_musan.py``'s wav-list
    parsing. Returns ``[(utt_id, speaker_id, wav_path), ...]`` in FILE
    order (no sort -- matches whatever order the caller's manifest/list
    already established), speaker id via :func:`_wavlist_speaker_id`."""
    entries: List[Tuple[str, str, Path]] = []
    with open(wav_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, wav_path = parts
            entries.append((utt_id, _wavlist_speaker_id(utt_id), Path(wav_path)))
            if limit is not None and len(entries) >= limit:
                break
    return entries


def _load_model(model_name: str, device: str):
    """Lazy FunASR import -- keeps ``--help`` working with no funasr/torch
    installed. Only called from :func:`main` after argument parsing."""
    try:
        from funasr import AutoModel  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "decode_paraformer.py: funasr is not installed in this environment. "
            "Install it on the decode box (`pip install funasr`) -- this is a "
            "box-side script, not part of the asr-gate package's own dependencies."
        ) from e
    return AutoModel(model=model_name, device=device)


def _strip_char_spacing(text: str) -> str:
    """Strip all whitespace from FunASR's decoded text. FunASR's own
    postprocessing inserts a literal ASCII space between every emitted
    character/token -- verified against real funasr 1.3.14 paraformer-zh
    output, e.g. ``'广 州 市 房 地 产 中 介 协 会 分 析'``. Stored
    ``hyp_text``/``nbest[*]["text"]`` should be clean, consistent with
    (rather than dependent on) ``asr_gate.cer.normalize_text``'s own
    whitespace-stripping."""
    if not text:
        return text
    return "".join(text.split())


def _install_token_logp_hook(model) -> Dict[str, Any]:
    """Monkeypatch the underlying SeaCo-Paraformer model's internal
    ``_seaco_decode_with_ASF`` (funasr 1.3.14,
    ``funasr/models/seaco_paraformer/model.py``) to capture the full
    per-token log-softmax tensor it computes on every ``generate()`` call,
    WITHOUT forking funasr. See the module docstring for the verified
    mechanics (why this method's return value IS the full per-token
    log-probability distribution when no ``hotword`` kwarg is used).

    Returns a mutable ``capture`` dict that gets a ``"log_probs"`` key
    (a numpy array, already moved off-GPU) set on every ``generate()``
    call the hook fires for; callers should ``capture.pop("log_probs",
    None)`` between utterances to avoid reusing a stale capture if a call
    unexpectedly doesn't route through the hooked method. If this
    funasr/model version doesn't expose ``_seaco_decode_with_ASF`` at
    all, the hook is not installed, a warning is printed once, and the
    returned dict never gains a ``"log_probs"`` key -- token_logps then
    degrades to ``None`` for every utterance rather than crashing.
    """
    capture: Dict[str, Any] = {}
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "_seaco_decode_with_ASF"):
        print(
            "warning: model has no '_seaco_decode_with_ASF' (unexpected funasr "
            "version/model) -- token_logps will be None for every utterance",
            file=sys.stderr,
        )
        return capture

    import numpy as np  # lazy: keeps --help working with no numpy installed

    original = inner._seaco_decode_with_ASF

    def _wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            capture["log_probs"] = out.detach().cpu().numpy()
        except AttributeError:
            capture["log_probs"] = np.asarray(out)
        return out

    inner._seaco_decode_with_ASF = _wrapped
    capture["_sos"] = getattr(inner, "sos", None)
    capture["_eos"] = getattr(inner, "eos", None)
    capture["_blank_id"] = getattr(inner, "blank_id", None)
    return capture


def _extract_hooked_token_logps(
    capture: Dict[str, Any], n_chars: int
) -> Optional[List[float]]:
    """Recover per-token (per-character) log-probabilities for the just-
    decoded utterance from the tensor :func:`_install_token_logp_hook`
    captured, replaying funasr's own argmax + sos/eos/blank-id filtering
    (see module docstring) to align 1:1 with ``n_chars`` emitted
    characters. Returns ``None`` (degraded mode, never raises) if the
    hook never fired for this call, or if the recovered count doesn't
    match ``n_chars`` (alignment sanity check -- an unexpected funasr
    internal change should degrade gracefully, not silently misalign
    scores to the wrong characters)."""
    log_probs = capture.get("log_probs")
    if log_probs is None:
        return None
    try:
        import numpy as np  # lazy: keeps --help working with no numpy installed

        lp = np.asarray(log_probs)[0]  # [T, vocab]; batch_size=1 contract
        chosen_ids = np.argmax(lp, axis=-1)
        chosen_logps = np.max(lp, axis=-1)
        skip_ids = {capture.get("_sos"), capture.get("_eos"), capture.get("_blank_id")}
        kept = [
            float(chosen_logps[i])
            for i in range(chosen_ids.shape[0])
            if int(chosen_ids[i]) not in skip_ids
        ]
    except Exception as e:  # noqa: BLE001 - best-effort extraction, never fatal
        print(f"warning: token_logp extraction failed: {e}", file=sys.stderr)
        return None
    if len(kept) != n_chars:
        print(
            f"warning: token_logp alignment mismatch (got {len(kept)} scores for "
            f"{n_chars} chars) -- degrading token_logps to None for this utterance",
            file=sys.stderr,
        )
        return None
    return kept


def _extract_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Tolerant extraction of one FunASR result item into
    ``{"text": str, "logp": float|None, "token_logps": list|None}``.
    Handles the verified real text-only shape (see module docstring) and
    best-effort probes a few version-specific alternate key names for
    per-token/sequence scores without assuming any of them are present
    (none are, in the verified funasr 1.3.14 shape -- real token_logps
    come from :func:`_extract_hooked_token_logps` instead, applied by the
    caller). ``text`` is cleaned of FunASR's inter-character spacing via
    :func:`_strip_char_spacing`."""
    text = _strip_char_spacing(item.get("text", ""))
    if not text:
        print(f"warning: empty/missing 'text' in funasr result item: {item!r}", file=sys.stderr)
    logp = None
    for key in ("logp", "score", "avg_logprob", "sentence_logp"):
        if key in item and isinstance(item[key], (int, float)):
            logp = float(item[key])
            break
    token_logps = None
    for key in ("token_logps", "token_score", "us_score", "token_confidence"):
        if key in item and isinstance(item[key], list):
            try:
                token_logps = [float(v) for v in item[key]]
            except (TypeError, ValueError):
                token_logps = None
            break
    return {"text": text, "logp": logp, "token_logps": token_logps}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decode_paraformer.py",
        description="Decode an Aishell-1 split with FunASR Paraformer-zh -> canonical asr-gate JSONL.",
    )
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument(
        "--corpus", default="aishell", choices=sorted(corpora.CORPUS_DISCOVERERS),
        help="'aishell' (default) uses the original inlined Aishell discovery UNTOUCHED; "
             "any other corpus routes through asr_gate.corpora.discover_corpus. Ignored "
             "when --wav-list is given.",
    )
    parser.add_argument(
        "--data-root", default="/root/autodl-tmp/data_aishell",
        help="corpus root. Aishell default is openslr-33 layout ({root}/wav/{split}/..., "
             "{root}/transcript/...); other --corpus values use their own layout via "
             "asr_gate.corpora.discover_corpus.",
    )
    parser.add_argument(
        "--transcript-file", default=None,
        help="default: {data-root}/transcript/aishell_transcript_v0.8.txt",
    )
    parser.add_argument(
        "--wav-list", default=None,
        help="ALTERNATE input mode: 'utt_id wav_path' per line (e.g. a MUSAN/ESC-50-mixed "
             "noise arm's manifest-derived wav list) instead of discovering the Aishell wav "
             "tree under --data-root/--split -- see module docstring",
    )
    parser.add_argument(
        "--transcript-source", default="aishell", choices=["aishell", "none"],
        help="only used with --wav-list: 'aishell' (default) looks up ref_text in "
             "--transcript-file by utt_id; 'none' leaves ref_text null for every utterance",
    )
    parser.add_argument("--out", required=True, help="output canonical decode JSONL path")
    parser.add_argument("--model-name", default="paraformer-zh")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--nbest", type=int, default=1, help="beam width for N-best (if supported)")
    parser.add_argument("--limit", type=int, default=None, help="decode only the first N utterances (smoke test)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="if --out already exists and is non-empty, do nothing (idempotent pipelines)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"decode_paraformer: {out_path} already exists, skipping (--skip-existing)")
        return 0

    data_root = Path(args.data_root)
    transcript_file = Path(args.transcript_file) if args.transcript_file else (
        data_root / "transcript" / "aishell_transcript_v0.8.txt"
    )

    if args.wav_list:
        # ALTERNATE input mode -- see module docstring. The default
        # Aishell-wav-tree discovery below is untouched by this branch.
        if args.transcript_source == "aishell":
            if not transcript_file.exists():
                raise SystemExit(
                    "decode_paraformer.py: --transcript-source aishell (default) requires a "
                    f"transcript file; not found: {transcript_file} (pass --transcript-file, "
                    "or --transcript-source none to leave ref_text null)"
                )
            refs = _load_transcripts(transcript_file)
        else:
            refs = {}
        entries = _load_wav_list_entries(Path(args.wav_list), args.limit)
        if not entries:
            raise SystemExit(f"decode_paraformer.py: no entries found in --wav-list {args.wav_list}")
    elif args.corpus != "aishell":
        # NON-Aishell --corpus: route through the shared discoverer (THCHS-30 /
        # aidatatang / MagicData). The discoverer carries ref_text per utterance;
        # flatten to the same (utt_id, speaker_id, wav_path) entry tuples + a
        # {utt_id: ref_text} refs dict the untouched decode loop below consumes,
        # so only the DISCOVERY differs -- decode/hook/output stay identical.
        utts = corpora.discover_corpus(args.corpus, data_root, args.split, args.limit)
        if not utts:
            raise SystemExit(
                f"decode_paraformer.py: no utterances for corpus={args.corpus} "
                f"split={args.split} under {data_root}"
            )
        entries = [(u.utt_id, u.speaker_id, u.wav_path) for u in utts]
        refs = {u.utt_id: u.ref_text for u in utts if u.ref_text is not None}
    else:
        wav_dir = data_root / "wav" / args.split
        if not wav_dir.exists():
            raise SystemExit(f"decode_paraformer.py: wav dir not found: {wav_dir}")
        if not transcript_file.exists():
            raise SystemExit(f"decode_paraformer.py: transcript file not found: {transcript_file}")
        refs = _load_transcripts(transcript_file)
        entries = _discover_wavs(wav_dir, args.limit)
        if not entries:
            raise SystemExit(f"decode_paraformer.py: no .wav files found under {wav_dir}")

    model = _load_model(args.model_name, args.device)
    logp_capture = _install_token_logp_hook(model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_missing_ref = 0
    skipped: List[Dict[str, str]] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for utt_id, speaker_id, wav_path in entries:
            logp_capture.pop("log_probs", None)
            # Robustness (verified need: Aishell-1's extracted dev split
            # contains 2 zero-frame 44-byte wavs, train 15 more; funasr's
            # fbank frontend asserts on them and would kill the whole run).
            # Skips are COUNTED and logged to a sidecar, never silent.
            dur = _wav_duration_s(wav_path)
            if dur < 0.1:
                skipped.append({"utt_id": utt_id, "reason": f"too_short:{dur:.3f}s"})
                continue
            try:
                raw_results = model.generate(input=str(wav_path), batch_size=1)
            except Exception as exc:  # one bad utterance must not kill 14k
                skipped.append({"utt_id": utt_id, "reason": f"generate_error:{type(exc).__name__}:{exc}"})
                continue
            if isinstance(raw_results, dict):
                raw_results = [raw_results]
            if not raw_results:
                skipped.append({"utt_id": utt_id, "reason": "empty_result"})
                continue
            nbest_items = [_extract_result(r) for r in raw_results[: max(args.nbest, 1)]]
            # The hooked tensor corresponds to THIS generate() call's single
            # batch entry -- only the top-1 hypothesis (the only one that
            # materializes for plain paraformer-zh, no beam search) can be
            # aligned to it.
            if nbest_items and nbest_items[0]["token_logps"] is None:
                token_logps = _extract_hooked_token_logps(
                    logp_capture, len(nbest_items[0]["text"])
                )
                if token_logps:
                    nbest_items[0]["token_logps"] = token_logps
                    nbest_items[0]["logp"] = float(sum(token_logps))
            ref_text = refs.get(utt_id)
            if ref_text is None:
                n_missing_ref += 1
            record = {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "duration_s": _wav_duration_s(wav_path),
                "hyp_text": nbest_items[0]["text"],
                "nbest": [
                    {"text": h["text"], "logp": h["logp"], "token_logps": h["token_logps"]}
                    for h in nbest_items
                ],
                "ref_text": ref_text,
                "gender": None,
                "region": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    if skipped:
        skip_path = out_path.with_suffix(".skipped.jsonl")
        with open(skip_path, "w", encoding="utf-8") as sf:
            for s in skipped:
                sf.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(
        f"decode_paraformer: split={args.split} n_written={n_written} "
        f"n_missing_ref={n_missing_ref} n_skipped={len(skipped)} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
