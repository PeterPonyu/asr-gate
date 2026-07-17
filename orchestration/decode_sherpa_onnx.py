#!/usr/bin/env python3
"""Box-side decode script: a sherpa-onnx OFFLINE TRANSDUCER (zipformer RNN-T)
over a Mandarin corpus split, emitting the canonical ``asr_gate`` decode JSONL
directly. This is the one new decoder the landscape expansion needs
(COMPUTE-PLAN-2026-07-13.md §4, Path D): it buys a THIRD architecture
(pruned/RNN-T transducer, e.g. ``zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2``
with fully-open training data) through the pip-installable ONNX runtime -- no
k2 / kaldifeat / sherpa from-source build (the deliberate answer to the prior
WeNet build failure).

Runs on the decode box, NOT in CI/tests: ``sherpa_onnx``/``soundfile`` are
imported lazily inside :func:`_load_recognizer` / the decode loop so ``--help``
and argument parsing work in ANY environment (including one with neither
installed) -- this is exercised by
``tests/test_decode_sherpa_onnx.py::test_help_runs_without_sherpa_onnx``. It
dispatches corpus layout to ``asr_gate.corpora.discover_corpus`` (the shared
``--corpus`` switch), exactly like ``decode_whisper.py`` /
``decode_conformer_ms.py``.

Posterior extraction (Path D -- the make-or-break criterion)
------------------------------------------------------------
For an offline transducer, ``sherpa_onnx.OfflineRecognizer.from_transducer(...)``
returns, per stream, an ``OfflineRecognizerResult`` exposing (current master
pybind, ``sherpa-onnx/python/csrc/offline-stream.cc``):

    result.text            -- recognized text
    result.tokens          -- List[str], the emitted (non-blank) tokens
    result.timestamps      -- per-token times
    result.ys_log_probs    -- List[float], per-token ACOUSTIC log-probabilities
                              (older/alternate builds name this `ys_probs`;
                              :func:`_extract_ys_log_probs` reads either)

``ys_log_probs`` is aligned 1:1 with ``tokens`` and IS exactly s1's per-token
log-posterior source (s1 = mean, s2 = exp(min) -- ``asr_gate.scores``). For the
targeted Chinese transducer models the modeling unit is char-level, so each
token is a single CJK character and ``token_logps`` is naturally
character-granular (the same convention ``decode_paraformer.py`` /
``decode_whisper.py`` rely on -- Mandarin is effectively character-tokenized).
English word-starts, when present, carry the k2/icefall word-boundary marker
``"▁"`` (``▁``); :func:`_clean_text` strips it (and any whitespace) for a
clean ``hyp_text`` consistent with ``asr_gate.cer``'s normalizer, while
``token_logps`` stays 1:1 with the RAW tokens (scores consume per-token logps,
not characters).

DEGRADE-GRACEFULLY contract (mirrors the other decoders): if ``ys_log_probs``
is absent/empty, or its length does not match ``tokens`` (an unexpected build
change), ``token_logps`` degrades to ``None`` for that utterance (s1/s2 become
``None`` downstream, never silently misaligned) with a stderr warning -- the
decode never crashes. **The real ``ys_log_probs`` population under the
transducer decoding mode is verified on-box in the ``--probe`` smoke BEFORE the
full decode is trusted (FREEZE-AMENDMENT-2026-07-13.md §1); this script's own
tests verify the token->char alignment LOGIC on a synthetic fixture only.**

Record schema mirrors the other decoders exactly:
``{utt_id, speaker_id, duration_s, hyp_text, nbest:[{text, logp, token_logps}],
ref_text, gender, region}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/asr-gate/ root
from asr_gate import corpora  # noqa: E402

_WORD_BOUNDARY = "▁"  # k2/icefall BPE word-start marker "▁"


def _clean_text(text: str) -> str:
    """Clean decoded text for ``hyp_text``: drop the k2 word-boundary marker
    ``▁`` and strip all whitespace -- matching ``decode_paraformer.py``'s
    ``_strip_char_spacing`` convention (Mandarin refs are space-free; the
    ``asr-gate-cer-normalizer-v1`` strips whitespace regardless, so this is
    consistent with rather than dependent on it)."""
    if not text:
        return text
    return "".join(text.replace(_WORD_BOUNDARY, " ").split())


def _extract_ys_log_probs(result: Any) -> Optional[List[float]]:
    """Per-token acoustic log-probs off a sherpa-onnx ``OfflineRecognizerResult``.
    Reads ``ys_log_probs`` (current master) or ``ys_probs`` (the name the
    COMPUTE-PLAN referenced / older builds) -- whichever is present and
    non-empty. ``None`` if neither is populated."""
    for attr in ("ys_log_probs", "ys_probs"):
        v = getattr(result, attr, None)
        if v:
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None
    return None


def _result_to_nbest(result: Any) -> Dict[str, Any]:
    """Map one sherpa-onnx ``OfflineRecognizerResult`` (or any object exposing
    ``.text`` / ``.tokens`` / ``.ys_log_probs``) to a canonical top-1 nbest
    entry ``{"text", "logp", "token_logps"}``.

    ``hyp_text`` is built from ``tokens`` when available (so word-boundary
    markers are handled), falling back to ``result.text``. ``token_logps`` is
    the per-token ``ys_log_probs`` aligned 1:1 with ``tokens``; it degrades to
    ``None`` (with a stderr warning, never a crash or a silent misalignment) if
    the log-probs are missing or their count does not match ``tokens``.
    ``logp`` is ``sum(token_logps)`` when available (matching the other
    decoders' sequence-level logp)."""
    tokens = list(getattr(result, "tokens", None) or [])
    ys = _extract_ys_log_probs(result)
    raw_text = getattr(result, "text", "") or ""
    hyp_text = _clean_text("".join(tokens)) if tokens else _clean_text(raw_text)

    token_logps: Optional[List[float]] = None
    if ys is not None:
        if tokens and len(ys) != len(tokens):
            print(
                f"warning: sherpa-onnx ys_log_probs length {len(ys)} != tokens "
                f"length {len(tokens)} -- degrading token_logps to None for this "
                f"utterance (text={hyp_text!r})",
                file=sys.stderr,
            )
        else:
            token_logps = ys
    logp = float(sum(token_logps)) if token_logps else None
    return {"text": hyp_text, "logp": logp, "token_logps": token_logps}


def _load_recognizer(args) -> Any:
    """Lazy sherpa-onnx import + offline transducer construction. Only called
    from :func:`main` after argument parsing, so ``--help`` works with no
    sherpa_onnx installed. ``--model-dir`` is a directory holding the exported
    ONNX encoder/decoder/joiner + ``tokens.txt`` (the standard sherpa-onnx
    pretrained-model layout); explicit paths override the auto-resolved ones."""
    try:
        import sherpa_onnx  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "decode_sherpa_onnx.py: sherpa_onnx is not installed in this "
            "environment. Install it on the decode box (`pip install sherpa-onnx` "
            "-- pure-Python wheels, no k2/kaldifeat build) -- this is a box-side "
            "script, not part of the asr-gate package's own dependencies."
        ) from e

    model_dir = Path(args.model_dir)
    encoder = args.encoder or _resolve_one(model_dir, "encoder")
    decoder = args.decoder or _resolve_one(model_dir, "decoder")
    joiner = args.joiner or _resolve_one(model_dir, "joiner")
    tokens = args.tokens or str(model_dir / "tokens.txt")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        debug=False,
    )


def _resolve_one(model_dir: Path, kind: str) -> str:
    """Best-effort single-file resolver for the encoder/decoder/joiner ONNX in a
    sherpa-onnx pretrained model dir (names vary: ``encoder-*.onnx``,
    ``*.int8.onnx``, etc.). Prefers a non-int8 file for numeric fidelity of the
    posteriors; raises an actionable error if 0 or >1 candidates remain (pass
    ``--encoder/--decoder/--joiner`` explicitly then)."""
    cands = sorted(model_dir.glob(f"{kind}*.onnx"))
    if not cands:
        raise SystemExit(
            f"decode_sherpa_onnx.py: no {kind}*.onnx under {model_dir} "
            f"(pass --{kind} explicitly)"
        )
    non_int8 = [c for c in cands if "int8" not in c.name]
    pref = non_int8 or cands
    if len(pref) > 1:
        raise SystemExit(
            f"decode_sherpa_onnx.py: {len(pref)} candidate {kind} ONNX files under "
            f"{model_dir} ({[c.name for c in pref]}) -- pass --{kind} explicitly"
        )
    return str(pref[0])


def _decode_wav(recognizer: Any, wav_path: str):
    """Decode one wav -> ``OfflineRecognizerResult``. Reads audio via soundfile
    (float32, 16 kHz mono expected -- the corpora are all 16 kHz), asserting the
    sample rate so a mismatched corpus fails loudly rather than mis-scoring."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr} Hz ({wav_path})")
    if audio.ndim > 1:  # stereo -> mono (defensive; corpora are mono)
        audio = audio.mean(axis=1)
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio)
    recognizer.decode_stream(stream)
    return stream.result, len(audio) / float(sr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decode_sherpa_onnx.py",
        description="Decode a Mandarin corpus split with a sherpa-onnx offline "
        "zipformer transducer -> canonical asr-gate JSONL (Path D posteriors).",
    )
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument(
        "--corpus", default="aishell", choices=sorted(corpora.CORPUS_DISCOVERERS)
    )
    parser.add_argument(
        "--data-root", required=True,
        help="corpus root (layout resolved by asr_gate.corpora.discover_corpus)",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="sherpa-onnx pretrained model dir (encoder/decoder/joiner ONNX + tokens.txt)",
    )
    parser.add_argument("--encoder", default=None, help="override encoder ONNX path")
    parser.add_argument("--decoder", default=None, help="override decoder ONNX path")
    parser.add_argument("--joiner", default=None, help="override joiner ONNX path")
    parser.add_argument("--tokens", default=None, help="override tokens.txt path")
    parser.add_argument("--decoding-method", default="greedy_search",
                        choices=["greedy_search", "modified_beam_search"])
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--out", required=True, help="output canonical decode JSONL path")
    parser.add_argument("--limit", type=int, default=None,
                        help="decode only the first N utterances (smoke test)")
    parser.add_argument(
        "--probe", type=int, default=None, metavar="N",
        help="PROBE mode: decode N utts, dump raw {text, tokens, ys_log_probs "
             "lengths, token_logps} to stdout + a '<out>.probe.json' sidecar for "
             "on-box posterior-shape verification -- writes NO canonical JSONL",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="if --out exists and is non-empty, do nothing (idempotent)")
    parser.add_argument("--resume", action="store_true",
                        help="skip utt_ids already in --out and append (hang-recovery)")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"decode_sherpa_onnx: {out_path} already exists, skipping (--skip-existing)")
        return 0

    limit = args.probe if args.probe is not None else args.limit
    entries = corpora.discover_corpus(args.corpus, Path(args.data_root), args.split, limit)
    if not entries:
        print(f"error: no utterances for corpus={args.corpus} split={args.split} "
              f"under {args.data_root}", file=sys.stderr)
        return 1

    recognizer = _load_recognizer(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- PROBE mode: raw-shape dump, no canonical output (mirror conformer) ----
    if args.probe is not None:
        probed: List[Dict[str, Any]] = []
        for e in entries:
            try:
                result, _dur = _decode_wav(recognizer, str(e.wav_path))
                tokens = list(getattr(result, "tokens", None) or [])
                ys = _extract_ys_log_probs(result)
                probed.append({
                    "utt_id": e.utt_id,
                    "text": getattr(result, "text", None),
                    "n_tokens": len(tokens),
                    "n_ys_log_probs": (len(ys) if ys is not None else None),
                    "tokens_aligned": (ys is not None and len(ys) == len(tokens)),
                    "sample_tokens": tokens[:8],
                    "sample_ys_log_probs": (ys[:8] if ys is not None else None),
                    "nbest0": _result_to_nbest(result),
                })
            except Exception as exc:  # noqa: BLE001 - probe must not crash on one bad utt
                probed.append({"utt_id": e.utt_id, "error": f"{type(exc).__name__}: {exc}"})
        probe_path = out_path.with_suffix(".probe.json")
        with open(probe_path, "w", encoding="utf-8") as pf:
            json.dump(probed, pf, ensure_ascii=False, indent=2)
        n_aligned = sum(1 for p in probed if p.get("tokens_aligned"))
        print(f"decode_sherpa_onnx: PROBE mode -- decoded {len(probed)} utts, "
              f"{n_aligned}/{len(probed)} with token_logps aligned -> {probe_path}")
        for p in probed:
            print(json.dumps(p, ensure_ascii=False))
        return 0

    # ---- resume: skip utt_ids already written, append ----
    done_ids = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["utt_id"])
            except Exception:
                pass  # torn tail line from a killed run; re-decoded below
        entries = [e for e in entries if e.utt_id not in done_ids]
        print(f"resume: {len(done_ids)} done, {len(entries)} remaining", flush=True)

    n_written = 0
    n_missing_ref = 0
    n_degraded_logp = 0
    skipped: List[Dict[str, str]] = []
    mode = "a" if (args.resume and done_ids) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for e in entries:
            try:
                result, dur = _decode_wav(recognizer, str(e.wav_path))
            except Exception as exc:  # noqa: BLE001 - one bad utt must not kill the run
                skipped.append({"utt_id": e.utt_id, "reason": f"{type(exc).__name__}:{exc}"})
                continue
            nbest0 = _result_to_nbest(result)
            if not nbest0["text"]:
                skipped.append({"utt_id": e.utt_id, "reason": "empty_result"})
                continue
            if nbest0["token_logps"] is None:
                n_degraded_logp += 1
            if e.ref_text is None:
                n_missing_ref += 1
            record = {
                "utt_id": e.utt_id,
                "speaker_id": e.speaker_id,
                "duration_s": dur,
                "hyp_text": nbest0["text"],
                "nbest": [nbest0],
                "ref_text": e.ref_text,
                "gender": None,
                "region": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 250 == 0:
                print(f"[{n_written}]", flush=True)

    if skipped:
        with open(out_path.with_suffix(".skipped.jsonl"), "w", encoding="utf-8") as sf:
            for s in skipped:
                sf.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"decode_sherpa_onnx: corpus={args.corpus} split={args.split} "
          f"n_written={n_written} n_missing_ref={n_missing_ref} "
          f"n_degraded_logp={n_degraded_logp} n_skipped={len(skipped)} -> {out_path}")
    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
