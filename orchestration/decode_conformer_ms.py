#!/usr/bin/env python3
"""Box-side decode script: FunASR/ModelScope autoregressive
Conformer-Aishell (``iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch``,
confirmed downloadable via ``modelscope.snapshot_download`` on the AutoDL box
2026-07-09, ``/root/stage_expand.log``: ``CONFORMER_SNAPSHOT_OK`` -- this is
the one model id in the expansion's staging run that resolved cleanly; see
the expansion report for THCHS-30/MUSAN/Whisper's download status) over an
Aishell-1 or THCHS-30 split, emitting the canonical ``asr_gate`` decode
JSONL directly. Mirrors ``orchestration/decode_paraformer.py``'s
canonical-JSONL-out contract and lazy-import / degrade-gracefully
conventions.

Runs on the AutoDL box, NOT in CI/tests: ``funasr``/``torch`` are imported
lazily inside :func:`_load_model` so ``--help`` works in any environment,
including one with neither installed.

UNVERIFIED result shape
------------------------
Unlike ``decode_paraformer.py`` (whose FunASR result shape and internal
token-logp-hook mechanics were verified empirically against real box
output), this Conformer-Aishell model's real ``AutoModel.generate()``
output shape has NOT been probed against a live model (no box access to a
model that loads in this build environment). This script therefore:

1. Uses a TOLERANT extraction (:func:`_extract_result`, deliberately
   mirroring ``decode_paraformer.py``'s ``_extract_result``) that probes
   several plausible key names for text/score/token-level detail rather
   than assuming one.
2. Does NOT attempt a model-internals monkeypatch hook for token_logps (the
   ``_seaco_decode_with_ASF`` hook is SeaCo-Paraformer-specific internal
   plumbing; guessing at this Conformer AED model's internal method names
   without empirical verification would be worse than admitting the
   limitation) -- ``token_logps`` stays ``None`` (degraded mode: s1/s2/s4
   are ``None``) unless a future verified hook is added here. This is the
   design's §2.2 "backbone exposing only 1-best + token posteriors"
   degraded-mode case, except here even token posteriors are absent --
   flagged explicitly rather than silently accepted.
3. Requests N-best via a best-effort ``beam_size=`` kwarg
   (:func:`_generate`, try/except -- FunASR ``generate()`` kwargs are
   model-family-specific and this AED Conformer's beam-search support is
   unverified); on ``TypeError`` (kwarg rejected) falls back to plain
   ``generate()`` and stays 1-best-only for that utterance (s3 degrades to
   ``None``), with a once-per-run warning.

Verify this script's actual output shape against a real decode call at
Phase 0 (per the design's standing convention) before trusting anything
beyond ``hyp_text``.

Probe mode (``--probe N``, added 2026-07-09 after the box smoke test found
usable=0/3 despite ``n_written=3``: the real result item shape has no score
keys this script's tolerant extraction recognized) dumps the FULL raw
``generate()`` result structure for the first ``N`` utterances, unmodified,
to stdout and to a ``.probe.json`` sidecar next to ``--out`` -- for a live
one-shot shape verification at next boot, no guessing required.
``_extract_result`` also now searches a few plausible NESTED shapes
(``item["nbest"][0]``, ``item["sentence_info"]`` per-token dicts, ct/us
score arrays) in addition to the original flat-key probe, staying tolerant
rather than assuming any one of them is real (see :func:`_search_nested_logp`
/ :func:`_search_nested_token_logps`).
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/asr-gate/ root
from asr_gate import corpora  # noqa: E402

DEFAULT_MODEL_NAME = "iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch"


def _wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception as e:  # noqa: BLE001 - best-effort, never fatal
        print(f"warning: could not read duration of {path}: {e}", file=sys.stderr)
    return 1.0


def _load_model(model_name: str, device: str):
    """Lazy FunASR import -- keeps ``--help`` working with no funasr/torch
    installed. Only called from :func:`main` after argument parsing."""
    try:
        from funasr import AutoModel  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "decode_conformer_ms.py: funasr is not installed in this "
            "environment. Install it on the decode box (`pip install funasr "
            "modelscope`) -- this is a box-side script, not part of the "
            "asr-gate package's own dependencies."
        ) from e
    return AutoModel(model=model_name, device=device)


def _strip_char_spacing(text: str) -> str:
    """Same convention as ``decode_paraformer.py``'s helper of the same
    name -- defensively stripped even though this model's real output
    spacing behavior has not been verified (see module docstring)."""
    if not text:
        return text
    return "".join(text.split())


_SENTENCE_LOGP_KEYS = ("logp", "score", "avg_logprob", "sentence_logp")
_TOKEN_LOGPS_KEYS = ("token_logps", "token_score", "us_score", "token_confidence")
_NESTED_ARRAY_KEYS = ("ctc_score", "us_alphas", "us_scores")
_TOKEN_DICT_SCORE_KEYS = ("score", "avg_logprob", "logp", "confidence")


def _search_nested_logp(item: Dict[str, Any]) -> Optional[float]:
    """Best-effort search for a sentence-level logp on plausible NESTED
    shapes (module docstring point (b)): ``item["nbest"][0]``, itself
    probed with the same flat key aliases as the top level. Returns
    ``None`` (never raises) if nothing recognizable is found."""
    nbest = item.get("nbest")
    if isinstance(nbest, list) and nbest and isinstance(nbest[0], dict):
        for key in _SENTENCE_LOGP_KEYS:
            v = nbest[0].get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _search_nested_token_logps(item: Dict[str, Any]) -> Optional[List[float]]:
    """Best-effort search for a per-token score sequence on plausible
    NESTED shapes (module docstring point (b)):

    - ``item["nbest"][0]``, probed with the same flat per-token key aliases.
    - ``item["sentence_info"]``: a list of per-token dicts, each carrying a
      score under one of a few plausible key names -- ALL tokens must carry
      a recognized key or this degrades to ``None`` rather than returning a
      partial/misaligned sequence.
    - plain arrays under alternate ct/us-branch key names
      (``_NESTED_ARRAY_KEYS``) some FunASR AED variants use.

    Never raises, never guesses at a shape it cannot confirm is a flat
    numeric sequence or a complete list-of-dicts."""
    nbest = item.get("nbest")
    if isinstance(nbest, list) and nbest and isinstance(nbest[0], dict):
        for key in _TOKEN_LOGPS_KEYS:
            v = nbest[0].get(key)
            if isinstance(v, list):
                try:
                    return [float(x) for x in v]
                except (TypeError, ValueError):
                    pass

    sentence_info = item.get("sentence_info")
    if isinstance(sentence_info, list) and sentence_info and all(
        isinstance(tok, dict) for tok in sentence_info
    ):
        vals: List[float] = []
        complete = True
        for tok in sentence_info:
            v = None
            for key in _TOKEN_DICT_SCORE_KEYS:
                if isinstance(tok.get(key), (int, float)):
                    v = float(tok[key])
                    break
            if v is None:
                complete = False
                break
            vals.append(v)
        if complete and vals:
            return vals

    for key in _NESTED_ARRAY_KEYS:
        v = item.get(key)
        if isinstance(v, list):
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                pass

    return None


def _extract_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Tolerant extraction, deliberately mirroring
    ``decode_paraformer.py``'s ``_extract_result`` -- see module docstring
    for why no result shape is assumed verified here. Flat top-level keys
    are tried first; if absent, falls back to plausible NESTED shapes via
    :func:`_search_nested_logp` / :func:`_search_nested_token_logps`."""
    text = _strip_char_spacing(item.get("text", ""))
    if not text:
        print(f"warning: empty/missing 'text' in funasr result item: {item!r}", file=sys.stderr)
    logp = None
    for key in _SENTENCE_LOGP_KEYS:
        if key in item and isinstance(item[key], (int, float)):
            logp = float(item[key])
            break
    if logp is None:
        logp = _search_nested_logp(item)
    token_logps = None
    for key in _TOKEN_LOGPS_KEYS:
        if key in item and isinstance(item[key], list):
            try:
                token_logps = [float(v) for v in item[key]]
            except (TypeError, ValueError):
                token_logps = None
            break
    if token_logps is None:
        token_logps = _search_nested_token_logps(item)
    return {"text": text, "logp": logp, "token_logps": token_logps}


def _generate(
    model, wav_path: Path, nbest: int, warned: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """Best-effort ``beam_size=`` request (see module docstring point 3);
    ``warned`` is a mutable dict used to print the fallback warning at most
    once per run."""
    raw: Any
    if nbest > 1:
        try:
            raw = model.generate(input=str(wav_path), batch_size=1, beam_size=nbest)
        except TypeError:
            if not warned.get("beam_size"):
                print(
                    "warning: model.generate() rejected beam_size= -- falling back to "
                    "1-best decoding for all utterances (s3 will be None)",
                    file=sys.stderr,
                )
                warned["beam_size"] = True
            raw = model.generate(input=str(wav_path), batch_size=1)
    else:
        raw = model.generate(input=str(wav_path), batch_size=1)
    if isinstance(raw, dict):
        raw = [raw]
    return raw or []


def _probe_raw_results(
    model, entries, n: int, warned: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """Decode the first ``n`` entries and return their raw FunASR result
    structures UNMODIFIED (never routed through :func:`_extract_result`) --
    for a live one-shot shape verification, per module docstring. One bad
    utterance is recorded-and-skipped, never fatal to the probe."""
    probed: List[Dict[str, Any]] = []
    for utt in entries[:n]:
        try:
            raw_results = _generate(model, utt.wav_path, 1, warned)
        except Exception as exc:  # noqa: BLE001 - probe must not crash on one bad utterance
            probed.append({"utt_id": utt.utt_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        probed.append(
            {
                "utt_id": utt.utt_id,
                "n_items": len(raw_results),
                "raw_repr": [repr(item)[:2000] for item in raw_results],
            }
        )
    return probed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decode_conformer_ms.py",
        description="Decode a corpus split with FunASR/ModelScope Conformer-Aishell -> canonical asr-gate JSONL.",
    )
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--corpus", default="aishell", choices=sorted(corpora.CORPUS_DISCOVERERS))
    parser.add_argument(
        "--data-root", default=None,
        help="corpus root; default /root/autodl-tmp/data_aishell or "
             "/root/autodl-tmp/data_thchs30 depending on --corpus",
    )
    parser.add_argument("--out", required=True, help="output canonical decode JSONL path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--nbest", type=int, default=1, help="beam width for N-best (best-effort)")
    parser.add_argument("--limit", type=int, default=None, help="decode only the first N utterances")
    parser.add_argument(
        "--probe", type=int, default=None, metavar="N",
        help="PROBE MODE: decode the first N utterances and dump the raw funasr result "
             "structure (repr, truncated ~2000 chars/item) to stdout and to a "
             "'<out>.probe.json' sidecar, for live shape verification -- no canonical "
             "JSONL is written, no _extract_result is applied",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="if --out already exists and is non-empty, do nothing (idempotent pipelines)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"decode_conformer_ms: {out_path} already exists, skipping (--skip-existing)")
        return 0

    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        default_dir = "data_aishell" if args.corpus == "aishell" else "data_thchs30"
        data_root = Path(f"/root/autodl-tmp/{default_dir}")

    limit = args.probe if args.probe is not None else args.limit
    entries = corpora.discover_corpus(args.corpus, data_root, args.split, limit)
    if not entries:
        raise SystemExit(
            f"decode_conformer_ms.py: no utterances discovered for corpus={args.corpus} "
            f"split={args.split} under {data_root}"
        )

    model = _load_model(args.model_name, args.device)
    warned: Dict[str, bool] = {}

    if args.probe is not None:
        probed = _probe_raw_results(model, entries, args.probe, warned)
        probe_path = out_path.with_suffix(".probe.json")
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as pf:
            json.dump(probed, pf, ensure_ascii=False, indent=2)
        print(
            f"decode_conformer_ms: PROBE mode -- decoded {len(probed)} utterances, "
            f"raw result structure below and in {probe_path}"
        )
        for p in probed:
            print(json.dumps(p, ensure_ascii=False))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_missing_ref = 0
    n_with_logp = 0
    n_with_token_logps = 0
    skipped: List[Dict[str, str]] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for utt in entries:
            dur = _wav_duration_s(utt.wav_path)
            if dur < 0.1:
                skipped.append({"utt_id": utt.utt_id, "reason": f"too_short:{dur:.3f}s"})
                continue
            try:
                raw_results = _generate(model, utt.wav_path, args.nbest, warned)
            except Exception as exc:  # noqa: BLE001 - one bad utterance must not kill the run
                skipped.append(
                    {"utt_id": utt.utt_id, "reason": f"generate_error:{type(exc).__name__}:{exc}"}
                )
                continue
            if not raw_results:
                skipped.append({"utt_id": utt.utt_id, "reason": "empty_result"})
                continue
            nbest_items = [_extract_result(r) for r in raw_results[: max(args.nbest, 1)]]
            if utt.ref_text is None:
                n_missing_ref += 1
            if nbest_items[0]["logp"] is not None:
                n_with_logp += 1
            if nbest_items[0]["token_logps"] is not None:
                n_with_token_logps += 1
            record = {
                "utt_id": utt.utt_id,
                "speaker_id": utt.speaker_id,
                "duration_s": dur,
                "hyp_text": nbest_items[0]["text"],
                "nbest": [
                    {"text": h["text"], "logp": h["logp"], "token_logps": h["token_logps"]}
                    for h in nbest_items
                ],
                "ref_text": utt.ref_text,
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
        f"decode_conformer_ms: corpus={args.corpus} split={args.split} n_written={n_written} "
        f"n_missing_ref={n_missing_ref} n_skipped={len(skipped)} n_with_logp={n_with_logp} "
        f"n_with_token_logps={n_with_token_logps} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
