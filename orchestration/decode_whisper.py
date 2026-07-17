#!/usr/bin/env python3
"""Box-side decode script: HF transformers Whisper large-v3 (zero-shot,
``language=zh``) over an Aishell-1 or THCHS-30 split, emitting the canonical
``asr_gate`` decode JSONL directly (mirrors
``orchestration/decode_paraformer.py``'s canonical-JSONL-out contract; see
that script's module docstring for the lazy-import / degrade-gracefully
pattern this one follows).

Runs on the AutoDL box, NOT in CI/tests: ``transformers``/``torch``/
``soundfile`` are imported lazily inside :func:`_load_model` /
:func:`_load_audio` so ``--help`` (and argument parsing) works in ANY
environment, including one with none of them installed -- exercised by
``tests/test_decode_whisper.py::test_decode_whisper_help_runs_without_transformers``.

Corpus layout: dispatches to ``asr_gate.corpora.discover_corpus`` via
``--corpus {aishell,thchs30}``; see that module's docstrings for the exact
assumed on-disk layouts and their VERIFY-at-Phase-0 caveats (THCHS-30's
layout is additionally UNVERIFIED at build time because the box's no-card
download of it FAILED -- see the expansion report).

Transformers-version-adaptive score extraction (found broken 2026-07-09,
AutoDL box smoke test: SMOKE_whisper usable=0/3; the box's installed
transformers rejects ``output_scores``/``return_dict_in_generate`` as
DIRECT ``generate()`` kwargs -- ``"The following generation flags are not
valid and may be ignored: ['output_scores']"`` -- so ``out.scores`` came
back unusable and every row got ``logp=None``)
------------------------------------------------------------------------
:func:`_generate_with_scores` tries, in order: (1) a ``GenerationConfig``
with ``output_scores``/``return_dict_in_generate`` set, using the modern
``language=``/``task=`` generate() kwargs (preferred over the deprecated
``processor.get_decoder_prompt_ids``/``forced_decoder_ids`` path); (2) the
same ``GenerationConfig`` approach falling back to ``forced_decoder_ids``
if ``language=``/``task=`` are rejected (older transformers); (3) legacy
direct kwargs (``output_scores=True`` etc. passed straight to
``generate()``, no ``GenerationConfig`` involved) with
``forced_decoder_ids`` -- covers transformers old enough that this is the
only mechanism that honors the flags at all. An attempt is abandoned (next
one tried) on either a ``TypeError`` (kwarg genuinely rejected) or a
successful call whose ``out.scores`` comes back ``None``/empty (kwarg
silently ignored, the exact failure mode above). If every attempt is
exhausted with no usable scores, :class:`WhisperScoresUnavailableError` is
raised -- this is a SYSTEMATIC transformers-API mismatch, not a one-off
bad utterance, so :func:`main` lets it abort the whole run loudly (a clear
per-run error) rather than silently emitting ``logp=None`` rows.

Token-logp -> character alignment convention (this script's genuinely new
piece -- Whisper's byte-level BPE vocabulary does NOT character-tokenize
Mandarin the way FunASR's SeaCo-Paraformer vocab effectively does; see
``decode_paraformer.py``'s module docstring for that contrast)
------------------------------------------------------------------------
Once ``out.scores`` is in hand (see above),
``GenerationMixin.compute_transition_scores``
(``normalize_logits=True``) turns these into a per-BPE-TOKEN log-probability
of the CHOSEN token, aligned 1:1 with ``outputs.sequences``. A BPE token can
decode to zero, one, or several CJK characters (Chinese text does not align
to whitespace/subword boundaries the way this vocabulary was built), so a
further alignment step (:func:`_align_token_logps_to_chars`) is needed to
produce the canonical schema's per-CHARACTER ``token_logps``:

1. Decode the token sequence incrementally (``tokenizer.decode`` on growing
   prefixes, ``skip_special_tokens=True``); each token's textual
   contribution is whatever NEW characters appear in the decoded string
   after adding it.
2. A token contributing ``k >= 1`` new characters has its logp split
   EQUALLY across those ``k`` characters (keeps
   ``sum(token_logps) == sum(the original per-token logps)`` exactly, so
   s1 -- the character-level mean -- is a faithful redistribution rather
   than an approximation that drops or double-counts probability mass; the
   sequence-level ``logp`` field is computed independently as the RAW
   per-token sum, unaffected by this redistribution).
3. A token contributing 0 new characters (rare -- e.g. a token whose
   decoded contribution is fully absorbed by a later token, or a
   leading-space/BOM artifact) has its logp carried forward and folded into
   the NEXT character-contributing token's share, so no log-probability
   mass is silently discarded; if the sequence ends on such a token with no
   later character to absorb it, the leftover folds into the LAST
   character instead.

This is a genuine, documented DESIGN CHOICE (a per-BPE-token log-probability
is not itself an established "per-character confidence" -- there is no
canonical convention here), not a verified empirical fact the way
``decode_paraformer.py``'s hook mechanics are. Flag for review before
trusting this alignment as anything beyond "a reasonable score", per the
design's honesty rules.

N-best via beam search: ``--nbest > 1`` requests
``num_beams=num_return_sequences=nbest``; ``compute_transition_scores`` is
called with ``outputs.beam_indices`` in that case (required for correct
per-token attribution across beams, per the transformers API). Hypotheses
are ordered by ``outputs.sequences_scores`` (length-normalized log-prob)
when available, else by summed token logp.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/asr-gate/ root
from asr_gate import corpora  # noqa: E402

_DEFAULT_TASK = "transcribe"


class WhisperScoresUnavailableError(RuntimeError):
    """Raised by :func:`_generate_with_scores` when ``model.generate()``
    does not yield usable per-token scores after every adaptive attempt.
    This is a SYSTEMATIC transformers-API mismatch (see module docstring),
    not a per-utterance decode failure -- callers MUST let it propagate
    and abort the run rather than skip-and-continue like other
    per-utterance errors (:func:`main` does exactly this)."""


def _wav_duration_s(path: Path) -> float:
    """Duration via the stdlib ``wave`` module -- see
    ``decode_paraformer.py``'s identical helper for the fallback rationale."""
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
    """Lazy transformers/torch import -- keeps ``--help`` working with
    neither installed. Only called from :func:`main` after argument
    parsing."""
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as e:
        raise SystemExit(
            "decode_whisper.py: transformers/torch are not installed in this "
            "environment. Install them on the decode box (`pip install "
            "transformers torch soundfile`) -- this is a box-side script, "
            "not part of the asr-gate package's own dependencies."
        ) from e
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor, torch


def _load_audio(wav_path: Path, target_sr: int = 16000):
    """Mono float32 waveform at ``target_sr``. Lazy ``soundfile`` import
    (keeps ``--help`` working with no soundfile installed). Resamples via
    plain linear interpolation (no extra dependency beyond numpy) if the
    source rate differs -- Aishell-1/THCHS-30 are both natively 16kHz per
    their corpus specs, so this path is expected to be a rare fallback, not
    the common case; documented as an approximation, not a claim of
    high-fidelity resampling."""
    import numpy as np  # lazy
    import soundfile as sf  # type: ignore  # lazy

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr and len(data) > 0:
        duration = len(data) / sr
        n_target = max(int(round(duration * target_sr)), 1)
        x_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
        data = np.interp(x_new, x_old, data).astype("float32")
        sr = target_sr
    return data, sr


def _align_token_logps_to_chars(
    tokenizer, token_ids: List[int], token_logps: List[float]
) -> Tuple[str, Optional[List[float]]]:
    """See module docstring for the alignment convention. Returns
    ``(text, char_logps_or_None)``; ``char_logps`` is ``None`` only if the
    decoded text ends up empty (nothing to align)."""
    running_text = ""
    char_logps: List[float] = []
    pending = 0.0
    seen: List[int] = []
    for tid, lp in zip(token_ids, token_logps):
        seen.append(tid)
        new_text = tokenizer.decode(seen, skip_special_tokens=True)
        delta = new_text[len(running_text):]
        n_new = len(delta)
        total = lp + pending
        if n_new == 0:
            pending = total
            continue
        pending = 0.0
        share = total / n_new
        char_logps.extend([share] * n_new)
        running_text = new_text
    if pending and char_logps:
        char_logps[-1] += pending
    if not running_text:
        return running_text, None
    return running_text, char_logps


def _generate_with_scores(
    model,
    processor,
    input_features,
    language: str,
    task: str,
    num_beams: int,
    # English-arm fix (2026-07-12): 256 truncated LibriSpeech tails (median
    # CER 13% from cut hypotheses); env-overridable, Mandarin default kept.
    max_new_tokens: int = int(os.environ.get("WHISPER_MAX_NEW_TOKENS", "256")),
):
    """Transformers-version-adaptive scored generation -- see module
    docstring for the three-attempt strategy and why an attempt is
    abandoned (``TypeError``, or a call that succeeds but returns no
    usable ``out.scores``). Raises :class:`WhisperScoresUnavailableError`
    (never returns ``None``) if no attempt produces usable scores."""
    try:
        from transformers import GenerationConfig  # type: ignore
    except ImportError:
        GenerationConfig = None  # type: ignore

    attempts: List[str] = []

    def _try(label: str, call) -> Optional[Any]:
        try:
            out = call()
        except TypeError as e:
            attempts.append(f"{label}: TypeError: {e}")
            return None
        if getattr(out, "scores", None):
            return out
        attempts.append(f"{label}: generate() returned no usable out.scores")
        return None

    if GenerationConfig is not None:
        base_config = getattr(model, "generation_config", None)
        gen_config = copy.deepcopy(base_config) if base_config is not None else GenerationConfig()
        gen_config.output_scores = True
        gen_config.return_dict_in_generate = True
        gen_config.num_beams = num_beams
        gen_config.num_return_sequences = num_beams
        gen_config.max_new_tokens = max_new_tokens

        out = _try(
            "generation_config+language/task",
            lambda: model.generate(
                input_features, generation_config=gen_config, language=language, task=task,
            ),
        )
        if out is not None:
            return out

        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        except Exception:
            forced_decoder_ids = None
        out = _try(
            "generation_config+forced_decoder_ids",
            lambda: model.generate(
                input_features, generation_config=gen_config, forced_decoder_ids=forced_decoder_ids,
            ),
        )
        if out is not None:
            return out

    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    except Exception:
        forced_decoder_ids = None
    out = _try(
        "legacy direct kwargs",
        lambda: model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
        ),
    )
    if out is not None:
        return out

    raise WhisperScoresUnavailableError(
        "model.generate() did not return usable per-token scores after all "
        "adaptive attempts (generation_config-based output_scores, "
        "language=/task=, forced_decoder_ids, and legacy direct kwargs). "
        "Attempts tried: " + "; ".join(attempts)
    )


def _decode_utterance(
    model, processor, torch, wav_path: Path, device: str, language: str, nbest: int
) -> List[Dict[str, Any]]:
    """Returns a ranked ``nbest`` list of ``{"text", "logp", "token_logps"}``
    for one utterance (see module docstring for the alignment convention,
    the version-adaptive score-extraction strategy, and beam-search
    mechanics)."""
    audio, sr = _load_audio(wav_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    # match the checkpoint's parameter dtype: the staged cache keeps only the
    # fp16 safetensors variant (disclosed in EXPANSION-AMENDMENT-2026-07-09),
    # and half-precision weights reject float32 features at the first conv.
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    num_beams = max(nbest, 1)
    with torch.no_grad():
        out = _generate_with_scores(
            model, processor, input_features, language, _DEFAULT_TASK, num_beams,
        )

    beam_indices = getattr(out, "beam_indices", None) if num_beams > 1 else None
    transition_scores = model.compute_transition_scores(
        out.sequences, out.scores, beam_indices, normalize_logits=True
    )
    seq_scores = getattr(out, "sequences_scores", None)

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    hyps: List[Dict[str, Any]] = []
    for i in range(out.sequences.shape[0]):
        ids_all = [int(t) for t in out.sequences[i].tolist()]
        scores = [float(s) for s in transition_scores[i].tolist()]
        # ``sequences`` = forced/prompt prefix + GENERATED suffix, but
        # ``transition_scores`` covers only the generated suffix. The old
        # ``zip(ids, scores)`` paired prompt ids with generated-token scores
        # and silently DROPPED the last len(prefix) generated tokens -- every
        # hypothesis lost its final 1-2 words (found 2026-07-12 on
        # LibriSpeech tails; Mandarin whisper arms carry the same bias --
        # disclosed in the red-team remediation). Align to the suffix.
        ids = ids_all[len(ids_all) - len(scores):]
        filtered = [(tid, sc) for tid, sc in zip(ids, scores) if tid != pad_id]
        if not filtered:
            hyps.append({"text": "", "logp": None, "token_logps": None})
            continue
        f_ids = [t for t, _ in filtered]
        f_scores = [s for _, s in filtered]
        text, char_logps = _align_token_logps_to_chars(tokenizer, f_ids, f_scores)
        logp = float(sum(f_scores))
        hyps.append({"text": text, "logp": logp, "token_logps": char_logps})

    if seq_scores is not None and len(seq_scores) == len(hyps):
        order = sorted(range(len(hyps)), key=lambda i: -float(seq_scores[i]))
        hyps = [hyps[i] for i in order]
    else:
        hyps = sorted(
            hyps, key=lambda h: -(h["logp"] if h["logp"] is not None else float("-inf"))
        )
    return hyps


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decode_whisper.py",
        description="Decode a corpus split with HF transformers Whisper large-v3 -> canonical asr-gate JSONL.",
    )
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--corpus", default="aishell", choices=sorted(corpora.CORPUS_DISCOVERERS))
    parser.add_argument(
        "--data-root", default=None,
        help="corpus root; default /root/autodl-tmp/data_aishell or "
             "/root/autodl-tmp/data_thchs30 depending on --corpus",
    )
    parser.add_argument("--out", required=True, help="output canonical decode JSONL path")
    parser.add_argument("--resume", action="store_true",
                        help="skip utt_ids already present in --out and append (crash recovery)")
    parser.add_argument("--model-name", default="openai/whisper-large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--nbest", type=int, default=1, help="beam width for N-best (if affordable)")
    parser.add_argument("--limit", type=int, default=None, help="decode only the first N utterances")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="if --out already exists and is non-empty, do nothing (idempotent pipelines)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"decode_whisper: {out_path} already exists, skipping (--skip-existing)")
        return 0

    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        default_dir = "data_aishell" if args.corpus == "aishell" else "data_thchs30"
        data_root = Path(f"/root/autodl-tmp/{default_dir}")

    entries = corpora.discover_corpus(args.corpus, data_root, args.split, args.limit)
    if not entries:
        raise SystemExit(
            f"decode_whisper.py: no utterances discovered for corpus={args.corpus} "
            f"split={args.split} under {data_root}"
        )

    model, processor, torch = _load_model(args.model_name, args.device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_missing_ref = 0
    skipped: List[Dict[str, str]] = []
    done_ids = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["utt_id"])
            except Exception:
                pass  # torn tail line; re-decoded
        entries = [u for u in entries if u.utt_id not in done_ids]
        print(f"resume: {len(done_ids)} done, {len(entries)} remaining", flush=True)
    with open(out_path, "a" if (args.resume and done_ids) else "w", encoding="utf-8") as f:
        for utt in entries:
            dur = _wav_duration_s(utt.wav_path)
            if dur < 0.1:
                skipped.append({"utt_id": utt.utt_id, "reason": f"too_short:{dur:.3f}s"})
                continue
            try:
                hyps = _decode_utterance(
                    model, processor, torch, utt.wav_path, args.device, args.language, args.nbest
                )
            except WhisperScoresUnavailableError as exc:
                # SYSTEMATIC transformers-API issue (see module docstring),
                # not a one-off bad utterance -- must not be swallowed into
                # a per-utterance skip like the generic handler below.
                raise SystemExit(
                    f"decode_whisper.py: ABORTING at utt_id={utt.utt_id} -- {exc}\n"
                    "This is a systematic score-extraction failure (transformers-version "
                    "mismatch), not a per-utterance decode error; every prior row in "
                    f"{out_path} is now suspect/incomplete. Fix the generation-config "
                    "mechanics (see decode_whisper.py's module docstring) before re-running."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - one bad utterance must not kill the run
                skipped.append(
                    {"utt_id": utt.utt_id, "reason": f"generate_error:{type(exc).__name__}:{exc}"}
                )
                continue
            if not hyps:
                skipped.append({"utt_id": utt.utt_id, "reason": "empty_result"})
                continue
            if utt.ref_text is None:
                n_missing_ref += 1
            record = {
                "utt_id": utt.utt_id,
                "speaker_id": utt.speaker_id,
                "duration_s": dur,
                "hyp_text": hyps[0]["text"],
                "nbest": hyps,
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
        f"decode_whisper: corpus={args.corpus} split={args.split} n_written={n_written} "
        f"n_missing_ref={n_missing_ref} n_skipped={len(skipped)} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
