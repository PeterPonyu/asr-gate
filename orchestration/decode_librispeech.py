#!/usr/bin/env python3
"""LibriSpeech (test-clean/test-other) -> canonical asr-gate decode JSONL.
G1 English arm (prereg amendment, signed 2026-07-11). Two backends:

* ``whisper``: REUSES decode_whisper.py's ``_load_model``/``_decode_utterance``
  verbatim (same nbest schema: {text, logp, token_logps}); language="en".
* ``wav2vec2``: facebook/wav2vec2-base-960h CTC. Its vocabulary IS characters,
  so per-character logps are the frame log-probs at (deduped, non-blank)
  emission positions -- the native analogue of decode_whisper's
  ``_align_token_logps_to_chars`` output.

Record schema mirrors decode_whisper exactly:
{utt_id, speaker_id, duration_s, hyp_text, nbest, ref_text, gender, region}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _libri_entries(root: Path, subset: str):
    out = []
    for trans in sorted((root / subset).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            utt_id, ref = line.split(" ", 1)
            spk = utt_id.split("-")[0]
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists():
                out.append({"utt_id": utt_id, "speaker_id": spk,
                            "flac": str(flac), "ref": ref.strip()})
    return out


def _wav2vec2_decode(model, processor, torch, flac: str, device: str) -> List[Dict[str, Any]]:
    import soundfile as sf
    audio, sr = sf.read(flac, dtype="float32")
    assert sr == 16000, sr
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
    ids = logp.argmax(-1)
    blank = model.config.pad_token_id
    chars, char_logps, prev = [], [], None
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    for t in range(ids.shape[0]):
        tid = int(ids[t])
        if tid != blank and tid != prev:
            tok = vocab.get(tid, "")
            chars.append(" " if tok == "|" else tok)
            char_logps.append(float(logp[t, tid]))
        prev = tid
    text = "".join(chars).strip()
    return [{"text": text, "logp": float(sum(char_logps)) if char_logps else None,
             "token_logps": char_logps or None}]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=["whisper", "wav2vec2"])
    p.add_argument("--libri-root", default="/root/autodl-tmp/LibriSpeech")
    p.add_argument("--subset", required=True, choices=["test-clean", "test-other"])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip utt_ids already in --out and append (hang-recovery)")
    args = p.parse_args(argv)

    import torch  # noqa: F401 (device checks below)
    import soundfile as sf

    entries = _libri_entries(Path(args.libri_root), args.subset)
    if not entries:
        print(f"error: no utterances under {args.libri_root}/{args.subset}", file=sys.stderr)
        return 1
    if args.limit:
        entries = entries[: args.limit]

    if args.backend == "whisper":
        import decode_whisper as dw
        model_name = args.model_name or "openai/whisper-large-v3"
        model, processor, _torch = dw._load_model(model_name, args.device)
        decode = lambda flac: dw._decode_utterance(  # noqa: E731
            model, processor, torch, flac, args.device, "en", 1)
    else:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model_name = args.model_name or "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(args.device).eval()
        decode = lambda flac: _wav2vec2_decode(model, processor, torch, flac, args.device)  # noqa: E731

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["utt_id"])
            except Exception:
                pass  # torn tail line from a killed run; re-decoded below
        entries = [e for e in entries if e["utt_id"] not in done_ids]
        print(f"resume: {len(done_ids)} already decoded, {len(entries)} remaining", flush=True)
    skipped: List[Dict[str, str]] = []
    n = 0
    with open(out_path, "a" if (args.resume and done_ids) else "w", encoding="utf-8") as f:
        for e in entries:
            try:
                info = sf.info(e["flac"])
                dur = info.frames / info.samplerate
                hyps = decode(e["flac"])
            except Exception as exc:  # noqa: BLE001 one bad utt must not kill the run
                skipped.append({"utt_id": e["utt_id"], "reason": f"{type(exc).__name__}:{exc}"})
                continue
            if not hyps or not hyps[0].get("text"):
                skipped.append({"utt_id": e["utt_id"], "reason": "empty_result"})
                continue
            rec = {"utt_id": e["utt_id"], "speaker_id": e["speaker_id"],
                   "duration_s": dur, "hyp_text": hyps[0]["text"], "nbest": hyps,
                   "ref_text": e["ref"], "gender": None, "region": None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 250 == 0:
                print(f"[{n}/{len(entries)}]", flush=True)
    if skipped:
        with open(out_path.with_suffix(".skipped.jsonl"), "w") as f:
            for s in skipped:
                f.write(json.dumps(s) + "\n")
    print(f"decode_librispeech: backend={args.backend} subset={args.subset} "
          f"written={n} skipped={len(skipped)} -> {out_path}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
