#!/usr/bin/env python3
"""build_primewords_pool.py -- freeze the Primewords (openslr-47) evaluation
pool for the TASLP round-2 external-replication OOS violation check, and stage a
clean pool root the shared ``asr_gate.corpora.discover_primewords`` returns IN
FULL (so ``--corpus primewords --data-root {STAGE}`` decodes exactly the frozen
pool, no ``--limit`` truncation that would break speaker stratification).

===========================================================================
FROZEN SAMPLING RULE  --  PRIMEWORDS-POOL-2026-07-16  (frozen BEFORE any decode)
===========================================================================
This is a confirmatory-style check: the pool is fixed by a deterministic rule
dated 2026-07-16, before the corpus is decoded, and never tuned to the result.

Primewords set1 is a single undivided release (~50k utts, ~296 speakers, read/
prompted Mandarin recorded on mobile) with an explicit per-utterance ``user_id``
(speaker) in ``set1_transcript.json``. The whole corpus is out-of-sample by
construction: the accept/defer gates are aishell-dev-calibrated and never touch
Primewords. The rule builds a speaker-stratified, speaker-balanced pool:

  1. Parse set1_transcript.json. Drop a record if: its wav is absent under
     audio_files/, its text is empty after whitespace-strip, or it has no
     user_id. Duplicate records sharing a wav file (the official distribution
     carries 518 such duplicates among 50,902 records) are dropped keeping the
     FIRST occurrence in file order -- deterministic, part of the frozen rule's
     data-hygiene step (added 2026-07-16 when the staging count gate caught 4
     duplicate draws; the rule's intent was always distinct utterances).
  2. Group surviving utterances by user_id. A speaker is ELIGIBLE iff it has
     >= CAP_PER_SPK (=8) utterances.
  3. Order eligible speakers by ascending user_id. Take the first
     N_SPK_CAP = ceil(TARGET / CAP_PER_SPK) = ceil(2500/8) = 313 of them
     (fewer if <313 are eligible -- then all eligible are used).
  4. For each chosen speaker, draw CAP_PER_SPK utterances WITHOUT replacement
     using a single random.Random(SEED=0) consumed in speaker order, from that
     speaker's utt_ids sorted ascending; keep the drawn utt_ids sorted.
  5. The frozen pool = the union of those per-speaker draws. Final size is
     CAP_PER_SPK * min(313, #eligible speakers) -- ~2,368-2,504 utts, i.e.
     the requested ~2,000-3,000 window, and a WHOLE number of speaker strata
     (so the paper's speaker-level bootstrap-by-speaker CIs stay clean).

The draw is fully deterministic given (SEED, the eligible-speaker ordering, each
speaker's sorted utt list); re-running reproduces the identical pool. The rule
does NOT look at CER, decode output, or any gate -- it is frozen data selection.

Staging: symlink each pool wav into {STAGE}/primewords_md_2018_set1/audio_files/
pool/ (flat is fine -- the discoverer rglobs) and write a FILTERED
set1_transcript.json holding only the pool records. A pool manifest + digest
(seed, caps, speaker/utt counts, per-speaker utt lists) is written for audit.

Usage:
  python3 build_primewords_pool.py --src   /root/autodl-tmp/primewords_raw \
                                   --stage /root/autodl-tmp/primewords_pool \
                                   --manifest /root/autodl-tmp/primewords_pool/POOL_MANIFEST.json
  python3 build_primewords_pool.py --print-rule   # print frozen params as JSON, touch nothing
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Frozen rule constants (PRIMEWORDS-POOL-2026-07-16). Env-overridable ONLY via
# CLI flags below; the defaults ARE the frozen rule and must not change.
RULE_ID = "PRIMEWORDS-POOL-2026-07-16"
SEED_DEFAULT = 0
CAP_PER_SPK_DEFAULT = 8
TARGET_DEFAULT = 2500


def _base_dir(root: Path) -> Path:
    """The dir directly holding set1_transcript.json / audio_files -- either
    {root}/primewords_md_2018_set1 (raw extraction) or {root} itself."""
    for cand in (root / "primewords_md_2018_set1", root):
        if (cand / "set1_transcript.json").exists() or (cand / "audio_files").is_dir():
            return cand
    return root / "primewords_md_2018_set1"


def load_records(transcript_path: Path):
    """[(utt_id, user_id, ref_text), ...] from set1_transcript.json, ref_text
    whitespace-stripped to the char-level convention. Skips malformed records."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    seen = set()
    for rec in data:
        if not isinstance(rec, dict):
            continue
        file_field = rec.get("file")
        user_id = rec.get("user_id")
        text = rec.get("text")
        if not file_field or user_id is None or text is None:
            continue
        utt_id = str(file_field).split(".")[0]
        # 518 duplicate records share a wav in the official transcript; keep
        # the first valid occurrence in file order (deterministic).
        if utt_id in seen:
            continue
        ref_text = "".join(str(text).split())
        if not ref_text:
            continue
        seen.add(utt_id)
        out.append((utt_id, str(user_id), ref_text))
    return out


def build_stem_index(audio_dir: Path):
    """{stem: wav_path} over the (sharded) audio tree via recursive glob."""
    idx = {}
    if audio_dir.is_dir():
        for wav_path in audio_dir.rglob("*.wav"):
            idx.setdefault(wav_path.stem, wav_path)
    return idx


def select_pool(records, stem_index, seed, cap_per_spk, target):
    """Apply the frozen PRIMEWORDS-POOL rule. Returns (selected, meta) where
    selected is [(utt_id, user_id, ref_text, wav_path), ...] sorted by
    (user_id, utt_id) and meta is the digest dict."""
    # group by speaker, keeping only utts whose wav exists
    by_spk = {}
    n_no_wav = 0
    for utt_id, user_id, ref_text in records:
        wav = stem_index.get(utt_id)
        if wav is None:
            n_no_wav += 1
            continue
        by_spk.setdefault(user_id, []).append((utt_id, ref_text, wav))

    eligible = sorted(spk for spk, utts in by_spk.items() if len(utts) >= cap_per_spk)
    n_spk_cap = -(-target // cap_per_spk)  # ceil(target/cap_per_spk)
    chosen_speakers = eligible[:n_spk_cap]

    rng = random.Random(seed)
    selected = []
    per_speaker = {}
    for spk in chosen_speakers:
        utts = sorted(by_spk[spk], key=lambda t: t[0])  # ascending utt_id
        pick_idx = rng.sample(range(len(utts)), cap_per_spk)
        picked = sorted((utts[i] for i in pick_idx), key=lambda t: t[0])
        per_speaker[spk] = [u[0] for u in picked]
        for utt_id, ref_text, wav in picked:
            selected.append((utt_id, spk, ref_text, wav))

    selected.sort(key=lambda t: (t[1], t[0]))
    meta = {
        "rule_id": RULE_ID,
        "seed": seed,
        "cap_per_spk": cap_per_spk,
        "target": target,
        "n_spk_cap": n_spk_cap,
        "n_records_total": len(records),
        "n_records_no_wav": n_no_wav,
        "n_speakers_total": len(by_spk),
        "n_speakers_eligible": len(eligible),
        "n_speakers_selected": len(chosen_speakers),
        "n_pool_utts": len(selected),
        "per_speaker_utts": per_speaker,
    }
    return selected, meta


def stage_pool(selected, stage_root: Path):
    """Symlink pool wavs into {stage}/primewords_md_2018_set1/audio_files/pool/
    and write the FILTERED set1_transcript.json. Idempotent."""
    base = stage_root / "primewords_md_2018_set1"
    pool_audio = base / "audio_files" / "pool"
    pool_audio.mkdir(parents=True, exist_ok=True)
    recs = []
    for utt_id, spk, ref_text, wav in selected:
        link = pool_audio / f"{utt_id}.wav"
        if not link.exists() or (link.is_symlink() and os.readlink(link) != str(wav)):
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(os.path.abspath(wav), link)
        # write text back space-joined per char so the discoverer's whitespace
        # strip is a no-op either way; keep user_id for speaker-level analysis.
        recs.append({"file": f"{utt_id}.wav", "text": ref_text, "user_id": spk})
    with open(base / "set1_transcript.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="raw Primewords extraction root (holds primewords_md_2018_set1/)")
    ap.add_argument("--stage", help="pool root to create (staged clean pool the discoverer returns in full)")
    ap.add_argument("--manifest", default=None, help="write pool digest JSON here (default {stage}/POOL_MANIFEST.json)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--cap-per-spk", type=int, default=CAP_PER_SPK_DEFAULT)
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--print-rule", action="store_true",
                    help="print the frozen rule params as JSON and exit (touches nothing)")
    args = ap.parse_args(argv)

    if args.print_rule:
        print(json.dumps({
            "rule_id": RULE_ID, "seed": args.seed, "cap_per_spk": args.cap_per_spk,
            "target": args.target, "n_spk_cap": -(-args.target // args.cap_per_spk),
        }, indent=2))
        return 0

    if not args.src or not args.stage:
        ap.error("--src and --stage are required unless --print-rule")

    src_base = _base_dir(Path(args.src))
    transcript = src_base / "set1_transcript.json"
    audio_dir = src_base / "audio_files"
    if not transcript.exists():
        print(f"FATAL: transcript not found: {transcript}", file=sys.stderr)
        return 1
    if not audio_dir.is_dir():
        print(f"FATAL: audio_files not found: {audio_dir}", file=sys.stderr)
        return 1

    records = load_records(transcript)
    stem_index = build_stem_index(audio_dir)
    selected, meta = select_pool(records, stem_index, args.seed, args.cap_per_spk, args.target)

    if not selected:
        print("FATAL: frozen rule selected 0 utterances", file=sys.stderr)
        print(json.dumps(meta, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    base = stage_pool(selected, Path(args.stage))

    # content gate: every pool wav symlink resolves; filtered transcript count matches
    n_links = len(list((base / "audio_files" / "pool").glob("*.wav")))
    n_trans = len(json.load(open(base / "set1_transcript.json", encoding="utf-8")))
    meta["n_staged_wav"] = n_links
    meta["n_staged_transcript"] = n_trans
    meta["stage_base"] = str(base)
    ok = (n_links == meta["n_pool_utts"] == n_trans)
    meta["stage_ok"] = ok

    manifest_path = Path(args.manifest) if args.manifest else Path(args.stage) / "POOL_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{RULE_ID}] pool: {meta['n_pool_utts']} utts / "
          f"{meta['n_speakers_selected']} speakers "
          f"(eligible={meta['n_speakers_eligible']}, total={meta['n_speakers_total']}); "
          f"staged wav={n_links} transcript={n_trans}; manifest={manifest_path}")
    if not ok:
        print("POOL_STAGE_GATE_FAIL: staged counts disagree", file=sys.stderr)
        return 1
    print("POOL_STAGE_GATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
