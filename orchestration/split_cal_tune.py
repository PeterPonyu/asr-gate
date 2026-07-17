#!/usr/bin/env python3
"""Speaker-disjoint 50/50 carve of a canonical utterance JSONL, used by
``run_pilot.sh`` to build dev's ``cal20``/``tune20`` split. A thin CLI
wrapper over :func:`asr_gate.gate.split_by_speaker` (never splits a
speaker's utterances across both halves -- design §3.1(a)).

Usage: split_cal_tune.py SRC.jsonl CAL_OUT.jsonl TUNE_OUT.jsonl [SEED]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/asr-gate/ root

from asr_gate import gate, io  # noqa: E402


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    src, cal_out, tune_out = argv[:3]
    seed = int(argv[3]) if len(argv) == 4 else 0

    utterances = io.load_utterances(src)
    # Exclude-and-count no-ref utterances BEFORE the carve: calibrate/audit
    # refuse on any record without a reference (design §2.5 no-reference
    # rule), and Aishell-1's dev split really does contain 3 decodable wavs
    # with no transcript line (verified: 5 untranscribed dev wavs total, 2 of
    # them zero-frame and already skipped at decode). Exclusion is loud,
    # never silent — the count is part of the pilot record.
    n_total = len(utterances)
    utterances = [u for u in utterances if u.get("ref_text")]
    n_noref = n_total - len(utterances)
    if n_noref:
        print(f"split_cal_tune: excluded {n_noref}/{n_total} no-ref utterance(s) "
              f"from the cal/tune carve (usable only in ref-free `apply` mode)")
    cal, tune = gate.split_by_speaker(utterances, frac=0.5, seed=seed)
    io.write_jsonl(cal_out, cal)
    io.write_jsonl(tune_out, tune)
    print(
        f"cal: speakers={len({u['speaker_id'] for u in cal})} n={len(cal)} -> {cal_out}\n"
        f"tune: speakers={len({u['speaker_id'] for u in tune})} n={len(tune)} -> {tune_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
