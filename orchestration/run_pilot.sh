#!/usr/bin/env bash
# Phase-1 pilot (design §7): full loop on dev ONLY, one backbone (B2 / FunASR
# paraformer-zh) -- decode dev -> ingest -> score -> calibrate (cal 20 spk)
# -> certify+audit on pseudo-test (tune 20 spk) -> report. Every step is
# idempotent (skips if its output already exists) so the pilot can be
# re-run/resumed freely on the AutoDL box.
#
# Terminology note (read before touching this script): design §2.3 uses
# "tune" for the split s5/G2 are FIT on, calibrated/certified on a
# DIFFERENT "cal" split. Design §7's Phase-1 pilot ALSO calls one of its two
# 20-speaker dev-carves "tune", but uses it as a PSEUDO-TEST evaluation
# split (the real Aishell-1 test set is off-limits pre-freeze), not as the
# §2.3 fit split. To keep these two uses from colliding, this script:
#   - runs `asr-gate calibrate` on `cal20.jsonl` ALONE (no `--tune` flag),
#     letting it auto-split cal20 internally (fit_frac) into the
#     leakage-safe fit/conformalize pair per §2.3;
#   - only ever points `apply`/`audit` (never `calibrate`) at
#     `tune20.jsonl`, exactly matching §7's "certify + audit on tune"
#     wording.
#
# Usage: DATA_ROOT=/root/autodl-tmp/data_aishell RESULTS_DIR=./pilot_results \
#        DEVICE=cuda ALPHA=0.02 DELTA=0.1 ./run_pilot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data_aishell}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../pilot_results}"
DEVICE="${DEVICE:-cuda}"
ALPHA="${ALPHA:-0.02}"
DELTA="${DELTA:-0.1}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"

mkdir -p "$RESULTS_DIR"

step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }

# 1. Decode dev (GPU step; box-side, requires funasr).
step "1/7 decode dev"
if [ ! -s "$RESULTS_DIR/decode_dev.jsonl" ]; then
  limit_arg=()
  if [ -n "$LIMIT" ]; then limit_arg=(--limit "$LIMIT"); fi
  python3 "$SCRIPT_DIR/decode_paraformer.py" --split dev --data-root "$DATA_ROOT" \
    --model-name paraformer-zh --device "$DEVICE" \
    --out "$RESULTS_DIR/decode_dev.jsonl" "${limit_arg[@]}"
else
  skip "$RESULTS_DIR/decode_dev.jsonl"
fi

# 2. ingest (validates the canonical shape decode_paraformer.py already emits).
step "2/7 ingest"
if [ ! -s "$RESULTS_DIR/dev_canonical.jsonl" ]; then
  asr-gate ingest --hyps "$RESULTS_DIR/decode_dev.jsonl" --format custom-schema \
    --out "$RESULTS_DIR/dev_canonical.jsonl"
else
  skip "$RESULTS_DIR/dev_canonical.jsonl"
fi

# 3. speaker-disjoint cal/tune carve of dev (20/20 speakers), seeded.
step "3/7 cal/tune speaker carve (speaker-disjoint, never by utterance)"
if [ ! -s "$RESULTS_DIR/cal20.jsonl" ] || [ ! -s "$RESULTS_DIR/tune20.jsonl" ]; then
  python3 "$SCRIPT_DIR/split_cal_tune.py" \
    "$RESULTS_DIR/dev_canonical.jsonl" "$RESULTS_DIR/cal20.jsonl" "$RESULTS_DIR/tune20.jsonl" "$SEED"
else
  skip "$RESULTS_DIR/cal20.jsonl / tune20.jsonl"
fi

# 3b. score both halves (s1-s4 + CER; the step the header always promised —
#     it was missing from the original script, found by the first real pilot
#     run: calibrate refuses unscored input by design).
step "3b/7 score cal20 + tune20"
if [ ! -s "$RESULTS_DIR/cal20_scored.jsonl" ]; then
  asr-gate score --instances "$RESULTS_DIR/cal20.jsonl" --out "$RESULTS_DIR/cal20_scored.jsonl"
else
  skip "$RESULTS_DIR/cal20_scored.jsonl"
fi
if [ ! -s "$RESULTS_DIR/tune20_scored.jsonl" ]; then
  asr-gate score --instances "$RESULTS_DIR/tune20.jsonl" --out "$RESULTS_DIR/tune20_scored.jsonl"
else
  skip "$RESULTS_DIR/tune20_scored.jsonl"
fi

# 4. calibrate on cal20 ALONE (auto-split internally; see terminology note above).
step "4/7 calibrate (G1 LTT + G2 Mondrian) on cal20"
if [ ! -s "$RESULTS_DIR/gate.json" ]; then
  asr-gate calibrate --instances "$RESULTS_DIR/cal20_scored.jsonl" \
    --alpha "$ALPHA" --delta "$DELTA" --guarantee ltt \
    --strata duration_tercile --seed "$SEED" --out "$RESULTS_DIR/gate.json"
else
  skip "$RESULTS_DIR/gate.json"
fi

# 5. certify: apply the gate to tune20 (pseudo-test) and record coverage.
step "5/7 apply (certify) on tune20"
if [ ! -s "$RESULTS_DIR/applied_tune20.json" ]; then
  asr-gate apply --gate "$RESULTS_DIR/gate.json" --instances "$RESULTS_DIR/tune20_scored.jsonl" \
    --out "$RESULTS_DIR/applied_tune20.json"
else
  skip "$RESULTS_DIR/applied_tune20.json"
fi

# 6. audit: s1-s5 vs analytic random-deferral null, Holm, on tune20.
step "6/7 audit on tune20"
if [ ! -s "$RESULTS_DIR/audit_tune20.json" ]; then
  asr-gate audit --instances "$RESULTS_DIR/tune20_scored.jsonl" --n-perm 2000 --alpha 0.05 \
    --seed "$SEED" --out "$RESULTS_DIR/audit_tune20.json"
else
  skip "$RESULTS_DIR/audit_tune20.json"
fi

# 7. compact report (F1-F3 inputs live in gate.json/audit_tune20.json; figure
#    scripts consume those result JSONs directly, per design §5's provenance
#    -stamped-JSON-first convention).
step "7/7 report"
asr-gate report --audit "$RESULTS_DIR/audit_tune20.json" --gate "$RESULTS_DIR/gate.json" \
  -o "$RESULTS_DIR/pilot_report.md"

echo "pilot complete -> $RESULTS_DIR"
echo "K1-K3 (design §4) must still be checked by hand against gate.json/audit_tune20.json."
