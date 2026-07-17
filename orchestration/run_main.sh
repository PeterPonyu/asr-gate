#!/usr/bin/env bash
# Main-run SKELETON (design §7 Phase 3): decode B1 (WeNet) + B3 (Whisper,
# exploratory) once each, 20 calibration reseeds against a FROZEN config,
# official Aishell-1 test evaluated exactly once. Structure only -- this is
# NOT meant to run unattended before the design freeze (§7 Phase 2: fix
# m=10 roster, alpha/delta targets, strata-cut rule, normalizer version;
# K6 scoop re-scan; freeze note written).
#
# Guarded by REQUIRES_FREEZE: set REQUIRES_FREEZE=confirmed (after the
# design-freeze checklist above is actually done) to let this script run
# past the guard. Anything else aborts immediately -- this script must
# never accidentally touch the official test split pre-freeze.

set -euo pipefail

if [ "${REQUIRES_FREEZE:-}" != "confirmed" ]; then
  cat >&2 <<'EOF'
run_main.sh: REFUSING TO RUN.

This is the MAIN run against Aishell-1's official test split -- it may
only run after the design freeze (§7 Phase 2): m=10 score/backbone roster
fixed, alpha/delta targets fixed, Mondrian strata-cut rule fixed,
normalizer version pinned, K6 scoop re-scan done, freeze note written.

Set REQUIRES_FREEZE=confirmed to proceed once that checklist is actually
complete. This is a structure-only skeleton either way -- read it before
trusting it with the real test set.
EOF
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data_aishell}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../main_results}"
DEVICE="${DEVICE:-cuda}"
ALPHA="${ALPHA:?set ALPHA to the frozen target, e.g. 0.02}"
DELTA="${DELTA:-0.1}"
N_RESEEDS="${N_RESEEDS:-20}"

mkdir -p "$RESULTS_DIR"

echo "== decode test (B1 WeNet placeholder + B3 Whisper exploratory; wire up per frozen config) =="
echo "NOTE: decode_paraformer.py only covers B2 (FunASR paraformer-zh, the pilot"
echo "backbone). B1 (WeNet)/B3 (Whisper) decode scripts are frozen-config work,"
echo "not written here -- add wenet/whisper decode entry points mirroring"
echo "decode_paraformer.py's canonical-JSONL-out contract before running this."

if [ ! -s "$RESULTS_DIR/decode_test_b2.jsonl" ]; then
  python3 "$SCRIPT_DIR/decode_paraformer.py" --split test --data-root "$DATA_ROOT" \
    --model-name paraformer-zh --device "$DEVICE" --out "$RESULTS_DIR/decode_test_b2.jsonl"
fi

echo "== ingest + score test (once; official Aishell-1 test, never re-touched) =="
if [ ! -s "$RESULTS_DIR/test_canonical.jsonl" ]; then
  asr-gate ingest --hyps "$RESULTS_DIR/decode_test_b2.jsonl" --format custom-schema \
    --out "$RESULTS_DIR/test_canonical.jsonl"
fi
if [ ! -s "$RESULTS_DIR/test_scored.jsonl" ]; then
  asr-gate score --instances "$RESULTS_DIR/test_canonical.jsonl" --out "$RESULTS_DIR/test_scored.jsonl"
fi

echo "== $N_RESEEDS calibration reseeds (dev cal/tune carve; no re-decode per reseed) =="
for seed in $(seq 0 $((N_RESEEDS - 1))); do
  reseed_dir="$RESULTS_DIR/reseed_$seed"
  mkdir -p "$reseed_dir"
  if [ -s "$reseed_dir/gate.json" ]; then
    echo "skip reseed $seed (exists)"
    continue
  fi
  # Re-carve dev speaker-disjoint cal/tune at this seed (dev decode is
  # cached/shared across reseeds -- see run_pilot.sh's decode_dev.jsonl).
  dev_canonical="$RESULTS_DIR/../pilot_results/dev_canonical.jsonl"
  if [ ! -s "$dev_canonical" ]; then
    echo "error: expected dev decode/ingest output at $dev_canonical" \
      "(run run_pilot.sh's decode+ingest steps first -- dev is decoded ONCE, reused across reseeds)" >&2
    exit 1
  fi
  python3 "$SCRIPT_DIR/split_cal_tune.py" \
    "$dev_canonical" "$reseed_dir/cal20.jsonl" "$reseed_dir/tune20.jsonl" "$seed"
  # Score-before-calibrate (the pilot's missing-step lesson, applied here);
  # strata = duration_tercile ONLY: decode records carry gender=null (the
  # speaker.info join is future work), so a gender axis would degenerate to
  # all-"gender_unseen" -- frozen out, disclosed in the freeze note.
  asr-gate score --instances "$reseed_dir/cal20.jsonl" --out "$reseed_dir/cal20_scored.jsonl"
  asr-gate calibrate --instances "$reseed_dir/cal20_scored.jsonl" \
    --alpha "$ALPHA" --delta "$DELTA" --guarantee ltt --strata duration_tercile \
    --seed "$seed" --out "$reseed_dir/gate.json"
  asr-gate apply --gate "$reseed_dir/gate.json" --instances "$RESULTS_DIR/test_scored.jsonl" \
    --out "$reseed_dir/applied_test.json"
done

echo "== audit on official test (ONCE; s-scores vs analytic random-deferral null) =="
if [ ! -s "$RESULTS_DIR/audit_test.json" ]; then
  asr-gate audit --instances "$RESULTS_DIR/test_scored.jsonl" --n-perm 2000 --alpha 0.05 \
    --seed 0 --out "$RESULTS_DIR/audit_test.json"
fi

echo "== K1-K5 evaluation (design §4) still requires manual review of the"
echo "   $N_RESEEDS reseed outputs under $RESULTS_DIR/reseed_*/ -- this script"
echo "   does not auto-adjudicate the kill criteria."
echo "main run skeleton complete -> $RESULTS_DIR"
