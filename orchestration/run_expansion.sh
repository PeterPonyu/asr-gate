#!/usr/bin/env bash
# Expansion run SKELETON (EXPANSION-PLAN-2026-07-09.md §2.1): decode
# {aishell,thchs30} x {paraformer,whisper,conformer} clean + aishell x
# {5,15,25}dB MUSAN noise; calibrate-on-aishell-dev / apply-on-{thchs30,
# noisy-aishell} cross-corpus + noise-stratified arms; audit per
# (backbone,corpus) with a roster-derived Holm family (NEVER hardcoded --
# computed from whichever cells actually decoded, mirroring
# FREEZE-NOTE-2026-07-09.md's disclosed-roster precedent for the main run,
# where m dropped from the design's 10 to 2 because B1/WeNet had no
# installable wheel).
#
# Structure only, like run_main.sh -- this is NOT meant to run unattended.
# Guarded by REQUIRES_EXPANSION_FREEZE: an amendment to the design doc
# (apps-design/03-APP-aishell-asr-audit.md) preregistering the expanded
# roster/grid/Holm-family-recomputation rule must land and be confirmed
# before this touches any official test split, per the portfolio's
# preregistration discipline (EXPANSION-PLAN §3 "Experimental core").
#
# Every completion marker below asserts on RESULT CONTENT (row counts,
# non-null hyp_text fraction), never a bare exit code -- the 2026-07-09
# DOFA lesson referenced in EXPANSION-PLAN's header.

set -euo pipefail

if [ "${REQUIRES_EXPANSION_FREEZE:-}" != "confirmed" ]; then
  cat >&2 <<'EOF'
run_expansion.sh: REFUSING TO RUN.

This sketches the multi-backbone/multi-corpus/noise-stratified expansion
grid (EXPANSION-PLAN-2026-07-09.md §2.1). It may only run once:
  1. THCHS-30 + MUSAN + Whisper snapshot downloads succeed on the box (as
     of this build: THCHS_FAILED/MUSAN_FAILED/WHISPER_FAILED in
     /root/stage_expand.log -- a retry is needed before ANY of stages
     B/C/D below can run; only the Conformer-Aishell ModelScope download
     succeeded, CONFORMER_OK);
  2. an amendment to apps-design/03-APP-aishell-asr-audit.md preregisters
     the expanded backbone/corpus/SNR roster and the Holm-family
     recomputation rule (m grows from the frozen main run's 2 -- the exact
     new m is fixed there, not guessed here);
  3. decode_whisper.py / decode_conformer_ms.py have each been smoke-tested
     against real box output at least once (their token-logp/N-best output
     shapes are UNVERIFIED -- see their module docstrings).

Set REQUIRES_EXPANSION_FREEZE=confirmed once all three are actually true.
This script is a structure-only skeleton either way -- read it before
trusting it with real compute.
EOF
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../expansion_results}"
DEVICE="${DEVICE:-cuda}"
ALPHA="${ALPHA:-0.02}"
DELTA="${DELTA:-0.1}"
SEED="${SEED:-0}"
DATA_ROOT_AISHELL="${DATA_ROOT_AISHELL:-${AUTODL_TMP}/data_aishell}"
DATA_ROOT_THCHS30="${DATA_ROOT_THCHS30:-${AUTODL_TMP}/data_thchs30}"
MUSAN_DIR="${MUSAN_DIR:-${AUTODL_TMP}/musan}"

mkdir -p "$RESULTS_DIR"

# --- CONTENT-GATED completion markers (2026-07-09 DOFA lesson: never trust
#     a bare exit code) ---------------------------------------------------

# Asserts a canonical decode JSONL has >= min_rows rows and a non-null
# hyp_text fraction above a floor.
assert_decode_content() {
  local path="$1" min_rows="${2:-1}"
  python3 - "$path" "$min_rows" <<'PYEOF'
import json, sys
path, min_rows = sys.argv[1], int(sys.argv[2])
n = 0
n_nonempty = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("hyp_text"):
            n_nonempty += 1
if n < min_rows:
    print(f"CONTENT_GATE_FAIL {path}: n={n} < min_rows={min_rows}", file=sys.stderr)
    sys.exit(1)
frac = n_nonempty / n if n else 0.0
if frac < 0.5:
    print(f"CONTENT_GATE_FAIL {path}: non-empty hyp_text fraction {frac:.3f} < 0.5", file=sys.stderr)
    sys.exit(1)
print(f"CONTENT_GATE_OK {path}: n={n} non_empty_frac={frac:.3f}")
PYEOF
}

# Asserts a MUSAN mix manifest has >= min_rows rows (mix_wav_list already
# skip-and-counts unreadable utterances; this just checks the run wasn't a
# total wipeout).
assert_mix_content() {
  local manifest="$1" min_rows="${2:-1}"
  python3 - "$manifest" "$min_rows" <<'PYEOF'
import json, sys
path, min_rows = sys.argv[1], int(sys.argv[2])
n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
if n < min_rows:
    print(f"CONTENT_GATE_FAIL {path}: n={n} < min_rows={min_rows}", file=sys.stderr)
    sys.exit(1)
print(f"CONTENT_GATE_OK {path}: n={n}")
PYEOF
}

step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }

decode_backbone() {
  # decode_backbone BACKBONE CORPUS SPLIT OUT_PATH
  local backbone="$1" corpus="$2" split="$3" out="$4"
  local data_root
  if [ "$corpus" = "aishell" ]; then data_root="$DATA_ROOT_AISHELL"; else data_root="$DATA_ROOT_THCHS30"; fi
  case "$backbone" in
    paraformer)
      python3 "$SCRIPT_DIR/decode_paraformer.py" --split "$split" --data-root "$data_root" \
        --device "$DEVICE" --out "$out"
      ;;
    whisper)
      python3 "$SCRIPT_DIR/decode_whisper.py" --split "$split" --corpus "$corpus" \
        --data-root "$data_root" --device "$DEVICE" --out "$out"
      ;;
    conformer)
      python3 "$SCRIPT_DIR/decode_conformer_ms.py" --split "$split" --corpus "$corpus" \
        --data-root "$data_root" --device "$DEVICE" --out "$out"
      ;;
    *)
      echo "decode_backbone: unknown backbone $backbone" >&2; return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Stage A: clean decode, {aishell,thchs30} x {paraformer,whisper,conformer}.
# NOTE: decode_paraformer.py is Aishell-only as written (openslr-33 layout
# hardcoded in its own module -- see its docstring); a THCHS-30 variant is
# future work if the paraformer x thchs30 cell is wanted -- skipped below,
# disclosed rather than silently attempted.
# ---------------------------------------------------------------------------
step "A: clean decode grid"
CORPORA=(aishell thchs30)
BACKBONES=(paraformer whisper conformer)
for corpus in "${CORPORA[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [ "$backbone" = "paraformer" ] && [ "$corpus" = "thchs30" ]; then
      echo "skip: paraformer x thchs30 (decode_paraformer.py is Aishell-layout-only)"
      continue
    fi
    out="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
    if [ -s "$out" ]; then skip "$out"; else decode_backbone "$backbone" "$corpus" test "$out"; fi
    assert_decode_content "$out"
  done
done

# ---------------------------------------------------------------------------
# Stage B: aishell x {5,15,25}dB MUSAN noise (reinstates SNR Mondrian
# strata, design §2.4's dropped-for-headline axis). Mixes the ALREADY
# DECODED-CLEAN aishell test wavs' source audio, then re-decodes the mixed
# wavs with the primary backbone (paraformer, matching the frozen main
# run's roster) -- noise augmentation happens on RAW AUDIO, not decode
# artifacts, so this is a re-decode step, not a JSONL transform.
# ---------------------------------------------------------------------------
step "B: aishell noise-stratified decode (MUSAN, paraformer backbone)"
SNRS=(5 15 25)
for snr in "${SNRS[@]}"; do
  wav_list="$RESULTS_DIR/aishell_test_wavlist.txt"
  if [ ! -s "$wav_list" ]; then
    find "$DATA_ROOT_AISHELL/wav/test" -name '*.wav' -printf '%f %p\n' | sed 's/\.wav / /' > "$wav_list"
  fi
  mixed_dir="$RESULTS_DIR/aishell_test_musan_${snr}db"
  manifest="$mixed_dir/manifest.jsonl"
  if [ ! -s "$manifest" ]; then
    python3 "$SCRIPT_DIR/mix_musan.py" --wav-list "$wav_list" --musan-dir "$MUSAN_DIR" \
      --snr-db "$snr" --out-dir "$mixed_dir" --seed "$SEED"
  else
    skip "$manifest"
  fi
  assert_mix_content "$manifest"
  # TODO (post-freeze): re-decode $mixed_dir with decode_paraformer.py once
  # it accepts an arbitrary wav-list/dir input (currently Aishell-wav-tree-
  # only) rather than a fresh {data_root}/wav/{split}/... discovery -- a
  # small follow-up to decode_paraformer.py's CLI, not sketched here to
  # avoid touching that already-verified script pre-freeze.
done

# ---------------------------------------------------------------------------
# Stage C: cross-corpus transfer arm -- calibrate on aishell dev (reusing
# run_pilot.sh's cal20/tune20 carve), apply/audit on thchs30 test (domain
# shift) and on the MUSAN-noised aishell test (degradation shift).
# ---------------------------------------------------------------------------
step "C: cross-corpus + noise-stratified calibrate/apply/audit"
for backbone in "${BACKBONES[@]}"; do
  cal_gate="$RESULTS_DIR/gate_${backbone}.json"
  cal_scored="$RESULTS_DIR/../pilot_results/cal20_scored.jsonl"  # reuse the frozen pilot carve for paraformer;
                                                                  # whisper/conformer need their OWN dev decode +
                                                                  # cal/tune carve (not sketched -- mirrors
                                                                  # run_pilot.sh steps 1-4, per-backbone).
  echo "TODO: calibrate on aishell dev for backbone=$backbone -> $cal_gate (see run_pilot.sh steps 1-4)"
  for corpus in thchs30; do
    decoded="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
    [ -s "$decoded" ] || { echo "skip apply/audit: $decoded missing"; continue; }
    echo "TODO: asr-gate apply/audit --gate $cal_gate --instances $decoded (cross-corpus arm)"
  done
  for snr in "${SNRS[@]}"; do
    mixed_manifest="$RESULTS_DIR/aishell_test_musan_${snr}db/manifest.jsonl"
    [ -s "$mixed_manifest" ] || continue
    echo "TODO: re-decode + apply/audit the ${snr}dB MUSAN arm for backbone=$backbone (noise-stratified arm)"
  done
done

# ---------------------------------------------------------------------------
# Stage D: audit, roster-derived Holm family (per (backbone,corpus) cell
# that actually has a scored/decoded JSONL on disk -- m is COMPUTED here,
# never hardcoded, matching FREEZE-NOTE-2026-07-09.md's precedent).
# ---------------------------------------------------------------------------
step "D: roster-derived Holm audit"
echo "TODO: enumerate \$RESULTS_DIR/decode_*_*_test.jsonl + the MUSAN arms," \
     "count m = (present backbone,corpus,condition) x 5 primary scores," \
     "run 'asr-gate audit' per cell, Holm-correct across the REALIZED m" \
     "(never the design doc's original m=10) -- mirrors audit.py's own" \
     "roster-derived-never-hardcoded convention."

echo "expansion run skeleton complete -> $RESULTS_DIR"
echo "This is structure only -- Stage C/D TODOs above must be filled in" \
     "against the frozen expansion design amendment before this script" \
     "does anything beyond decode+mix."
