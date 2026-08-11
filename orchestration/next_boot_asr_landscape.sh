#!/usr/bin/env bash
# next_boot_asr_landscape.sh -- REAL box chain for the ASR-LANDSCAPE roster
# preregistered in FREEZE-AMENDMENT-2026-07-13.md (backbone x corpus matrix
# that answers the red-team's M1/M2 scope levers). Successor to
# next_boot_asr_expansion.sh; same hard-won conventions:
#   * decoding ONLY through the verified decode_*.py CLIs (each takes --split
#     explicitly, so the discover_corpus-missing-split bug class is impossible);
#   * every completion MARKER asserts on RESULT CONTENT (row counts, non-null
#     score fractions), never a bare exit code (the DOFA lesson);
#   * all tunables env-overridable with NAMED DEFAULTS (no hardcoded paths /
#     statistical denominators inline); model IDs are named tunables because
#     the landscape's whole point is specific documented-data backbones
#     (FREEZE-AMENDMENT §1), so they are frozen defaults, still overridable.
#
# NEW vs the expansion chain:
#   * roster = B2 paraformer-zh + B3' Belle-whisper-large-v3-zh (decode_whisper
#     VERBATIM, --model-name + --language zh) + B4 zipformer transducer
#     (decode_sherpa_onnx, Path-D ys_probs); stretch B5/B6 GATED off by default;
#   * corpora = aishell + thchs30 + aidatatang (+ optional magicdata);
#   * a POSTERIOR-SHAPE SMOKE per backbone (--probe 20 where the decoder
#     supports it, else --limit 20 + usable-score gate) runs BEFORE any full
#     decode -- the FREEZE-AMENDMENT §1/§5 gate that must pass before B4/stretch
#     are trusted;
#   * --skip-existing + optional --resume on every decode (idempotent/resumable);
#   * --dry-run validates paths + decoder CLIs + the plan LOCALLY, no GPU / no
#     model download / no data required -- run it before booking the box.
#
# Usage:
#   bash next_boot_asr_landscape.sh --dry-run     # local plan validation, no GPU
#   bash next_boot_asr_landscape.sh               # real box run
#   bash next_boot_asr_landscape.sh --resume      # real box run, resume decodes
#   [--with-magicdata] [--with-stretch]           # optional 4th corpus / stretch backbones

set -uo pipefail  # NOT -e: later stages/markers/epilogue must run after an
                  # earlier content gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
DRY_RUN=0
RESUME=0
WITH_MAGICDATA=0
WITH_STRETCH=0
# Mandarin corpus roster (env-overridable). DEVIATION 2026-07-13: openslr
# SLR62 (aidatatang_200zh) has been WITHDRAWN from openslr.org ("Resource
# not found: 62", verified from two networks at launch time), so the frozen
# default of aishell+thchs30+aidatatang is unobtainable; the FREEZE-AMENDMENT
# matrix's own frozen OPTIONAL corpus (MagicData, SLR68, test_set.tar.gz
# still live) substitutes as the third corpus via
#   MANDARIN_CORPORA="aishell thchs30 magicdata"
# Recorded as a dated disclosed deviation in FREEZE-AMENDMENT-2026-07-13.md §D1.
MANDARIN_CORPORA="${MANDARIN_CORPORA:-aishell thchs30 aidatatang}"
for arg in "$@"; do
  case "$arg" in
    --dry-run)        DRY_RUN=1 ;;
    --resume)         RESUME=1 ;;
    --with-magicdata) WITH_MAGICDATA=1 ;;
    --with-stretch)   WITH_STRETCH=1 ;;
    -h|--help)
      grep -E '^# (Usage|  bash|  \[)' "${BASH_SOURCE[0]}" | sed 's/^# //'
      exit 0 ;;
    *) echo "unknown flag: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults). Model IDs are frozen-per-amendment
# defaults (FREEZE-AMENDMENT-2026-07-13 §1), still overridable.
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-${AUTODL_TMP}/asr_landscape_results}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

DATA_ROOT_AISHELL="${DATA_ROOT_AISHELL:-${AUTODL_TMP}/data_aishell}"
DATA_ROOT_THCHS30="${DATA_ROOT_THCHS30:-${AUTODL_TMP}/data_thchs30}"
DATA_ROOT_AIDATATANG="${DATA_ROOT_AIDATATANG:-${AUTODL_TMP}/aidatatang_200zh}"
DATA_ROOT_MAGICDATA="${DATA_ROOT_MAGICDATA:-${AUTODL_TMP}/magicdata}"

# B3' Belle (AED, documented data) + B4 zipformer (RNN-T, open data). Frozen ids.
BELLE_MODEL="${BELLE_MODEL:-BELLE-2/Belle-whisper-large-v3-zh}"
ZIPFORMER_DIR="${ZIPFORMER_DIR:-${AUTODL_TMP}/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2}"
# Stretch (GATED, --with-stretch only; only load-bearing if the posterior hook
# verifies in the smoke, FREEZE-AMENDMENT §5).
SENSEVOICE_MODEL="${SENSEVOICE_MODEL:-iic/SenseVoiceSmall}"

# Frozen certificate config (FREEZE-AMENDMENT §4 -- landscape alpha-grid).
ALPHA_PRIMARY="${ALPHA_PRIMARY:-0.02}"
ALPHA_FRONTIER="${ALPHA_FRONTIER:-0.03,0.05,0.10}"  # landscape grid {2,3,5,10}%
ALPHA_BINDING="${ALPHA_BINDING:-0.015}"             # sub-base-rate binding probe (alpha015)
DELTA="${DELTA:-0.1}"
LTT_PROCEDURE="${LTT_PROCEDURE:-bonferroni}"
LTT_PVALUE="${LTT_PVALUE:-eb}"
N_RESEEDS="${N_RESEEDS:-20}"
AUDIT_ALPHA="${AUDIT_ALPHA:-0.05}"
AUDIT_N_PERM="${AUDIT_N_PERM:-2000}"

# Smoke / cap knobs.
SMOKE_N="${SMOKE_N:-20}"                 # posterior-shape smoke size per backbone
AIDATATANG_TEST_CAP="${AIDATATANG_TEST_CAP:-4000}"  # FREEZE-AMENDMENT §2 speaker-disjoint cap
MAGICDATA_TEST_CAP="${MAGICDATA_TEST_CAP:-4000}"

HF_HOME="${HF_HOME:-${AUTODL_TMP}/hf-cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ASR_LOG="${ASR_LOG:-/root/asr_landscape.log}"
GPU_UTIL_LOG="${GPU_UTIL_LOG:-/root/gpu_util_landscape.log}"
GPU_LOG_PIDFILE="${GPU_LOG_PIDFILE:-/root/gpu_util_landscape.pid}"

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIR/markers}"

B4_DEGRADED=0        # set if the zipformer ys_probs smoke fails (drop B4)
SENSEVOICE_DEGRADED=1  # stretch off unless --with-stretch AND its smoke passes
FAILED_MARKERS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }

# run CMD... -- in --dry-run, print instead of executing (for GPU/box work).
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRYRUN would run: $*"
    return 0
  fi
  "$@"
}

mark() {
  local name="$1" status="$2"
  if [ "$DRY_RUN" -eq 1 ]; then echo "DRYRUN marker ${name}=${status}"; return 0; fi
  mkdir -p "$MARKERS_DIR"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  [ "$status" = "FAILED" ] && FAILED_MARKERS+=("$name")
  return 0
}

# resume flag string passed to decoders (empty unless --resume).
resume_flag() { [ "$RESUME" -eq 1 ] && printf -- "--resume"; }

# Content gate: >=min_rows rows with non-empty hyp_text (DOFA lesson).
assert_decode_content() {
  local path="$1" min_rows="${2:-1}"
  [ -s "$path" ] || { echo "CONTENT_GATE_FAIL $path: missing/empty" >&2; return 1; }
  python3 - "$path" "$min_rows" <<'PYEOF'
import json, sys
path, min_rows = sys.argv[1], int(sys.argv[2])
n = nonempty = 0
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    rec = json.loads(line); n += 1
    if rec.get("hyp_text"): nonempty += 1
if n < min_rows:
    print(f"CONTENT_GATE_FAIL {path}: n={n} < {min_rows}", file=sys.stderr); sys.exit(1)
frac = nonempty / n if n else 0.0
if frac < 0.5:
    print(f"CONTENT_GATE_FAIL {path}: nonempty_frac {frac:.3f} < 0.5", file=sys.stderr); sys.exit(1)
print(f"CONTENT_GATE_OK {path}: n={n} nonempty_frac={frac:.3f}")
PYEOF
}

# Fraction of records with a usable per-token score (nbest[0].token_logps set)
# -- the posterior-exposure check (usable=N/M), for decoders lacking --probe.
assert_usable_scores() {
  local path="$1" min_usable="${2:-1}"
  [ -s "$path" ] || { echo "usable=0/0 ($path missing)" >&2; return 1; }
  python3 - "$path" "$min_usable" <<'PYEOF'
import json, sys
path, min_usable = sys.argv[1], int(sys.argv[2])
n = usable = 0
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    rec = json.loads(line); n += 1
    nb = rec.get("nbest") or []
    if nb and nb[0].get("token_logps") is not None: usable += 1
print(f"usable={usable}/{n} ({path})")
sys.exit(0 if usable >= min_usable else 1)
PYEOF
}

# Fraction of a sherpa-onnx probe sidecar whose token_logps aligned (ys_probs
# populated 1:1 with tokens) -- the FREEZE-AMENDMENT §1 B4 gate.
assert_probe_aligned() {
  local probe_json="$1" min_aligned="${2:-1}"
  [ -s "$probe_json" ] || { echo "aligned=0/0 ($probe_json missing)" >&2; return 1; }
  python3 - "$probe_json" "$min_aligned" <<'PYEOF'
import json, sys
probe, min_aligned = sys.argv[1], int(sys.argv[2])
rows = json.load(open(probe, encoding="utf-8"))
n = len(rows)
aligned = sum(1 for r in rows if r.get("tokens_aligned"))
print(f"aligned={aligned}/{n} ({probe})")
sys.exit(0 if aligned >= min_aligned else 1)
PYEOF
}

data_root_for() {
  case "$1" in
    aishell)    echo "$DATA_ROOT_AISHELL" ;;
    thchs30)    echo "$DATA_ROOT_THCHS30" ;;
    aidatatang) echo "$DATA_ROOT_AIDATATANG" ;;
    magicdata)  echo "$DATA_ROOT_MAGICDATA" ;;
    *) echo "" ;;
  esac
}

# decode_backbone BACKBONE CORPUS SPLIT OUT [EXTRA...] -- dispatch through the
# verified CLIs; every decode gets --skip-existing (+ --resume if requested).
decode_backbone() {
  local backbone="$1" corpus="$2" split="$3" out="$4"; shift 4
  local data_root; data_root="$(data_root_for "$corpus")"
  local rf; rf="$(resume_flag)"
  case "$backbone" in
    paraformer)
      run python3 "$SCRIPT_DIR/decode_paraformer.py" --corpus "$corpus" --split "$split" \
        --data-root "$data_root" --device "$DEVICE" --skip-existing $rf --out "$out" "$@" ;;
    belle)
      run python3 "$SCRIPT_DIR/decode_whisper.py" --corpus "$corpus" --split "$split" \
        --data-root "$data_root" --model-name "$BELLE_MODEL" --language zh \
        --device "$DEVICE" --skip-existing $rf --out "$out" "$@" ;;
    zipformer)
      run python3 "$SCRIPT_DIR/decode_sherpa_onnx.py" --corpus "$corpus" --split "$split" \
        --data-root "$data_root" --model-dir "$ZIPFORMER_DIR" --skip-existing $rf --out "$out" "$@" ;;
    *) echo "decode_backbone: unknown backbone $backbone" >&2; return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# --dry-run: validate the plan LOCALLY (no GPU / no model / no data needed).
# ---------------------------------------------------------------------------
dryrun_validate() {
  step "DRY-RUN: local plan validation (no GPU, no model download, no data required)"
  local ok=1

  echo "-- decoder CLI arg validation (--help must parse; proves arg names are valid) --"
  local d
  for d in decode_paraformer.py decode_whisper.py decode_sherpa_onnx.py decode_conformer_ms.py; do
    if python3 "$SCRIPT_DIR/$d" --help >/dev/null 2>&1; then
      echo "  OK   $d --help"
    else
      echo "  FAIL $d --help" >&2; ok=0
    fi
  done

  echo "-- corpus registry (the landscape corpora must be discoverable) --"
  python3 - "$SCRIPT_DIR/.." <<'PYEOF' || ok=0
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
need = {"aishell", "thchs30", "aidatatang", "magicdata"}
have = set(corpora.CORPUS_DISCOVERERS)
missing = need - have
if missing:
    print(f"  FAIL missing discoverers: {sorted(missing)}"); sys.exit(1)
print(f"  OK   CORPUS_DISCOVERERS = {sorted(have)}")
PYEOF

  echo "-- resolved decode command lines (backbone x corpus, test split) --"
  local backbone corpus corpora_list="$MANDARIN_CORPORA"
  [ "$WITH_MAGICDATA" -eq 1 ] && corpora_list="$corpora_list magicdata"
  for backbone in paraformer belle zipformer; do
    for corpus in $corpora_list; do
      echo "  [$backbone x $corpus] -> $(printf 'DECODE ')$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
    done
  done

  echo "-- data roots (present now, or WOULD-STAGE on the box) --"
  local c r
  for c in $MANDARIN_CORPORA $( [ "$WITH_MAGICDATA" -eq 1 ] && echo magicdata ); do
    r="$(data_root_for "$c")"
    if [ -d "$r" ]; then echo "  PRESENT  $c: $r"; else echo "  WOULD-STAGE $c: $r (credential-free; openslr / autodl-pub)"; fi
  done
  echo "-- model artifacts (WOULD-STAGE on the box) --"
  echo "  B3' Belle : $BELLE_MODEL (hf download)"
  echo "  B4  zipf. : $ZIPFORMER_DIR ($( [ -d "$ZIPFORMER_DIR" ] && echo PRESENT || echo 'hf download zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2' ))"

  echo "-- asr-gate CLI (calibrate/apply/audit) availability --"
  if command -v asr-gate >/dev/null 2>&1; then echo "  OK   asr-gate on PATH"; else echo "  NOT-LOCAL asr-gate (present on the box env)"; fi

  echo "-- frozen certificate config (FREEZE-AMENDMENT §4) --"
  echo "  alpha grid = ${ALPHA_PRIMARY},${ALPHA_FRONTIER} (+binding ${ALPHA_BINDING}); delta=${DELTA}; reseeds=${N_RESEEDS}; strata=duration_tercile"

  echo "-- calibration-pool sweep (0 GPU, LOCAL) --"
  echo "  mandarin_calsweep_2026-07-13/run_mandarin_calsweep.py is a CACHED-score sweep (already runnable off-box; not part of this box chain)"

  if [ "$ok" -eq 1 ]; then
    echo "DRY-RUN OK: decoder CLIs valid, corpora registered, plan resolves. Safe to book the box."
    return 0
  fi
  echo "DRY-RUN FAILED: see FAIL lines above." >&2
  return 1
}

# ---------------------------------------------------------------------------
# Prologue (real run only).
# ---------------------------------------------------------------------------
stage_prologue() {
  step "Prologue"
  local conda_sh=""
  for candidate in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh \
                   "${HOME:-/root}/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$candidate" ] && { conda_sh="$candidate"; break; }
  done
  if [ -n "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"; conda activate base || echo "warning: conda activate base failed" >&2
  else
    echo "warning: no conda.sh found -- continuing with current environment" >&2
  fi
  export HF_HOME HF_ENDPOINT
  echo "HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
  mkdir -p "$RESULTS_DIR" "$MARKERS_DIR" "$(dirname "$ASR_LOG")" 2>/dev/null || true
  exec > >(tee -a "$ASR_LOG") 2>&1
  echo "logging to $ASR_LOG"
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while true; do
        { date -u +"%Y-%m-%dT%H:%M:%SZ"
          nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
        } >> "$GPU_UTIL_LOG" 2>&1; sleep 60
      done ) &
    echo $! > "$GPU_LOG_PIDFILE"
    echo "started nvidia-smi logger (pid $(cat "$GPU_LOG_PIDFILE")) -> $GPU_UTIL_LOG"
  fi
}
_cleanup_loggers() {
  [ -f "$GPU_LOG_PIDFILE" ] && { kill "$(cat "$GPU_LOG_PIDFILE")" 2>/dev/null || true; rm -f "$GPU_LOG_PIDFILE"; }
}
trap _cleanup_loggers EXIT

# ---------------------------------------------------------------------------
# Data gates (content, not exit codes).
# ---------------------------------------------------------------------------
gate_corpus_split_present() {
  local corpus="$1" split="$2" data_root; data_root="$(data_root_for "$corpus")"
  python3 - "$SCRIPT_DIR/.." "$corpus" "$data_root" "$split" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
corpus, data_root, split = sys.argv[2], sys.argv[3], sys.argv[4]
try:
    entries = corpora.discover_corpus(corpus, data_root, split, limit=1)
except Exception as e:
    print(f"GATE_FAIL {corpus}/{split}: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
if not entries:
    print(f"GATE_FAIL {corpus}/{split}: no utterances under {data_root}", file=sys.stderr); sys.exit(1)
print(f"GATE_OK {corpus}/{split}: discovered under {data_root}")
PYEOF
}

stage_data_gates() {
  step "Data gates"
  local ok=1 c
  for c in $MANDARIN_CORPORA $( [ "$WITH_MAGICDATA" -eq 1 ] && echo magicdata ); do
    if gate_corpus_split_present "$c" test; then mark "DATA_${c}_TEST" OK; else mark "DATA_${c}_TEST" FAILED; ok=0; fi
  done
  # aishell dev is the calibration source.
  if gate_corpus_split_present aishell dev; then mark DATA_aishell_DEV OK; else mark DATA_aishell_DEV FAILED; ok=0; fi
  [ "$ok" -eq 1 ]
}

# ---------------------------------------------------------------------------
# Posterior-shape SMOKE per backbone (BEFORE full decode). FREEZE-AMENDMENT
# §1/§5: B4 (zipformer) and stretch backbones must expose posteriors here or be
# demoted to skipped-degraded; B2/B3' are proven and gated on content+usable.
# ---------------------------------------------------------------------------
stage_smoke_gates() {
  step "Posterior-shape smoke (SMOKE_N=$SMOKE_N utts/backbone, BEFORE full decode)"
  local smoke_dir="$RESULTS_DIR/smoke"
  run mkdir -p "$smoke_dir"

  # B2 paraformer (proven; --limit + usable-score gate).
  if run python3 "$SCRIPT_DIR/decode_paraformer.py" --corpus aishell --split test \
        --data-root "$DATA_ROOT_AISHELL" --device "$DEVICE" --limit "$SMOKE_N" \
        --out "$smoke_dir/smoke_paraformer.jsonl" \
     && { [ "$DRY_RUN" -eq 1 ] || { assert_decode_content "$smoke_dir/smoke_paraformer.jsonl" 1 \
          && assert_usable_scores "$smoke_dir/smoke_paraformer.jsonl" 1; }; }; then
    mark SMOKE_paraformer OK
  else
    mark SMOKE_paraformer FAILED; return 1
  fi

  # B3' Belle (AED, decode_whisper verbatim; --limit + usable-score gate --
  # certificate backbone, so usable scores REQUIRED).
  if run python3 "$SCRIPT_DIR/decode_whisper.py" --corpus aishell --split test \
        --data-root "$DATA_ROOT_AISHELL" --model-name "$BELLE_MODEL" --language zh \
        --device "$DEVICE" --limit "$SMOKE_N" --out "$smoke_dir/smoke_belle.jsonl" \
     && { [ "$DRY_RUN" -eq 1 ] || { assert_decode_content "$smoke_dir/smoke_belle.jsonl" 1 \
          && assert_usable_scores "$smoke_dir/smoke_belle.jsonl" 1; }; }; then
    mark SMOKE_belle OK
  else
    mark SMOKE_belle FAILED
    echo "ABORTING: Belle is a certificate backbone with zero usable scores in smoke" >&2
    return 1
  fi

  # B4 zipformer (Path D ys_probs -- the UNVERIFIED alignment; --probe dumps the
  # raw shape). DEGRADE (not abort) if ys_probs doesn't align: drop B4, keep
  # B2+B3' (FREEZE-AMENDMENT §1).
  run python3 "$SCRIPT_DIR/decode_sherpa_onnx.py" --corpus aishell --split test \
    --data-root "$DATA_ROOT_AISHELL" --model-dir "$ZIPFORMER_DIR" --probe "$SMOKE_N" \
    --out "$smoke_dir/smoke_zipformer.jsonl" \
    || echo "note: zipformer --probe exited non-zero -- see $smoke_dir/smoke_zipformer.probe.json"
  if [ "$DRY_RUN" -eq 1 ]; then
    mark SMOKE_zipformer OK
  elif assert_probe_aligned "$smoke_dir/smoke_zipformer.probe.json" 1; then
    mark SMOKE_zipformer OK; B4_DEGRADED=0
  else
    mark SMOKE_zipformer SKIPPED_DEGRADED
    echo "zipformer ys_probs did not align 1:1 with tokens -- dropping B4, keeping B2+B3' (FREEZE-AMENDMENT §1)"
    B4_DEGRADED=1
  fi

  # Stretch (GATED): SenseVoice CTC-logit hook, only if --with-stretch AND its
  # probe verifies. Off by default (FREEZE-AMENDMENT §5).
  if [ "$WITH_STRETCH" -eq 1 ]; then
    echo "note: --with-stretch set -- SenseVoice smoke is a manual/box step (needs a CTC-logit hook, FREEZE-AMENDMENT §5); left as SKIPPED_DEGRADED unless its hook lands"
    mark SMOKE_sensevoice SKIPPED_DEGRADED
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Full decode grid: active backbones x corpora, test split (+ aishell dev for
# calibration). --skip-existing/--resume throughout.
# ---------------------------------------------------------------------------
active_backbones() {
  local bb="paraformer belle"
  [ "$B4_DEGRADED" -eq 0 ] && bb="$bb zipformer"
  echo "$bb"
}

stage_decode_grid() {
  step "Decode grid (active backbones x corpora, test + aishell dev)"
  local corpora_list="$MANDARIN_CORPORA"
  [ "$WITH_MAGICDATA" -eq 1 ] && corpora_list="$corpora_list magicdata"
  local backbone corpus out
  for backbone in $(active_backbones); do
    # calibration source: aishell dev, once per backbone.
    out="$RESULTS_DIR/decode_aishell_${backbone}_dev.jsonl"
    if [ "$DRY_RUN" -eq 0 ] && [ -s "$out" ]; then skip "$out"; else decode_backbone "$backbone" aishell dev "$out"; fi
    [ "$DRY_RUN" -eq 1 ] || { assert_decode_content "$out" 1 && mark "DECODE_aishell_${backbone}_dev" OK || mark "DECODE_aishell_${backbone}_dev" FAILED; }
    # test splits.
    for corpus in $corpora_list; do
      out="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
      if [ "$DRY_RUN" -eq 0 ] && [ -s "$out" ]; then skip "$out"; else decode_backbone "$backbone" "$corpus" test "$out"; fi
      [ "$DRY_RUN" -eq 1 ] || { assert_decode_content "$out" 1 && mark "DECODE_${corpus}_${backbone}" OK || mark "DECODE_${corpus}_${backbone}" FAILED; }
    done
  done
}

# ---------------------------------------------------------------------------
# Calibrate (aishell dev) -> apply (each corpus test) + audit. Mirrors the
# expansion chain's stage C/D via the frozen asr-gate CLI. The calibration-pool
# sweep (mandarin_calsweep) is a separate 0-GPU cached-score run, not repeated
# here.
# ---------------------------------------------------------------------------
stage_calibrate_apply_audit() {
  step "Calibrate (aishell dev) -> apply per-corpus test + roster-derived audit"
  local corpora_list="$MANDARIN_CORPORA"
  [ "$WITH_MAGICDATA" -eq 1 ] && corpora_list="$corpora_list magicdata"
  local frontier; frontier="$(echo "$ALPHA_FRONTIER" | tr ',' ' ')"
  local backbone corpus
  for backbone in $(active_backbones); do
    local bb_dir="$RESULTS_DIR/backbone_${backbone}"; run mkdir -p "$bb_dir"
    local dev_decode="$RESULTS_DIR/decode_aishell_${backbone}_dev.jsonl"
    local dev_canonical="$bb_dir/dev_canonical.jsonl"
    run bash -c "[ -s '$dev_canonical' ] || asr-gate ingest --hyps '$dev_decode' --format custom-schema --out '$dev_canonical'"

    # per-corpus scored test targets.
    for corpus in $corpora_list; do
      local decoded="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
      local scored="$bb_dir/${corpus}_test_scored.jsonl"
      run bash -c "[ -s '$decoded' ] && [ ! -s '$scored' ] && { asr-gate ingest --hyps '$decoded' --format custom-schema --out '${scored%.jsonl}_canonical.jsonl' && asr-gate score --instances '${scored%.jsonl}_canonical.jsonl' --out '$scored'; } || true"
    done

    local reseed alpha
    for reseed in $(seq 0 $((N_RESEEDS - 1))); do
      local reseed_dir="$bb_dir/reseed_${reseed}"; run mkdir -p "$reseed_dir"
      run bash -c "[ -s '$reseed_dir/cal20_scored.jsonl' ] || { python3 '$SCRIPT_DIR/split_cal_tune.py' '$dev_canonical' '$reseed_dir/cal20.jsonl' '$reseed_dir/tune20.jsonl' '$reseed' && asr-gate score --instances '$reseed_dir/cal20.jsonl' --out '$reseed_dir/cal20_scored.jsonl'; }"
      for alpha in "$ALPHA_PRIMARY" $frontier "$ALPHA_BINDING"; do
        local gate_json="$reseed_dir/gate_alpha${alpha}.json"
        run bash -c "[ -s '$gate_json' ] || asr-gate calibrate --instances '$reseed_dir/cal20_scored.jsonl' --alpha '$alpha' --delta '$DELTA' --guarantee ltt --strata duration_tercile --seed '$reseed' --ltt-procedure '$LTT_PROCEDURE' --ltt-pvalue '$LTT_PVALUE' --out '$gate_json'"
        for corpus in $corpora_list; do
          local scored="$bb_dir/${corpus}_test_scored.jsonl"
          local applied="$reseed_dir/applied_${corpus}_alpha${alpha}.json"
          run bash -c "[ -s '$scored' ] && [ ! -s '$applied' ] && asr-gate apply --gate '$gate_json' --instances '$scored' --out '$applied' || true"
        done
      done
    done
    mark "CALIBRATE_${backbone}" OK
  done

  # Roster-derived Holm audit: m COMPUTED from realized cells (never hardcoded),
  # same rule as next_boot_asr_expansion.sh's Stage D.
  step "Roster-derived Holm audit (m computed from realized cells)"
  local audit_dir="$RESULTS_DIR/audit"; run mkdir -p "$audit_dir"
  for backbone in $(active_backbones); do
    for corpus in $corpora_list; do
      local scored="$RESULTS_DIR/backbone_${backbone}/${corpus}_test_scored.jsonl"
      local audit_json="$audit_dir/audit_${corpus}_${backbone}.json"
      run bash -c "[ -s '$scored' ] && [ ! -s '$audit_json' ] && asr-gate audit --instances '$scored' --n-perm '$AUDIT_N_PERM' --alpha '$AUDIT_ALPHA' --seed '$SEED' --out '$audit_json' || true"
    done
  done
  mark AUDIT_ROSTER OK
}

# ---------------------------------------------------------------------------
# Epilogue (real run only).
# ---------------------------------------------------------------------------
stage_epilogue() {
  step "Epilogue"
  if [ "${#FAILED_MARKERS[@]}" -eq 0 ]; then
    echo "ALL_DONE $(date -u +%FT%TZ)" > "$RESULTS_DIR/ASR_LANDSCAPE_ALL_DONE"
    echo "ASR_LANDSCAPE_ALL_DONE"
  else
    { echo "PARTIAL $(date -u +%FT%TZ)"; printf '%s\n' "${FAILED_MARKERS[@]}"; } > "$RESULTS_DIR/ASR_LANDSCAPE_PARTIAL"
    echo "ASR_LANDSCAPE_PARTIAL -- failed: ${FAILED_MARKERS[*]}"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  dryrun_validate
  exit $?
fi

stage_prologue
GRID_ABORTED=0
stage_data_gates   || GRID_ABORTED=1
[ "$GRID_ABORTED" -eq 0 ] && { stage_smoke_gates || GRID_ABORTED=1; }
if [ "$GRID_ABORTED" -eq 0 ]; then
  stage_decode_grid
  stage_calibrate_apply_audit
else
  echo "GRID ABORTED before full decode -- see FAILED markers ($MARKERS_DIR)" >&2
fi
stage_epilogue
