#!/usr/bin/env bash
# next_boot_asr_expansion.sh -- REAL grid runner for the expanded roster
# preregistered in EXPANSION-AMENDMENT-2026-07-09.md. Supersedes
# run_expansion.sh (which stays as a structure-only skeleton, TODOs and
# all) -- this is the versioned box chain that actually runs Stages A-D.
#
# Written after the 2026-07-09 boot attempt failed its pre-gates before
# spending any paid grid compute (expansion_results_2026-07-09/
# asr_expansion.log): (1) the ad-hoc chain called
# asr_gate.corpora.discover_corpus() without the required `split` arg --
# fixed here structurally by always decoding through the verified
# decode_*.py CLIs (which always pass split correctly), never
# reimplementing corpus discovery inline; (2) decode_whisper.py's score
# extraction broke on this transformers build -- fixed in
# decode_whisper.py itself (see its module docstring), this chain's
# whisper smoke gate now asserts on that fix; (3) decode_conformer_ms.py's
# tolerant extraction didn't recognize the real result shape -- also fixed
# in decode_conformer_ms.py (nested-shape search + --probe mode), this
# chain's conformer smoke gate degrades (not aborts) if it's still zero.
#
# Every completion marker below asserts on RESULT CONTENT (row counts,
# non-null score fractions), never a bare exit code -- the same DOFA
# lesson run_expansion.sh's header documents.
#
# All tunables are env-overridable with NAMED DEFAULTS (nothing hardcoded
# inline below) -- see the "Tunables" section. Model IDs are never
# hardcoded here either: every decode_*.py invocation omits --model-name,
# relying entirely on each script's own verified default.
#
# Usage (on the AutoDL box): bash next_boot_asr_expansion.sh
# (or: nohup bash next_boot_asr_expansion.sh > /root/asr_expansion2_boot.log 2>&1 &)

set -uo pipefail  # deliberately NOT -e: later stages/markers/epilogue
                   # (tar, shutdown) must still run after an earlier
                   # stage's content gate fails -- see stage_epilogue.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults -- ABSOLUTE RULE: no hardcoded
# paths/model IDs/statistical denominators inline in the stages below).
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-${AUTODL_TMP}/asr_expansion2_results}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

DATA_ROOT_AISHELL="${DATA_ROOT_AISHELL:-${AUTODL_TMP}/data_aishell}"
DATA_ROOT_THCHS30="${DATA_ROOT_THCHS30:-${AUTODL_TMP}/data_thchs30}"
ESC50_DIR="${ESC50_DIR:-${AUTODL_TMP}/noise_corpus}"  # materialized location on the asr box (2,000 clips measured)

THCHS_MIN_PAIRS="${THCHS_MIN_PAIRS:-1300}"  # measured mirror test roster = 1,339; see EXPANSION-AMENDMENT addendum 2026-07-09
ESC50_MIN_CLIPS="${ESC50_MIN_CLIPS:-1500}"

# FREEZE-NOTE-2026-07-09.md's frozen config, carried forward as
# env-overridable defaults (per the amendment: "unchanged from
# FREEZE-NOTE-2026-07-09").
ALPHA_PRIMARY="${ALPHA_PRIMARY:-0.02}"
ALPHA_FRONTIER="${ALPHA_FRONTIER:-0.01,0.03,0.05}"  # comma-separated
DELTA="${DELTA:-0.1}"
LTT_PROCEDURE="${LTT_PROCEDURE:-bonferroni}"
LTT_PVALUE="${LTT_PVALUE:-eb}"
N_RESEEDS="${N_RESEEDS:-20}"
SNRS="${SNRS:-5 15 25}"  # space-separated dB list
AUDIT_ALPHA="${AUDIT_ALPHA:-0.05}"
AUDIT_N_PERM="${AUDIT_N_PERM:-2000}"

HF_HOME="${HF_HOME:-${AUTODL_TMP}/hf-cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ASR_LOG="${ASR_LOG:-/root/asr_expansion2.log}"
BALANCE_GUARD="${BALANCE_GUARD:-/root/balance_guard.sh}"
GPU_UTIL_LOG="${GPU_UTIL_LOG:-/root/gpu_util.log}"
GPU_LOG_PIDFILE="${GPU_LOG_PIDFILE:-/root/gpu_util_logger.pid}"

ACK_TIMEOUT_S="${ACK_TIMEOUT_S:-1800}"
RESULTS_TARBALL="${RESULTS_TARBALL:-${AUTODL_TMP}/asr_expansion_results.tar.gz}"
ACK_FILE="${ACK_FILE:-/root/RESULTS_PULLED_ACK}"
NO_AUTOSHUTDOWN_FILE="${NO_AUTOSHUTDOWN_FILE:-/root/NO_AUTOSHUTDOWN}"

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIR/markers}"
mkdir -p "$RESULTS_DIR" "$MARKERS_DIR"

CONFORMER_DEGRADED=0
FAILED_MARKERS=()

# ---------------------------------------------------------------------------
# Small helpers (mirroring run_pilot.sh/run_expansion.sh's conventions).
# ---------------------------------------------------------------------------

step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }

mark() {
  # mark NAME STATUS  (STATUS: OK|FAILED|DEGRADED|SKIPPED_DISCLOSED|SKIPPED_DEGRADED)
  local name="$1" status="$2"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  if [ "$status" = "FAILED" ]; then
    FAILED_MARKERS+=("$name")
  fi
}

# Content-gated completion checks (2026-07-09 DOFA lesson: never trust a
# bare exit code) -- same convention as run_expansion.sh's identically
# named helpers, duplicated here since this chain supersedes rather than
# sources that skeleton.
assert_decode_content() {
  local path="$1" min_rows="${2:-1}"
  [ -s "$path" ] || { echo "CONTENT_GATE_FAIL $path: file missing/empty" >&2; return 1; }
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

assert_mix_content() {
  local manifest="$1" min_rows="${2:-1}"
  [ -s "$manifest" ] || { echo "CONTENT_GATE_FAIL $manifest: file missing/empty" >&2; return 1; }
  python3 - "$manifest" "$min_rows" <<'PYEOF'
import sys
path, min_rows = sys.argv[1], int(sys.argv[2])
n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
if n < min_rows:
    print(f"CONTENT_GATE_FAIL {path}: n={n} < min_rows={min_rows}", file=sys.stderr)
    sys.exit(1)
print(f"CONTENT_GATE_OK {path}: n={n}")
PYEOF
}

# Fraction of records whose top hypothesis carries a usable per-token
# score (nbest[0].token_logps is not null) -- the exact "usable=N/M" shape
# the 2026-07-09 smoke-test log line used (SMOKE_whisper usable=0/3).
assert_usable_scores() {
  local path="$1" min_usable="${2:-1}"
  [ -s "$path" ] || { echo "usable=0/0 ($path: file missing/empty)" >&2; return 1; }
  python3 - "$path" "$min_usable" <<'PYEOF'
import json, sys
path, min_usable = sys.argv[1], int(sys.argv[2])
n = 0
n_usable = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        nbest = rec.get("nbest") or []
        if nbest and nbest[0].get("token_logps") is not None:
            n_usable += 1
print(f"usable={n_usable}/{n} ({path})")
sys.exit(0 if n_usable >= min_usable else 1)
PYEOF
}

decode_backbone() {
  # decode_backbone BACKBONE CORPUS SPLIT OUT_PATH -- dispatches through
  # the verified CLI scripts only; every one of them takes --split
  # explicitly, so the discover_corpus-missing-split bug class is
  # structurally impossible here.
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

# Ingest (canonicalize) + score a decode JSONL exactly once, idempotently
# -- shared by Stage C's cross-corpus/noise targets and Stage D's audit
# inputs. No-op (returns 0) if the source decode file doesn't exist yet
# (a backbone/condition cell that simply isn't realized).
ingest_and_score_target() {
  local decoded="$1" scored_out="$2"
  [ -s "$decoded" ] || return 0
  [ -s "$scored_out" ] && return 0
  local canonical="${scored_out%.jsonl}_canonical.jsonl"
  asr-gate ingest --hyps "$decoded" --format custom-schema --out "$canonical" \
    && asr-gate score --instances "$canonical" --out "$scored_out"
}

# ---------------------------------------------------------------------------
# Prologue
# ---------------------------------------------------------------------------

stage_prologue() {
  step "Prologue"

  local conda_sh=""
  for candidate in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh \
                   "${HOME:-/root}/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then conda_sh="$candidate"; break; fi
  done
  if [ -n "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate base || echo "warning: conda activate base failed -- continuing with current environment" >&2
  else
    echo "warning: no conda.sh found under the usual install paths -- continuing with current environment (conda activate base skipped)" >&2
  fi

  export HF_HOME
  export HF_ENDPOINT
  echo "HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"

  mkdir -p "$(dirname "$ASR_LOG")" 2>/dev/null || true
  exec > >(tee -a "$ASR_LOG") 2>&1
  echo "logging to $ASR_LOG (this line and everything after it is duplicated there)"

  if [ -f "$BALANCE_GUARD" ]; then
    nohup bash "$BALANCE_GUARD" >"${BALANCE_GUARD}.out" 2>&1 &
    echo "started $BALANCE_GUARD (pid $!)"
  else
    echo "note: $BALANCE_GUARD not present -- skipping (not fatal)"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        {
          date -u +"%Y-%m-%dT%H:%M:%SZ"
          nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader 2>&1
        } >> "$GPU_UTIL_LOG" 2>&1
        sleep 60
      done
    ) &
    echo $! > "$GPU_LOG_PIDFILE"
    echo "started per-minute nvidia-smi logger (pid $(cat "$GPU_LOG_PIDFILE")) -> $GPU_UTIL_LOG"
  else
    echo "note: nvidia-smi not found -- skipping GPU utilization logging"
  fi
}

_cleanup_background_loggers() {
  # Kill-by-pidfile, never pkill -f matching on a command-string literal
  # (ABSOLUTE RULE) -- avoids ever writing a pkill invocation whose
  # pattern argument contains the very string being matched against.
  if [ -f "$GPU_LOG_PIDFILE" ]; then
    kill "$(cat "$GPU_LOG_PIDFILE")" 2>/dev/null || true
    rm -f "$GPU_LOG_PIDFILE"
  fi
}
trap _cleanup_background_loggers EXIT

# ---------------------------------------------------------------------------
# Data gates (content, not exit codes) -- fatal: no point running the grid
# on missing/undersized data.
# ---------------------------------------------------------------------------

gate_aishell_split_present() {
  local split="$1"
  python3 - "$SCRIPT_DIR/.." "$DATA_ROOT_AISHELL" "$split" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
data_root, split = sys.argv[2], sys.argv[3]
try:
    entries = corpora.discover_corpus("aishell", data_root, split, limit=1)
except Exception as e:
    print(f"GATE_FAIL aishell/{split}: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
if not entries:
    print(f"GATE_FAIL aishell/{split}: no utterances discovered under {data_root}", file=sys.stderr)
    sys.exit(1)
print(f"GATE_OK aishell/{split}: utterances discovered under {data_root}")
PYEOF
}

gate_thchs_min_pairs() {
  python3 - "$DATA_ROOT_THCHS30" "$THCHS_MIN_PAIRS" <<'PYEOF'
import sys
from pathlib import Path
data_root, min_pairs = Path(sys.argv[1]), int(sys.argv[2])
data_dir = data_root / "data"
if not data_dir.exists():
    print(f"GATE_FAIL thchs30_pairs: {data_dir} not found", file=sys.stderr)
    sys.exit(1)
n_pairs = sum(1 for p in data_dir.glob("*.wav") if (data_dir / f"{p.stem}.wav.trn").exists())
if n_pairs < min_pairs:
    print(f"GATE_FAIL thchs30_pairs: n_pairs={n_pairs} < min={min_pairs} under {data_dir}", file=sys.stderr)
    sys.exit(1)
print(f"GATE_OK thchs30_pairs: n_pairs={n_pairs} >= min={min_pairs}")
PYEOF
}

gate_esc50_min_clips() {
  python3 - "$ESC50_DIR" "$ESC50_MIN_CLIPS" <<'PYEOF'
import sys
from pathlib import Path
esc_dir, min_clips = Path(sys.argv[1]), int(sys.argv[2])
if not esc_dir.exists():
    print(f"GATE_FAIL esc50_clips: {esc_dir} not found", file=sys.stderr)
    sys.exit(1)
n = sum(1 for _ in esc_dir.rglob("*.wav"))
if n <= min_clips:
    print(f"GATE_FAIL esc50_clips: n={n} <= min={min_clips} under {esc_dir}", file=sys.stderr)
    sys.exit(1)
print(f"GATE_OK esc50_clips: n={n} > min={min_clips}")
PYEOF
}

stage_data_gates() {
  step "Data gates (content, not exit codes)"
  local ok=1
  if gate_aishell_split_present test; then mark DATA_AISHELL_TEST OK; else mark DATA_AISHELL_TEST FAILED; ok=0; fi
  if gate_aishell_split_present dev; then mark DATA_AISHELL_DEV OK; else mark DATA_AISHELL_DEV FAILED; ok=0; fi
  if gate_thchs_min_pairs; then mark DATA_THCHS30_PAIRS OK; else mark DATA_THCHS30_PAIRS FAILED; ok=0; fi
  if gate_esc50_min_clips; then mark DATA_ESC50_CLIPS OK; else mark DATA_ESC50_CLIPS FAILED; ok=0; fi
  [ "$ok" -eq 1 ]
}

# ---------------------------------------------------------------------------
# Smoke gates (3 utts/backbone, via the CLI scripts with --limit/--probe 3
# -- fatal for whisper (certificate backbone), degrading (not fatal) for
# conformer).
# ---------------------------------------------------------------------------

stage_smoke_gates() {
  step "Smoke gates (3 utts/backbone via CLI scripts -- split-arg bug class structurally impossible here)"
  local smoke_dir="$RESULTS_DIR/smoke"
  mkdir -p "$smoke_dir"

  if python3 "$SCRIPT_DIR/decode_paraformer.py" --split test --data-root "$DATA_ROOT_AISHELL" \
       --device "$DEVICE" --limit 3 --out "$smoke_dir/smoke_paraformer.jsonl" \
       && assert_decode_content "$smoke_dir/smoke_paraformer.jsonl" 1; then
    mark SMOKE_paraformer OK
  else
    mark SMOKE_paraformer FAILED
    return 1
  fi

  # whisper -- CERTIFICATE backbone: usable scores REQUIRED, else abort
  # before spending grid compute (per the amendment).
  if python3 "$SCRIPT_DIR/decode_whisper.py" --split test --corpus aishell --data-root "$DATA_ROOT_AISHELL" \
       --device "$DEVICE" --limit 3 --out "$smoke_dir/smoke_whisper.jsonl" \
       && assert_decode_content "$smoke_dir/smoke_whisper.jsonl" 1 \
       && assert_usable_scores "$smoke_dir/smoke_whisper.jsonl" 1; then
    mark SMOKE_whisper OK
  else
    mark SMOKE_whisper FAILED
    echo "ABORTING GRID: whisper is a certificate backbone and had zero usable scores in the smoke test" >&2
    return 1
  fi

  # conformer -- probe first (diagnostic dump, never gates), then the real
  # extraction check; DEGRADED (not FAILED) on zero usable scores, per the
  # amendment's roster-derived-m absorption rule.
  python3 "$SCRIPT_DIR/decode_conformer_ms.py" --split test --corpus aishell --data-root "$DATA_ROOT_AISHELL" \
    --device "$DEVICE" --probe 3 --out "$smoke_dir/smoke_conformer.jsonl" \
    || echo "note: conformer --probe 3 exited non-zero -- see $smoke_dir/smoke_conformer.probe.json if present"

  if ! python3 "$SCRIPT_DIR/decode_conformer_ms.py" --split test --corpus aishell --data-root "$DATA_ROOT_AISHELL" \
       --device "$DEVICE" --limit 3 --out "$smoke_dir/smoke_conformer.jsonl"; then
    mark SMOKE_conformer FAILED
    return 1
  fi
  if ! assert_decode_content "$smoke_dir/smoke_conformer.jsonl" 1; then
    mark SMOKE_conformer FAILED
    return 1
  fi
  if assert_usable_scores "$smoke_dir/smoke_conformer.jsonl" 1; then
    mark SMOKE_conformer OK
    CONFORMER_DEGRADED=0
  else
    mark SMOKE_conformer DEGRADED
    echo "conformer smoke: zero usable scores after extraction -- demoting to hyp_text/CER-only roster (grid continues, per EXPANSION-AMENDMENT-2026-07-09.md)"
    CONFORMER_DEGRADED=1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Stage A: clean decode grid.
# ---------------------------------------------------------------------------

stage_a_clean_decode() {
  step "Stage A: clean decode grid {aishell,thchs30} x {paraformer,whisper,conformer}"
  local corpus backbone out
  for corpus in aishell thchs30; do
    for backbone in paraformer whisper conformer; do
      if [ "$backbone" = "paraformer" ] && [ "$corpus" = "thchs30" ]; then
        echo "skip: paraformer x thchs30 (decode_paraformer.py is Aishell-layout-only, disclosed per run_expansion.sh's precedent)"
        mark "DECODE_${corpus}_${backbone}" SKIPPED_DISCLOSED
        continue
      fi
      out="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
      if [ -s "$out" ]; then
        skip "$out"
      else
        decode_backbone "$backbone" "$corpus" test "$out" \
          || echo "warning: decode_backbone $backbone/$corpus exited non-zero -- content gate below will catch it"
      fi
      if assert_decode_content "$out" 1; then
        mark "DECODE_${corpus}_${backbone}" OK
      else
        mark "DECODE_${corpus}_${backbone}" FAILED
      fi
    done
  done
}

# ---------------------------------------------------------------------------
# Stage B: aishell x {5,15,25} dB noise arm (ESC-50 via mix_musan.py's
# neutral --noise-dir-mode, per the amendment's disclosed MUSAN substitution),
# re-decoded with the primary (paraformer) backbone via decode_paraformer.py's
# new --wav-list mode.
# ---------------------------------------------------------------------------

stage_b_noise_arm() {
  step "Stage B: aishell noise-stratified decode (ESC-50, paraformer backbone)"
  local wav_list="$RESULTS_DIR/aishell_test_wavlist.txt"
  if [ ! -s "$wav_list" ]; then
    find "$DATA_ROOT_AISHELL/wav/test" -name '*.wav' -printf '%f %p\n' | sed 's/\.wav / /' > "$wav_list"
  fi

  local snr mixed_dir manifest decoded
  for snr in $SNRS; do
    mixed_dir="$RESULTS_DIR/aishell_test_esc50_${snr}db"
    manifest="$mixed_dir/manifest.jsonl"
    if [ ! -s "$manifest" ]; then
      python3 "$SCRIPT_DIR/mix_musan.py" --wav-list "$wav_list" --musan-dir "$ESC50_DIR" \
        --noise-dir-mode --snr-db "$snr" --out-dir "$mixed_dir" --seed "$SEED" \
        || echo "warning: mix_musan.py exited non-zero for ${snr}dB -- content gate below will catch it"
    else
      skip "$manifest"
    fi
    if assert_mix_content "$manifest" 1; then
      mark "MIX_aishell_${snr}db" OK
    else
      mark "MIX_aishell_${snr}db" FAILED
      continue
    fi

    # manifest utt_id -> mixed wav path, for decode_paraformer.py's
    # --wav-list mode (ref_text resolves via --transcript-source aishell,
    # since mixing preserves utt_id).
    local wavlist_for_decode="$mixed_dir/wavlist_for_decode.txt"
    python3 - "$manifest" "$mixed_dir" <<'PYEOF' > "$wavlist_for_decode"
import json, sys, pathlib
manifest, mixed_dir = sys.argv[1], pathlib.Path(sys.argv[2])
with open(manifest, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        utt_id = rec["utt_id"]
        print(f"{utt_id} {mixed_dir / (utt_id + '.wav')}")
PYEOF

    decoded="$RESULTS_DIR/decode_aishell_musan${snr}db_paraformer_test.jsonl"
    if [ -s "$decoded" ]; then
      skip "$decoded"
    else
      python3 "$SCRIPT_DIR/decode_paraformer.py" --split test --data-root "$DATA_ROOT_AISHELL" \
        --device "$DEVICE" --wav-list "$wavlist_for_decode" --out "$decoded" \
        || echo "warning: decode_paraformer --wav-list exited non-zero for ${snr}dB -- content gate below will catch it"
    fi
    if assert_decode_content "$decoded" 1; then
      mark "DECODE_aishell_musan${snr}db_paraformer" OK
    else
      mark "DECODE_aishell_musan${snr}db_paraformer" FAILED
    fi
  done
}

# ---------------------------------------------------------------------------
# Stage C: per-backbone calibrate-on-aishell-dev -> apply on thchs30
# (cross-corpus) and each noise arm, mirroring run_main.sh's steps exactly
# (per-reseed cal20 scoring, N_RESEEDS reseeds, primary + frontier alphas).
# ---------------------------------------------------------------------------

stage_c_calibrate_apply() {
  step "Stage C: per-backbone calibrate-on-aishell-dev -> apply cross-corpus + noise arms"
  local backbone
  for backbone in paraformer whisper conformer; do
    if [ "$backbone" = "conformer" ] && [ "$CONFORMER_DEGRADED" -eq 1 ]; then
      echo "note: conformer is score-DEGRADED (hyp_text/CER-only) -- calibrate/apply skipped, roster-derived m absorbs the exclusion (Stage D)"
      mark "CALIBRATE_${backbone}" SKIPPED_DEGRADED
      continue
    fi

    local bb_dir="$RESULTS_DIR/backbone_${backbone}"
    mkdir -p "$bb_dir"

    local dev_decode="$bb_dir/decode_dev.jsonl"
    if [ ! -s "$dev_decode" ]; then
      decode_backbone "$backbone" aishell dev "$dev_decode" \
        || echo "warning: dev decode for $backbone exited non-zero -- content gate below will catch it"
    else
      skip "$dev_decode"
    fi
    if ! assert_decode_content "$dev_decode" 1; then
      mark "CALIBRATE_${backbone}" FAILED
      continue
    fi
    local dev_canonical="$bb_dir/dev_canonical.jsonl"
    if [ ! -s "$dev_canonical" ]; then
      asr-gate ingest --hyps "$dev_decode" --format custom-schema --out "$dev_canonical"
    fi

    # Cross-corpus + noise TARGET canonical+scored files, once per backbone.
    ingest_and_score_target "$RESULTS_DIR/decode_thchs30_${backbone}_test.jsonl" "$bb_dir/thchs30_scored.jsonl"
    if [ "$backbone" = "paraformer" ]; then
      local snr
      for snr in $SNRS; do
        ingest_and_score_target \
          "$RESULTS_DIR/decode_aishell_musan${snr}db_paraformer_test.jsonl" \
          "$bb_dir/musan${snr}db_scored.jsonl"
      done
    fi

    local reseed alpha frontier_alphas
    frontier_alphas="$(echo "$ALPHA_FRONTIER" | tr ',' ' ')"
    for reseed in $(seq 0 $((N_RESEEDS - 1))); do
      local reseed_dir="$bb_dir/reseed_${reseed}"
      mkdir -p "$reseed_dir"
      if [ ! -s "$reseed_dir/cal20_scored.jsonl" ]; then
        python3 "$SCRIPT_DIR/split_cal_tune.py" \
          "$dev_canonical" "$reseed_dir/cal20.jsonl" "$reseed_dir/tune20.jsonl" "$reseed"
        asr-gate score --instances "$reseed_dir/cal20.jsonl" --out "$reseed_dir/cal20_scored.jsonl"
      fi

      for alpha in $ALPHA_PRIMARY $frontier_alphas; do
        local gate_json="$reseed_dir/gate_alpha${alpha}.json"
        if [ ! -s "$gate_json" ]; then
          asr-gate calibrate --instances "$reseed_dir/cal20_scored.jsonl" \
            --alpha "$alpha" --delta "$DELTA" --guarantee ltt --strata duration_tercile \
            --seed "$reseed" --ltt-procedure "$LTT_PROCEDURE" --ltt-pvalue "$LTT_PVALUE" \
            --out "$gate_json" \
            || echo "warning: calibrate alpha=$alpha reseed=$reseed backbone=$backbone exited non-zero"
        fi
        [ -s "$gate_json" ] || continue

        if [ -s "$bb_dir/thchs30_scored.jsonl" ]; then
          local applied="$reseed_dir/applied_thchs30_alpha${alpha}.json"
          [ -s "$applied" ] || asr-gate apply --gate "$gate_json" --instances "$bb_dir/thchs30_scored.jsonl" --out "$applied"
        fi
        if [ "$backbone" = "paraformer" ]; then
          local snr
          for snr in $SNRS; do
            [ -s "$bb_dir/musan${snr}db_scored.jsonl" ] || continue
            local applied_noisy="$reseed_dir/applied_musan${snr}db_alpha${alpha}.json"
            [ -s "$applied_noisy" ] || asr-gate apply --gate "$gate_json" --instances "$bb_dir/musan${snr}db_scored.jsonl" --out "$applied_noisy"
          done
        fi
      done
    done
    mark "CALIBRATE_${backbone}" OK
  done
}

# ---------------------------------------------------------------------------
# Stage D: roster-derived Holm audit -- m COMPUTED from realized
# (score,backbone,condition) cells with >=99% non-null, recorded in the
# result JSON. NEVER hardcoded (ABSOLUTE RULE), mirrors
# FREEZE-NOTE-2026-07-09.md's disclosed-roster precedent (m fell 10->2
# there) and asr_gate.audit's own roster-derived convention.
# ---------------------------------------------------------------------------

stage_d_holm_audit() {
  step "Stage D: roster-derived Holm audit"
  local audit_dir="$RESULTS_DIR/audit"
  mkdir -p "$audit_dir"
  local cell_jsons=()

  local backbone corpus cond scored audit_json
  for backbone in paraformer whisper conformer; do
    if [ "$backbone" = "conformer" ] && [ "$CONFORMER_DEGRADED" -eq 1 ]; then
      echo "note: conformer excluded from the score-audit family (CONFORMER_DEGRADED, hyp_text/CER-only per the amendment)"
      continue
    fi
    for corpus in aishell thchs30; do
      local decoded="$RESULTS_DIR/decode_${corpus}_${backbone}_test.jsonl"
      [ -s "$decoded" ] || continue
      scored="$RESULTS_DIR/audit_input_${corpus}_${backbone}_scored.jsonl"
      ingest_and_score_target "$decoded" "$scored"
      [ -s "$scored" ] || continue
      if [ "$corpus" = "aishell" ]; then cond="clean"; else cond="crosscorpus"; fi
      audit_json="$audit_dir/audit_${corpus}_${backbone}_${cond}.json"
      if [ ! -s "$audit_json" ]; then
        asr-gate audit --instances "$scored" --n-perm "$AUDIT_N_PERM" --alpha "$AUDIT_ALPHA" \
          --seed "$SEED" --out "$audit_json" \
          || echo "warning: audit exited non-zero for $corpus/$backbone"
      fi
      [ -s "$audit_json" ] && cell_jsons+=("$audit_json:$backbone:${corpus}_${cond}")
    done
    if [ "$backbone" = "paraformer" ]; then
      local snr
      for snr in $SNRS; do
        scored="$RESULTS_DIR/backbone_paraformer/musan${snr}db_scored.jsonl"
        [ -s "$scored" ] || continue
        audit_json="$audit_dir/audit_aishell_musan${snr}db_paraformer_noise.json"
        if [ ! -s "$audit_json" ]; then
          asr-gate audit --instances "$scored" --n-perm "$AUDIT_N_PERM" --alpha "$AUDIT_ALPHA" \
            --seed "$SEED" --out "$audit_json" \
            || echo "warning: audit exited non-zero for musan${snr}db/paraformer"
        fi
        [ -s "$audit_json" ] && cell_jsons+=("$audit_json:paraformer:musan${snr}db")
      done
    fi
  done

  if [ "${#cell_jsons[@]}" -eq 0 ]; then
    mark AUDIT_HOLM_ROSTER FAILED
    echo "no realized (backbone,condition) audit cells -- nothing to Holm-correct" >&2
    return 1
  fi

  local combined="$RESULTS_DIR/holm_audit_realized.json"
  python3 - "$combined" "$AUDIT_ALPHA" "${cell_jsons[@]}" <<'PYEOF'
import json, sys

out_path, alpha = sys.argv[1], float(sys.argv[2])
cell_specs = sys.argv[3:]

# m is COMPUTED from the realized roster below -- never asserted/hardcoded
# in advance, per EXPANSION-AMENDMENT-2026-07-09.md's Holm-family rule.
rows = []
for spec in cell_specs:
    path, backbone, condition = spec.split(":", 2)
    with open(path, encoding="utf-8") as f:
        cell = json.load(f)
    n_total = cell.get("n", 0)
    for r in cell.get("results", []):
        n_score = r.get("n", 0)
        non_null_frac = (n_score / n_total) if n_total else 0.0
        if non_null_frac < 0.99:
            continue  # amendment's rule: only cells with >=99% non-null enter the family
        rows.append({
            "backbone": backbone,
            "condition": condition,
            "score": r["score"],
            "p_value": r["p_value"],
            "excess_aurc": r.get("excess_aurc"),
            "n": n_score,
            "non_null_frac": non_null_frac,
        })

m = len(rows)

# Standard Holm step-down, recomputed fresh across this REALIZED global
# family (a per-cell `asr-gate audit` call only Holm-corrects within its
# own local family -- this combines every realized cell into one family,
# as the amendment's rule requires).
order = sorted(range(m), key=lambda i: rows[i]["p_value"])
running_max = 0.0
for rank, idx in enumerate(order):
    adj = rows[idx]["p_value"] * (m - rank)
    running_max = max(running_max, adj)
    rows[idx]["p_holm_global"] = min(1.0, running_max)
for row in rows:
    row["reject_holm_global"] = row["p_holm_global"] <= alpha

result = {"m": m, "alpha": alpha, "cells": cell_specs, "rows": rows}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"HOLM_AUDIT_REALIZED m={m} n_cells={len(cell_specs)} -> {out_path}")
PYEOF

  if [ -s "$combined" ]; then
    mark AUDIT_HOLM_ROSTER OK
  else
    mark AUDIT_HOLM_ROSTER FAILED
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Epilogue: tar results, write the ALL_DONE/PARTIAL marker, wait for the
# local watcher's pull ACK, then shut down regardless (results persist on
# the data disk either way). Honors NO_AUTOSHUTDOWN_FILE.
# ---------------------------------------------------------------------------

stage_epilogue() {
  step "Epilogue: tar results, write completion marker, wait for ACK, shutdown"

  mkdir -p "$(dirname "$RESULTS_TARBALL")" 2>/dev/null || true
  if tar czf "$RESULTS_TARBALL" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")"; then
    echo "tarball -> $RESULTS_TARBALL ($(du -h "$RESULTS_TARBALL" 2>/dev/null | cut -f1))"
  else
    echo "warning: tar failed -- results remain uncompressed at $RESULTS_DIR" >&2
  fi

  if [ "${#FAILED_MARKERS[@]}" -eq 0 ]; then
    echo "ALL_DONE $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$RESULTS_DIR/ASR_EXPANSION2_ALL_DONE"
    echo "ASR_EXPANSION2_ALL_DONE"
  else
    {
      echo "PARTIAL $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      printf '%s\n' "${FAILED_MARKERS[@]}"
    } > "$RESULTS_DIR/ASR_EXPANSION2_PARTIAL"
    echo "ASR_EXPANSION2_PARTIAL -- failed markers: ${FAILED_MARKERS[*]}"
  fi

  if [ -f "$NO_AUTOSHUTDOWN_FILE" ]; then
    echo "note: $NO_AUTOSHUTDOWN_FILE present -- skipping shutdown"
    return 0
  fi

  echo "waiting up to ${ACK_TIMEOUT_S}s for $ACK_FILE ..."
  local waited=0
  while [ ! -f "$ACK_FILE" ] && [ "$waited" -lt "$ACK_TIMEOUT_S" ]; do
    sleep 10
    waited=$((waited + 10))
  done
  if [ -f "$ACK_FILE" ]; then
    echo "ACK received after ${waited}s -- results confirmed pulled"
  else
    echo "ACK NOT received after ${ACK_TIMEOUT_S}s -- shutting down anyway (results persist at $RESULTS_DIR and $RESULTS_TARBALL)"
  fi

  echo "shutting down now"
  shutdown -h now 2>/dev/null || shutdown now 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

stage_prologue

GRID_ABORTED=0
if ! stage_data_gates; then
  GRID_ABORTED=1
fi

if [ "$GRID_ABORTED" -eq 0 ] && ! stage_smoke_gates; then
  GRID_ABORTED=1
fi

if [ "$GRID_ABORTED" -eq 0 ]; then
  stage_a_clean_decode
  stage_b_noise_arm
  stage_c_calibrate_apply
  stage_d_holm_audit
else
  echo "GRID ABORTED before Stage A -- see FAILED markers above ($MARKERS_DIR)" >&2
fi

stage_epilogue
