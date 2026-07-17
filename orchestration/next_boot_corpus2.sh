#!/usr/bin/env bash
# next_boot_corpus2.sh -- ROUND-2 external-replication lever for the TASLP asr-gate
# paper: decode ONE SECOND, INDEPENDENT open Mandarin read-test corpus once and run
# a fresh out-of-sample violation check at the frozen operating point, converting the
# round-1 "0/20 reseeds + repartitions of the SAME pools" into an external replication
# no reviewer can call correlated (FINAL-SCORE-R2-2026-07-16 reviewer-1, "single
# highest-leverage improvement"). Modeled VERBATIM in structure on
# thchs_official_redecode.sh (read it): stage corpus -> build frozen pool -> decode ->
# score -> APPLY the frozen aishell-dev per-reseed gates READ-ONLY -> audit -> digest.
#
# ===========================================================================
# CORPUS: Primewords Chinese Corpus Set 1 (openslr-47 / SLR47)
# ===========================================================================
#   * Fully OPEN, direct download, NO form/login (openslr direct):
#       https://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz
#       (verified live 2026-07-16: HTTP 200, Content-Length 9,057,625,192 bytes).
#   * License CC BY-NC-ND 4.0 (non-commercial research); 100h, 296 speakers,
#     ~50k read/prompted Mandarin utterances recorded on mobile.
#   * SPEAKER-labeled AND utterance-labeled: set1_transcript.json carries an
#     explicit per-utterance user_id + text -> enables the speaker-stratified
#     frozen pool below (cleaner than ST-CMDS SLR38, whose speaker info is in
#     per-utt .metadata sidecars; SLR38 verified live too, HTTP 200, 8.23 GB,
#     kept as the documented fallback if SLR47 ever 404s).
#   * NOT already used (used: aishell-1, THCHS-30, MagicData; withdrawn:
#     aidatatang). The WHOLE corpus is out-of-sample by construction: the
#     accept/defer gates are aishell-dev-calibrated and never touch Primewords.
#
# ===========================================================================
# FROZEN SAMPLING RULE -- PRIMEWORDS-POOL-2026-07-16 (frozen BEFORE decode)
# ===========================================================================
# See build_primewords_pool.py header for the full statement. In brief
# (confirmatory-style: the rule is dated 2026-07-16, fixed before the corpus is
# decoded, and never tuned to the result):
#   deterministic seed-0, speaker-stratified, speaker-BALANCED pool of
#   CAP_PER_SPK=8 utts/speaker over the first N_SPK_CAP=ceil(2500/8)=313 eligible
#   speakers (speakers with >=8 utts, ordered by user_id); 8 drawn without
#   replacement via one random.Random(0) in speaker order from each speaker's
#   utt_ids sorted ascending. With ~296 speakers this yields ~2,368 utts across
#   whole speaker strata (the ~2,000-3,000 window; whole strata keep the paper's
#   speaker-bootstrap CIs clean). The exact realized n is read back from the
#   POOL_MANIFEST.json this chain writes and used as the content-gate count.
#
# ===========================================================================
# PULL-TIME MANUSCRIPT EDIT MAP (which tables/prose get the new OOS row)
# ===========================================================================
# All numbers come from CORPUS2_PRIMEWORDS_DIGEST.json (schema mirrors
# THCHS_OFFICIAL_DIGEST.json; per backbone: full_set_macro_cer, full_set_micro_cer,
# n, per_alpha[label].{n_violations,acc_fraction_mean,acc_set_macro_cer_mean,
# certifies_nonvacuous_0_viol}, tightest_cert_alpha, cert_accept_mean,
# cert_acc_cer_mean). Manuscript: manuscripts/taslp/paper_ieeetran.tex.
#
#  (1) tab:landscape (l.991-1018) -- ADD two rows below the MagicData cells:
#        Paraformer (B2) & Primewords & {full_set_macro_cer}% & {tightest_cert_alpha}
#            & {cert_accept_mean}% & {cert_acc_cer_mean}%
#        Belle (B3') & Primewords & ... (same fields from backbones.belle)
#      (VAC. row with --- cells if a backbone certifies at no grid alpha). Update
#      caption: name Primewords (n={n_pool}, openslr-47, CC BY-NC-ND, Aishell-dev-
#      calibrated transfer cell, SECOND independent read corpus); bump "three
#      corpora"->"four" and the "six ... certified cells" count (l.1029-1035) by
#      the number of Primewords cells that certify non-vacuously.
#  (2) Genuine-disjointness / OOS paragraph (l.531-553) -- ADD one sentence, the
#      CORE deliverable of the reviewer ask: an EXTERNAL replication on a second
#      independent corpus never used in calibration -- Primewords set1 (openslr-47,
#      n={n_pool}, {n_speakers_selected} speakers), the frozen Aishell-dev gate at
#      the deployed operating point gives {per_alpha[DEPLOY_ALPHA].n_violations}/20
#      out-of-sample violations (full-set macro-CER {full_set_macro_cer}%), a fresh
#      OOS check that cannot be called a correlated reseed of the same pools.
#      DEPLOY_ALPHA = the paper's headline deployed target (alpha=0.02); ALSO report
#      the tightest_cert_alpha cell for parity with tab:landscape.
#  (3) Corpus/data table (l.353-368) -- ADD a row:
#        Primewords set1 & second independent read corpus (2 backbones) & {n_pool}
#            & CC BY-NC-ND 4.0; openslr-47; frozen seed-0 speaker-stratified pool
#            (PRIMEWORDS-POOL-2026-07-16)
#  (4) tab:deviations -- ADD a provenance line mirroring the THCHS re-decode line:
#        Primewords external replication & frozen seed-0 pool built + decoded
#        {date}; POOL_MANIFEST.json + CORPUS2_PRIMEWORDS_DIGEST.json.
#  (5) Abstract (l.52-53) -- OPTIONAL and only if the deployed-point sentence is
#      also trimmed (reviewer flags the abstract as already too heavy): one clause
#      "...and an external replication on a second independent corpus (Primewords)".
#      Recommend DEFERRING unless the readability trim happens in the same pass.
#
# Usage:
#   bash next_boot_corpus2.sh --dry-run       # local path/CLI/rule check, NO GPU/data, exits
#   bash next_boot_corpus2.sh --smoke-only    # stage+pool+10-utt/backbone smoke+score gate, exits
#   bash next_boot_corpus2.sh                 # full: download->pool->decode->score->apply->audit->digest
#   bash next_boot_corpus2.sh --resume        # full, resume partial belle decode

set -uo pipefail   # NOT -e: content gates / markers / epilogue must run after a failure.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
DRY_RUN=0
SMOKE_ONLY=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY_RUN=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --resume)     RESUME=1 ;;
    -h|--help)    grep -E '^#   bash' "${BASH_SOURCE[0]}" | sed 's/^# //'; exit 0 ;;
    *) echo "unknown flag: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults). Frozen to mirror the landscape.
# ---------------------------------------------------------------------------
CORPUS="${CORPUS:-primewords}"
PRIMEWORDS_URL="${PRIMEWORDS_URL:-https://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz}"
PRIMEWORDS_TAR_BYTES="${PRIMEWORDS_TAR_BYTES:-9057625192}"     # verified 2026-07-16
TARBALL="${TARBALL:-/root/autodl-tmp/primewords_md_2018_set1.tar.gz}"
SRC_RAW="${SRC_RAW:-/root/autodl-tmp/primewords_raw}"          # extraction root (holds primewords_md_2018_set1/)
STAGE="${STAGE:-/root/autodl-tmp/primewords_pool}"             # clean frozen-pool root the discoverer returns in full
LAND="${LAND:-/root/autodl-tmp/asr_landscape_results}"         # FROZEN landscape (read-only: aishell-dev gates)
OUT="${OUT:-/root/autodl-tmp/corpus2_primewords_2026-07-16}"   # ALL outputs land here
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

BELLE_MODEL="${BELLE_MODEL:-BELLE-2/Belle-whisper-large-v3-zh}"

# Frozen certificate config (FREEZE-AMENDMENT-2026-07-13 §4). Gate filenames on
# disk keep the trailing 0 (gate_alpha0.10.json) -- match the box's landscape run.
ALPHA_LABELS="${ALPHA_LABELS:-0.015 0.02 0.03 0.05 0.10}"
DELTA="${DELTA:-0.1}"
N_RESEEDS="${N_RESEEDS:-20}"
AUDIT_ALPHA="${AUDIT_ALPHA:-0.05}"
AUDIT_N_PERM="${AUDIT_N_PERM:-2000}"

# paraformer + belle only; zipformer SKIPPED (documented unusable posteriors --
# THCHS_OFFICIAL_DIGEST full_set_macro_cer 0.82, landscape 32-82% CER, VAC. everywhere).
BACKBONES="${BACKBONES:-paraformer belle}"
SMOKE_N="${SMOKE_N:-10}"

# Frozen pool rule params (PRIMEWORDS-POOL-2026-07-16). Defaults ARE the rule.
POOL_CAP_PER_SPK="${POOL_CAP_PER_SPK:-8}"
POOL_TARGET="${POOL_TARGET:-2500}"
POOL_BUILDER="$SCRIPT_DIR/build_primewords_pool.py"
POOL_MANIFEST="${POOL_MANIFEST:-$STAGE/POOL_MANIFEST.json}"

HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG="${LOG:-/root/corpus2_primewords.log}"
[ "$SMOKE_ONLY" -eq 1 ] && LOG="${SMOKE_LOG:-/root/corpus2_primewords_smoke.log}"
GPU_UTIL_LOG="${GPU_UTIL_LOG:-/root/gpu_util_corpus2.log}"
GPU_LOG_PIDFILE="${GPU_LOG_PIDFILE:-/root/gpu_util_corpus2.pid}"
MARKERS_DIR="${MARKERS_DIR:-$OUT/markers}"
DIGEST_JSON="${DIGEST_JSON:-$OUT/CORPUS2_PRIMEWORDS_DIGEST.json}"
DONE_SENTINEL="CORPUS2_ALL_DONE"

FAILED_MARKERS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }

mark() {
  local name="$1" status="$2"
  mkdir -p "$MARKERS_DIR"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  [ "$status" = "FAILED" ] && FAILED_MARKERS+=("$name")
  return 0
}

# Content gate: exactly want_rows rows, >=50% non-empty hyp_text (DOFA lesson).
assert_decode_content() {
  local path="$1" want_rows="$2"
  [ -s "$path" ] || { echo "CONTENT_GATE_FAIL $path: missing/empty" >&2; return 1; }
  python3 - "$path" "$want_rows" <<'PYEOF'
import json, sys
path, want = sys.argv[1], int(sys.argv[2])
n = nonempty = 0
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    rec = json.loads(line); n += 1
    if rec.get("hyp_text"): nonempty += 1
frac = nonempty / n if n else 0.0
if n != want:
    print(f"CONTENT_GATE_FAIL {path}: n={n} != {want}", file=sys.stderr); sys.exit(1)
if frac < 0.5:
    print(f"CONTENT_GATE_FAIL {path}: nonempty_frac {frac:.3f} < 0.5", file=sys.stderr); sys.exit(1)
print(f"CONTENT_GATE_OK {path}: n={n} nonempty_frac={frac:.3f}")
PYEOF
}

# Every scored instance carries a finite 'cer'; print macro-CER.
assert_scored_cer_finite() {
  local path="$1" want_rows="${2:-}"
  [ -s "$path" ] || { echo "CER_GATE_FAIL $path: missing/empty" >&2; return 1; }
  python3 - "$path" "$want_rows" <<'PYEOF'
import json, math, sys
path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
n = 0; cers = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    rec = json.loads(line); n += 1
    c = rec.get("cer")
    if c is None or not math.isfinite(float(c)):
        print(f"CER_GATE_FAIL {path}: utt {rec.get('utt_id')} cer={c}", file=sys.stderr); sys.exit(1)
    for fld in ("hyp_text", "ref_text", "nbest"):
        if fld not in rec:
            print(f"CER_GATE_FAIL {path}: utt {rec.get('utt_id')} missing {fld}", file=sys.stderr); sys.exit(1)
    cers.append(float(c))
if want is not None and n != want:
    print(f"CER_GATE_FAIL {path}: n={n} != {want}", file=sys.stderr); sys.exit(1)
macro = sum(cers)/len(cers) if cers else float("nan")
print(f"CER_GATE_OK {path}: n={n} macro_cer={macro:.5f}")
PYEOF
}

# Read the realized frozen pool size back from the manifest (content-gate count).
pool_n() {
  [ -s "$POOL_MANIFEST" ] || { echo "0"; return 1; }
  python3 - "$POOL_MANIFEST" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(d.get("n_pool_utts") or 0))
PYEOF
}

# ---------------------------------------------------------------------------
# DRY-RUN: validate paths, CLIs, the discoverer registration, the frozen rule,
# and the frozen-gate tree -- NO GPU, NO data download, NO decode. Exits.
# ---------------------------------------------------------------------------
stage_dry_run() {
  step "DRY-RUN (local path/CLI/rule check; no GPU, no data)"
  local ok=1
  echo "-- python + asr-gate --"
  command -v python3 >/dev/null 2>&1 && echo "python3: $(command -v python3)" || { echo "MISSING python3" >&2; ok=0; }
  command -v asr-gate >/dev/null 2>&1 && echo "asr-gate: $(command -v asr-gate)" \
    || echo "WARN asr-gate not on PATH (installed on box via pip -e .)"
  echo "-- decode CLIs + pool builder present --"
  for f in decode_paraformer.py decode_whisper.py build_primewords_pool.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then echo "ok   $f"; else echo "MISSING $f" >&2; ok=0; fi
  done
  echo "-- corpus '$CORPUS' registered in asr_gate.corpora --"
  if python3 - "$SCRIPT_DIR/.." "$CORPUS" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
c = sys.argv[2]
ok = c in corpora.CORPUS_DISCOVERERS
print(f"CORPUS_DISCOVERERS={sorted(corpora.CORPUS_DISCOVERERS)}")
print("CORPUS_REGISTERED_OK" if ok else f"CORPUS_MISSING {c}")
sys.exit(0 if ok else 1)
PYEOF
  then echo "ok   discoverer registered"; else echo "FAIL discoverer not registered" >&2; ok=0; fi
  echo "-- frozen sampling rule --"
  python3 "$POOL_BUILDER" --print-rule || ok=0
  echo "-- pool builder + decode scripts py_compile --"
  if python3 -m py_compile "$POOL_BUILDER" "$SCRIPT_DIR/decode_paraformer.py" "$SCRIPT_DIR/decode_whisper.py"; then
    echo "ok   py_compile"
  else echo "FAIL py_compile" >&2; ok=0; fi
  echo "-- frozen aishell-dev gate tree (read-only): $LAND --"
  if [ -d "$LAND" ]; then
    local bb label miss=0 seen=0
    for bb in $BACKBONES; do
      for label in $ALPHA_LABELS; do
        local g="$LAND/backbone_${bb}/reseed_0/gate_alpha${label}.json"
        if [ -s "$g" ]; then seen=$((seen+1)); else echo "  MISSING gate: $g" >&2; miss=$((miss+1)); fi
      done
    done
    echo "  gates present at reseed_0: $seen ; missing: $miss (full run needs reseeds 0..$((N_RESEEDS-1)))"
    [ "$miss" -gt 0 ] && echo "  NOTE: gates are staged on the box at \$LAND; absent here is expected off-box"
  else
    echo "  NOTE: \$LAND not present here ($LAND) -- expected; it is the box's frozen landscape tree"
  fi
  echo "-- live corpus URL (headers only) --"
  if command -v curl >/dev/null 2>&1; then
    curl -sSL -I --max-time 30 "$PRIMEWORDS_URL" 2>&1 | grep -iE "^HTTP|content-length" | head -4 \
      || echo "  (curl header probe skipped/failed -- non-fatal in dry-run)"
  fi
  echo "-- planned outputs --"
  echo "  OUT=$OUT  STAGE=$STAGE  DIGEST=$DIGEST_JSON  SENTINEL=$OUT/$DONE_SENTINEL"
  echo "  BACKBONES='$BACKBONES'  ALPHA_LABELS='$ALPHA_LABELS'  N_RESEEDS=$N_RESEEDS"
  echo "DRY_RUN $([ "$ok" -eq 1 ] && echo OK || echo FAIL)"
  return $([ "$ok" -eq 1 ] && echo 0 || echo 1)
}

# ---------------------------------------------------------------------------
# Prologue: conda base, HF env, network_turbo, GPU logger, tee log.
# ---------------------------------------------------------------------------
stage_prologue() {
  step "Prologue"
  local conda_sh=""
  for c in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh \
           "${HOME:-/root}/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && { conda_sh="$c"; break; }
  done
  if [ -n "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"; conda activate base || echo "warning: conda activate base failed" >&2
  else
    echo "warning: no conda.sh found -- current environment" >&2
  fi
  # NETWORK ROUTING (fixed 2026-07-16 after audit): do NOT source network_turbo
  # globally -- its exported proxies slow/break non-github/HF hosts (openslr!).
  # The belle pull uses HF_ENDPOINT=hf-mirror.com DIRECT (proven ~9MB/s from
  # this fleet without any proxy); openslr is a direct host. Proxies are
  # explicitly unset here in case the calling shell carried them.
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
  echo "network: direct (hf-mirror for HF; no proxy exported)"
  export HF_HOME HF_ENDPOINT
  echo "HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
  mkdir -p "$OUT" "$MARKERS_DIR" "$OUT/audit" "$OUT/smoke" "$SRC_RAW" "$STAGE" 2>/dev/null || true
  exec > >(tee -a "$LOG") 2>&1
  echo "logging to $LOG  ($(date -u +%FT%TZ))"
  command -v asr-gate >/dev/null 2>&1 && echo "asr-gate: $(command -v asr-gate)" || echo "warning: asr-gate not on PATH" >&2
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
# Stage the corpus: download the openslr-47 tarball (idempotent, size-gated) and
# extract it under $SRC_RAW/primewords_md_2018_set1/. openslr is a DIRECT host
# (turbo does not accelerate it); prefer aria2c, fall back to wget -c.
# ---------------------------------------------------------------------------
stage_download() {
  step "Download + extract Primewords (openslr-47)"
  local base="$SRC_RAW/primewords_md_2018_set1"
  if [ -s "$base/set1_transcript.json" ] && [ -d "$base/audio_files" ]; then
    skip "$base (already extracted)"; mark STAGE_CORPUS OK; return 0
  fi
  # download (resume-safe, idempotent on size)
  local have=0
  [ -f "$TARBALL" ] && have=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
  if [ "$have" -lt "$PRIMEWORDS_TAR_BYTES" ]; then
    echo "downloading $PRIMEWORDS_URL -> $TARBALL (have $have / $PRIMEWORDS_TAR_BYTES bytes)"
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -c -x8 -s8 -k1M --dir "$(dirname "$TARBALL")" -o "$(basename "$TARBALL")" "$PRIMEWORDS_URL" \
        || { echo "aria2c failed; retrying with wget" >&2; wget -c -O "$TARBALL" "$PRIMEWORDS_URL"; }
    else
      wget -c -O "$TARBALL" "$PRIMEWORDS_URL"
    fi
  else
    echo "tarball already complete ($have bytes)"
  fi
  have=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
  # accept >=99% of expected (mirror content-lengths vary slightly); hard-fail if short.
  if [ "$have" -lt $((PRIMEWORDS_TAR_BYTES * 99 / 100)) ]; then
    echo "FATAL: $TARBALL is $have bytes, expected ~$PRIMEWORDS_TAR_BYTES" >&2
    mark STAGE_CORPUS FAILED; return 1
  fi
  echo "extracting -> $SRC_RAW/"
  mkdir -p "$SRC_RAW"
  tar -xzf "$TARBALL" -C "$SRC_RAW" || { echo "FATAL: extract failed" >&2; mark STAGE_CORPUS FAILED; return 1; }
  # tarball may nest under primewords_md_2018_set1/ OR a wrapper dir -- normalize
  if [ ! -s "$base/set1_transcript.json" ]; then
    local found; found=$(find "$SRC_RAW" -maxdepth 3 -name set1_transcript.json 2>/dev/null | head -1)
    [ -n "$found" ] && base="$(dirname "$found")"
  fi
  if [ -s "$base/set1_transcript.json" ] && [ -d "$base/audio_files" ]; then
    echo "extracted: base=$base"; mark STAGE_CORPUS OK; return 0
  fi
  echo "FATAL: post-extract layout not found (no set1_transcript.json + audio_files/)" >&2
  mark STAGE_CORPUS FAILED; return 1
}

# ---------------------------------------------------------------------------
# Build the FROZEN speaker-stratified pool (PRIMEWORDS-POOL-2026-07-16) into
# $STAGE via build_primewords_pool.py. Idempotent + deterministic. Content gate:
# the builder's own POOL_STAGE_GATE_OK (staged wav == transcript == n_pool).
# ---------------------------------------------------------------------------
stage_build_pool() {
  step "Build frozen pool ($STAGE) via $POOL_BUILDER"
  if python3 "$POOL_BUILDER" --src "$SRC_RAW" --stage "$STAGE" --manifest "$POOL_MANIFEST" \
       --seed "$SEED" --cap-per-spk "$POOL_CAP_PER_SPK" --target "$POOL_TARGET"; then
    local n; n=$(pool_n)
    echo "frozen pool n=$n (manifest $POOL_MANIFEST)"
    # independent proof the shared discoverer returns exactly the frozen pool
    python3 - "$SCRIPT_DIR/.." "$STAGE" "$n" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
root, want = sys.argv[2], int(sys.argv[3])
utts = corpora.discover_corpus("primewords", root, "train")
n = len(utts); with_ref = sum(1 for u in utts if u.ref_text)
spk = sorted({u.speaker_id for u in utts})
print(f"discover primewords @ {root}: n={n} with_ref={with_ref} n_speakers={len(spk)}")
ok = (n == want) and (with_ref == want)
print("POOL_DISCOVER_OK" if ok else "POOL_DISCOVER_FAIL"); sys.exit(0 if ok else 1)
PYEOF
    if [ "$?" -eq 0 ] && [ "$n" -gt 0 ]; then mark POOL_FROZEN OK; return 0; fi
  fi
  mark POOL_FROZEN FAILED; return 1
}

# ---------------------------------------------------------------------------
# Decode dispatch (paraformer + belle only; frozen-pool STAGE data-root).
# ---------------------------------------------------------------------------
decode_backbone() {
  local backbone="$1" out="$2" limit_flag="$3"
  local rf=""; [ "$RESUME" -eq 1 ] && rf="--resume"
  case "$backbone" in
    paraformer)   # decode_paraformer.py has --skip-existing but NO --resume.
      python3 "$SCRIPT_DIR/decode_paraformer.py" --corpus "$CORPUS" --split train \
        --data-root "$STAGE" --device "$DEVICE" --skip-existing $limit_flag --out "$out" ;;
    belle)
      python3 "$SCRIPT_DIR/decode_whisper.py" --corpus "$CORPUS" --split train \
        --data-root "$STAGE" --model-name "$BELLE_MODEL" --language zh \
        --device "$DEVICE" --skip-existing $rf $limit_flag --out "$out" ;;
    *) echo "decode_backbone: unknown backbone $backbone (paraformer/belle only)" >&2; return 1 ;;
  esac
}

# ingest (custom-schema) -> score, exactly like the landscape stage C.
score_decode() {
  local decoded="$1" canonical="$2" scored="$3"
  [ -s "$scored" ] && { skip "$scored"; return 0; }
  asr-gate ingest --hyps "$decoded" --format custom-schema --out "$canonical" \
    && asr-gate score --instances "$canonical" --out "$scored"
}

# ---------------------------------------------------------------------------
# SMOKE (--smoke-only): 10 utts/backbone -> ingest+score -> CER-finite gate.
# ---------------------------------------------------------------------------
stage_smoke() {
  step "Smoke: decode $SMOKE_N utts/backbone -> score -> CER-finite gate"
  local bb
  for bb in $BACKBONES; do
    local dec="$OUT/smoke/smoke_${CORPUS}_${bb}.jsonl"
    local can="$OUT/smoke/smoke_${CORPUS}_${bb}_canonical.jsonl"
    local sc="$OUT/smoke/smoke_${CORPUS}_${bb}_scored.jsonl"
    rm -f "$dec" "$can" "$sc"
    echo "-- smoke decode: $bb --"
    if decode_backbone "$bb" "$dec" "--limit $SMOKE_N" \
       && assert_decode_content "$dec" "$SMOKE_N" \
       && score_decode "$dec" "$can" "$sc" \
       && assert_scored_cer_finite "$sc"; then
      mark "SMOKE_${bb}" OK
    else
      mark "SMOKE_${bb}" FAILED
    fi
  done
  [ "${#FAILED_MARKERS[@]}" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Full pipeline per backbone: decode N_POOL -> score -> apply (frozen aishell-dev
# gates, 20 reseeds x alpha grid, READ-ONLY) -> audit. paraformer first.
# ---------------------------------------------------------------------------
stage_pipeline() {
  local n_pool; n_pool=$(pool_n)
  step "Full pipeline (decode $n_pool -> score -> apply frozen gates -> audit)"
  if [ "$n_pool" -le 0 ]; then echo "FATAL: pool size unknown/zero" >&2; return 1; fi
  local bb
  for bb in $BACKBONES; do
    echo "########## BACKBONE $bb ##########"
    local bb_dir="$OUT/backbone_${bb}"; mkdir -p "$bb_dir"
    local dec="$OUT/decode_${CORPUS}_${bb}.jsonl"
    local can="$bb_dir/${CORPUS}_canonical.jsonl"
    local scored="$bb_dir/${CORPUS}_scored.jsonl"

    # (1) decode
    if [ -s "$dec" ] && assert_decode_content "$dec" "$n_pool" >/dev/null 2>&1; then
      skip "$dec"
    else
      decode_backbone "$bb" "$dec" ""
    fi
    if assert_decode_content "$dec" "$n_pool"; then mark "DECODE_${bb}" OK
    else mark "DECODE_${bb}" FAILED; echo "skip score/apply/audit for $bb (decode gate failed)"; continue; fi

    # (2) score
    if score_decode "$dec" "$can" "$scored" && assert_scored_cer_finite "$scored" "$n_pool"; then
      mark "SCORE_${bb}" OK
    else
      mark "SCORE_${bb}" FAILED; echo "skip apply/audit for $bb (score gate failed)"; continue
    fi

    # (3a) apply frozen aishell-dev gates (READ-ONLY) across 20 reseeds x alpha grid
    local reseed label gate applied n_applied=0 n_want=0
    for reseed in $(seq 0 $((N_RESEEDS - 1))); do
      local rdir="$bb_dir/reseed_${reseed}"; mkdir -p "$rdir"
      for label in $ALPHA_LABELS; do
        n_want=$((n_want + 1))
        gate="$LAND/backbone_${bb}/reseed_${reseed}/gate_alpha${label}.json"
        applied="$rdir/applied_${CORPUS}_alpha${label}.json"
        if [ -s "$applied" ]; then n_applied=$((n_applied + 1)); continue; fi
        if [ ! -s "$gate" ]; then echo "WARN missing frozen gate: $gate" >&2; continue; fi
        if asr-gate apply --gate "$gate" --instances "$scored" --out "$applied" >/dev/null 2>&1 \
           && [ -s "$applied" ]; then
          n_applied=$((n_applied + 1))
        else
          echo "WARN apply failed: reseed=$reseed alpha=$label" >&2
        fi
      done
    done
    echo "applied $n_applied / $n_want ($bb)"
    if [ "$n_applied" -eq "$n_want" ]; then mark "APPLY_${bb}" OK; else mark "APPLY_${bb}" FAILED; fi

    # (3b) audit scored (excess-AURC + perm-p + Holm), same flags as landscape
    local audit_json="$OUT/audit/audit_${CORPUS}_${bb}.json"
    if [ -s "$audit_json" ]; then
      skip "$audit_json"
    else
      asr-gate audit --instances "$scored" --n-perm "$AUDIT_N_PERM" --alpha "$AUDIT_ALPHA" \
        --seed "$SEED" --out "$audit_json" || echo "WARN audit exited non-zero ($bb)" >&2
    fi
    if python3 - "$audit_json" "$n_pool" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); want = int(sys.argv[2])
print(f"audit {sys.argv[1]}: n={d.get('n')} macro_cer={d.get('macro_cer')} "
      f"micro_cer={(d.get('micro_cer') or {}).get('point')} holm_m={d.get('holm_family_size')}")
sys.exit(0 if d.get("n") == want else 1)
PYEOF
    then mark "AUDIT_${bb}" OK; else mark "AUDIT_${bb}" FAILED; fi
  done
}

# ---------------------------------------------------------------------------
# Digest: schema MIRRORS THCHS_OFFICIAL_DIGEST.json so the Primewords row drops
# straight into tab:landscape -- per-backbone full-set CER + per-alpha 20-reseed
# mean acceptance / accepted-set macro-CER / #violations, and tightest-cert alpha.
# ---------------------------------------------------------------------------
stage_digest() {
  step "Digest (per-backbone: full-set CER + per-alpha attainment + tightest cert alpha)"
  python3 - "$OUT" "$N_RESEEDS" "$ALPHA_LABELS" "$DIGEST_JSON" "$BACKBONES" "$CORPUS" "$POOL_MANIFEST" <<'PYEOF'
import glob, json, os, sys
import numpy as np
OUT, N_RESEEDS = sys.argv[1], int(sys.argv[2])
LABELS = sys.argv[3].split()
DIGEST = sys.argv[4]
BACKBONES = sys.argv[5].split()
CORPUS = sys.argv[6]
POOL_MANIFEST = sys.argv[7]

def load(p):
    with open(p) as f: return json.load(f)
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def cer_map(scored):
    return {r["utt_id"]: r["cer"] for r in load_jsonl(scored)}

def accepted_set_stats(applied, cmap, alpha):
    a = load(applied); dec = a["decisions"]
    n_total = a["n"]
    acc = [d["utt_id"] for d in dec if d["action"] == "ACCEPT"]
    cers = [cmap[u] for u in acc if u in cmap]
    frac = len(acc)/n_total if n_total else 0.0
    macro = float(np.mean(cers)) if cers else None
    return {"acc_fraction": frac, "acc_set_macro_cer": macro,
            "violation": bool(cers and macro > alpha), "vacuous": len(acc) == 0}

pool_meta = {}
if os.path.exists(POOL_MANIFEST):
    try: pool_meta = load(POOL_MANIFEST)
    except Exception: pool_meta = {}

out = {"corpus": CORPUS, "rule_id": pool_meta.get("rule_id"),
       "n_pool": pool_meta.get("n_pool_utts"),
       "n_speakers_selected": pool_meta.get("n_speakers_selected"),
       "backbones": {}}
for bb in BACKBONES:
    bb_dir = os.path.join(OUT, f"backbone_{bb}")
    scored = os.path.join(bb_dir, f"{CORPUS}_scored.jsonl")
    audit = os.path.join(OUT, "audit", f"audit_{CORPUS}_{bb}.json")
    cell = {"full_set_macro_cer": None, "full_set_micro_cer": None, "n": None, "per_alpha": {}}
    if os.path.exists(audit):
        d = load(audit); cell["full_set_macro_cer"] = d.get("macro_cer")
        mc = d.get("micro_cer"); cell["full_set_micro_cer"] = mc.get("point") if isinstance(mc, dict) else mc
        cell["n"] = d.get("n")
        if out["n_pool"] is None: out["n_pool"] = d.get("n")
    if os.path.exists(scored):
        cmap = cer_map(scored)
        tightest = None
        for label in LABELS:
            alpha = float(label)
            per = []
            for r in range(N_RESEEDS):
                ap = os.path.join(bb_dir, f"reseed_{r}", f"applied_{CORPUS}_alpha{label}.json")
                if os.path.exists(ap): per.append(accepted_set_stats(ap, cmap, alpha))
            if not per: continue
            accs = [p["acc_fraction"] for p in per]
            mcs = [p["acc_set_macro_cer"] for p in per if p["acc_set_macro_cer"] is not None]
            n_viol = sum(1 for p in per if p["violation"])
            n_vac = sum(1 for p in per if p["vacuous"])
            nonvac = (n_vac == 0)
            summ = {"n_reseeds": len(per),
                    "acc_fraction_mean": float(np.mean(accs)),
                    "acc_set_macro_cer_mean": (float(np.mean(mcs)) if mcs else None),
                    "n_violations": n_viol, "n_vacuous": n_vac,
                    "certifies_nonvacuous_0_viol": bool(nonvac and n_viol == 0)}
            cell["per_alpha"][label] = summ
            if tightest is None and nonvac and n_viol == 0:
                tightest = label
        cell["tightest_cert_alpha"] = tightest
        if tightest:
            t = cell["per_alpha"][tightest]
            cell["cert_accept_mean"] = t["acc_fraction_mean"]
            cell["cert_acc_cer_mean"] = t["acc_set_macro_cer_mean"]
    out["backbones"][bb] = cell

json.dump(out, open(DIGEST, "w"), indent=2, ensure_ascii=False)
print(f"wrote {DIGEST}")
print("=== PRIMEWORDS (n={}) external-replication digest ===".format(out["n_pool"]))
print(f"{'backbone':12} {'full-CER':>8} {'tight-a':>8} {'accept':>8} {'acc-CER':>8} {'a=2% viol':>10}")
for bb in BACKBONES:
    c = out["backbones"][bb]
    fc = c["full_set_macro_cer"]; ta = c.get("tightest_cert_alpha")
    ac = c.get("cert_accept_mean"); acc = c.get("cert_acc_cer_mean")
    v02 = (c.get("per_alpha", {}).get("0.02") or {}).get("n_violations")
    print("{:12} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
        bb,
        f"{fc*100:.2f}%" if fc is not None else "-",
        (f"{float(ta)*100:g}%" if ta else "VAC."),
        f"{ac*100:.1f}%" if ac is not None else "-",
        f"{acc*100:.2f}%" if acc is not None else "-",
        f"{v02}/{N_RESEEDS}" if v02 is not None else "-"))
PYEOF
}

# ---------------------------------------------------------------------------
# Epilogue
# ---------------------------------------------------------------------------
stage_epilogue() {
  step "Epilogue"
  if [ "${#FAILED_MARKERS[@]}" -eq 0 ]; then
    echo "ALL_DONE $(date -u +%FT%TZ)" > "$OUT/$DONE_SENTINEL"
    echo "$DONE_SENTINEL"
  else
    { echo "PARTIAL $(date -u +%FT%TZ)"; printf '%s\n' "${FAILED_MARKERS[@]}"; } > "$OUT/CORPUS2_PARTIAL"
    echo "CORPUS2_PARTIAL -- failed: ${FAILED_MARKERS[*]}"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  stage_dry_run
  exit $?
fi

stage_prologue
stage_download   || { echo "CORPUS STAGE FAILED -- aborting" >&2; stage_epilogue; exit 1; }
stage_build_pool || { echo "POOL BUILD FAILED -- aborting" >&2; stage_epilogue; exit 1; }

if [ "$SMOKE_ONLY" -eq 1 ]; then
  stage_smoke
  echo "smoke-only done ($(date -u +%FT%TZ)); failed markers: ${FAILED_MARKERS[*]:-none}"
  exit $([ "${#FAILED_MARKERS[@]}" -eq 0 ] && echo 0 || echo 1)
fi

stage_pipeline
stage_digest
stage_epilogue
