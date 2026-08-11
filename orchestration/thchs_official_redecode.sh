#!/usr/bin/env bash
# thchs_official_redecode.sh -- re-decode the THCHS-30 LANDSCAPE cells on the
# OFFICIAL n=2,495 test (the ten D-prefix speakers) instead of the contaminated
# n=1,339 A/B/C/D mirror pool the frozen landscape run used. This is the planned
# upgrade recorded in the paper's tab:deviations (DATA-AUDIT-2026-07-16 finding
# C2). It DOES NOT touch the frozen landscape results -- all official outputs go
# to a SEPARATE $OUT tree, and the aishell-dev-calibrated per-reseed GATES are
# REUSED read-only (calibration never depends on the THCHS test pool, so only
# score/apply/audit are re-run).
#
# Conventions inherited VERBATIM from next_boot_asr_landscape.sh:
#   * decode ONLY through the verified decode_*.py CLIs (--corpus thchs30 --split
#     test routes through asr_gate.corpora.discover_corpus);
#   * every completion MARKER asserts on RESULT CONTENT (row count == 2495,
#     non-null hyp fraction), never a bare exit code (the DOFA lesson);
#   * all tunables env-overridable with NAMED DEFAULTS; model IDs frozen-default.
#
# CRUX (why a staging pool): discover_thchs30(root,"test") prefers {root}/test/
# (the 1,339 mirror) and only falls back to {root}/data/ -- but the REAL
# data_thchs30/data/ holds the FULL corpus (13,388 wavs, A/B/C/D). So we stage a
# clean root whose data/ contains ONLY the 2,495 D-prefix wavs + their .wav.trn
# sidecars (symlinks), and NO test/ subdir; the discoverer then globs exactly the
# official test. (Verified 2026-07-16: data/D*.wav = 2495 over 10 speakers.)
#
# Usage:
#   bash thchs_official_redecode.sh --smoke-only   # pool build + 10-utt smoke/backbone, exits
#   bash thchs_official_redecode.sh                # full: decode 2495 + score + apply + audit
#   bash thchs_official_redecode.sh --resume       # full, resume partial decodes (whisper/zipformer)

set -uo pipefail   # NOT -e: content gates / markers / epilogue must run after a failure.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SMOKE_ONLY=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --resume)     RESUME=1 ;;
    -h|--help)    grep -E '^#   bash' "${BASH_SOURCE[0]}" | sed 's/^# //'; exit 0 ;;
    *) echo "unknown flag: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults). Frozen to mirror the landscape.
# ---------------------------------------------------------------------------
SRC_THCHS="${SRC_THCHS:-${AUTODL_TMP}/data_thchs30}"      # real extraction (data/ = full corpus)
STAGE="${STAGE:-${AUTODL_TMP}/data_thchs30_official}"     # clean root: data/ = D* only, no test/
LAND="${LAND:-${AUTODL_TMP}/asr_landscape_results}"       # FROZEN landscape (read-only: gates)
OUT="${OUT:-${AUTODL_TMP}/thchs_official_2026-07-16}"     # ALL official outputs land here
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

BELLE_MODEL="${BELLE_MODEL:-BELLE-2/Belle-whisper-large-v3-zh}"
ZIPFORMER_DIR="${ZIPFORMER_DIR:-${AUTODL_TMP}/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2}"

# Frozen certificate config (FREEZE-AMENDMENT-2026-07-13 §4). On-disk alpha labels
# match the gate filenames the box wrote (gate_alpha0.10.json keeps the trailing 0).
ALPHA_LABELS="${ALPHA_LABELS:-0.015 0.02 0.03 0.05 0.10}"
DELTA="${DELTA:-0.1}"
N_RESEEDS="${N_RESEEDS:-20}"
AUDIT_ALPHA="${AUDIT_ALPHA:-0.05}"
AUDIT_N_PERM="${AUDIT_N_PERM:-2000}"

BACKBONES="${BACKBONES:-paraformer belle zipformer}"   # B2, B3', B4 (order = fastest first)
SMOKE_N="${SMOKE_N:-10}"
N_OFFICIAL="${N_OFFICIAL:-2495}"                        # expected official-test row count

HF_HOME="${HF_HOME:-${AUTODL_TMP}/hf-cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG="${LOG:-/root/thchs_redecode.log}"
[ "$SMOKE_ONLY" -eq 1 ] && LOG="${SMOKE_LOG:-/root/thchs_redecode_smoke.log}"
GPU_UTIL_LOG="${GPU_UTIL_LOG:-/root/gpu_util_thchs_redecode.log}"
GPU_LOG_PIDFILE="${GPU_LOG_PIDFILE:-/root/gpu_util_thchs_redecode.pid}"
MARKERS_DIR="${MARKERS_DIR:-$OUT/markers}"
DIGEST_JSON="${DIGEST_JSON:-$OUT/THCHS_OFFICIAL_DIGEST.json}"
DONE_SENTINEL="THCHS_OFFICIAL_ALL_DONE"

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
  # network_turbo accelerates github/HF (unset after model pulls are cached anyway).
  [ -f /etc/network_turbo ] && { source /etc/network_turbo || true; echo "sourced /etc/network_turbo"; }
  export HF_HOME HF_ENDPOINT
  echo "HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
  mkdir -p "$OUT" "$MARKERS_DIR" "$OUT/audit" "$OUT/smoke" 2>/dev/null || true
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
# Build the clean official-test staging pool: {STAGE}/data/ = D*.wav + D*.wav.trn
# (symlinks into the real extraction), NO {STAGE}/test/ (so the discoverer's
# data/ fallback returns exactly the official 2,495). Idempotent.
# ---------------------------------------------------------------------------
stage_build_pool() {
  step "Build official-test staging pool ($STAGE/data <- $SRC_THCHS/data/D*)"
  if [ -d "$STAGE/test" ]; then
    echo "FATAL: $STAGE/test exists -- would shadow the official data/ fallback with a mirror split" >&2
    mark POOL_OFFICIAL FAILED; return 1
  fi
  mkdir -p "$STAGE/data"
  # Symlink ONLY D-prefix wavs and their .wav.trn sidecars (glob D*.wav excludes
  # the .wav.trn files, which are linked separately).
  local n_src; n_src=$(ls "$SRC_THCHS"/data/D*.wav 2>/dev/null | wc -l)
  if [ "$n_src" -ne "$N_OFFICIAL" ]; then
    echo "FATAL: $SRC_THCHS/data has $n_src D*.wav, expected $N_OFFICIAL" >&2
    mark POOL_OFFICIAL FAILED; return 1
  fi
  ln -sf "$SRC_THCHS"/data/D*.wav      "$STAGE/data/"
  ln -sf "$SRC_THCHS"/data/D*.wav.trn  "$STAGE/data/"
  local n_wav n_trn n_spk
  n_wav=$(ls "$STAGE"/data/*.wav 2>/dev/null | wc -l)
  n_trn=$(ls "$STAGE"/data/*.wav.trn 2>/dev/null | wc -l)
  n_spk=$(ls "$STAGE"/data/*.wav 2>/dev/null | sed 's#.*/##; s#_.*##' | sort -u | wc -l)
  echo "staged: wav=$n_wav trn=$n_trn speakers=$n_spk (want wav=$N_OFFICIAL trn=$N_OFFICIAL)"
  # Independent proof the shared discoverer returns exactly the official pool.
  python3 - "$SCRIPT_DIR/.." "$STAGE" "$N_OFFICIAL" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from asr_gate import corpora
root, want = sys.argv[2], int(sys.argv[3])
utts = corpora.discover_corpus("thchs30", root, "test")
n = len(utts)
with_ref = sum(1 for u in utts if u.ref_text)
prefixes = sorted({u.utt_id[0] for u in utts})
spk = sorted({u.speaker_id for u in utts})
print(f"discover thchs30/test @ {root}: n={n} with_ref={with_ref} prefixes={prefixes} n_speakers={len(spk)}")
ok = (n == want) and (with_ref == want) and (prefixes == ['D'])
print("POOL_DISCOVER_OK" if ok else "POOL_DISCOVER_FAIL"); sys.exit(0 if ok else 1)
PYEOF
  local rc=$?
  if [ "$n_wav" -eq "$N_OFFICIAL" ] && [ "$n_trn" -eq "$N_OFFICIAL" ] && [ "$rc" -eq 0 ]; then
    mark POOL_OFFICIAL OK; return 0
  fi
  mark POOL_OFFICIAL FAILED; return 1
}

# ---------------------------------------------------------------------------
# Decode dispatch (mirrors landscape decode_backbone; official STAGE data-root).
# ---------------------------------------------------------------------------
decode_backbone() {
  local backbone="$1" out="$2" limit_flag="$3"
  local rf=""; [ "$RESUME" -eq 1 ] && rf="--resume"
  case "$backbone" in
    paraformer)   # NB: decode_paraformer.py has --skip-existing but NO --resume.
      python3 "$SCRIPT_DIR/decode_paraformer.py" --corpus thchs30 --split test \
        --data-root "$STAGE" --device "$DEVICE" --skip-existing $limit_flag --out "$out" ;;
    belle)
      python3 "$SCRIPT_DIR/decode_whisper.py" --corpus thchs30 --split test \
        --data-root "$STAGE" --model-name "$BELLE_MODEL" --language zh \
        --device "$DEVICE" --skip-existing $rf $limit_flag --out "$out" ;;
    zipformer)
      python3 "$SCRIPT_DIR/decode_sherpa_onnx.py" --corpus thchs30 --split test \
        --data-root "$STAGE" --model-dir "$ZIPFORMER_DIR" --skip-existing $rf $limit_flag --out "$out" ;;
    *) echo "decode_backbone: unknown backbone $backbone" >&2; return 1 ;;
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
  local ok=1 bb
  for bb in $BACKBONES; do
    local dec="$OUT/smoke/smoke_thchs30_${bb}.jsonl"
    local can="$OUT/smoke/smoke_thchs30_${bb}_canonical.jsonl"
    local sc="$OUT/smoke/smoke_thchs30_${bb}_scored.jsonl"
    rm -f "$dec" "$can" "$sc"
    echo "-- smoke decode: $bb --"
    if decode_backbone "$bb" "$dec" "--limit $SMOKE_N" \
       && assert_decode_content "$dec" "$SMOKE_N" \
       && score_decode "$dec" "$can" "$sc" \
       && assert_scored_cer_finite "$sc"; then
      mark "SMOKE_${bb}" OK
    else
      mark "SMOKE_${bb}" FAILED; ok=1  # mark() already records the failure; keep going
    fi
  done
  [ "${#FAILED_MARKERS[@]}" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Full pipeline per backbone: decode 2495 -> score -> apply (frozen gates,
# 20 reseeds x alpha grid) -> audit. paraformer first (fastest), then belle, zip.
# ---------------------------------------------------------------------------
stage_pipeline() {
  step "Full official-test pipeline (decode $N_OFFICIAL -> score -> apply -> audit)"
  local bb
  for bb in $BACKBONES; do
    echo "########## BACKBONE $bb ##########"
    local bb_dir="$OUT/backbone_${bb}"; mkdir -p "$bb_dir"
    local dec="$OUT/decode_thchs30_official_${bb}.jsonl"
    local can="$bb_dir/thchs30_official_canonical.jsonl"
    local scored="$bb_dir/thchs30_official_test_scored.jsonl"

    # (1) decode
    if [ -s "$dec" ] && assert_decode_content "$dec" "$N_OFFICIAL" >/dev/null 2>&1; then
      skip "$dec"
    else
      decode_backbone "$bb" "$dec" ""
    fi
    if assert_decode_content "$dec" "$N_OFFICIAL"; then mark "DECODE_${bb}" OK
    else mark "DECODE_${bb}" FAILED; echo "skip score/apply/audit for $bb (decode gate failed)"; continue; fi

    # (2) score
    if score_decode "$dec" "$can" "$scored" && assert_scored_cer_finite "$scored" "$N_OFFICIAL"; then
      mark "SCORE_${bb}" OK
    else
      mark "SCORE_${bb}" FAILED; echo "skip apply/audit for $bb (score gate failed)"; continue
    fi

    # (3a) apply frozen aishell-dev gates (read-only) across 20 reseeds x alpha grid
    local reseed label gate applied n_applied=0 n_want=0
    for reseed in $(seq 0 $((N_RESEEDS - 1))); do
      local rdir="$bb_dir/reseed_${reseed}"; mkdir -p "$rdir"
      for label in $ALPHA_LABELS; do
        n_want=$((n_want + 1))
        gate="$LAND/backbone_${bb}/reseed_${reseed}/gate_alpha${label}.json"
        applied="$rdir/applied_thchs30_official_alpha${label}.json"
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

    # (3b) audit official scored (excess-AURC + perm-p + Holm), same flags as landscape
    local audit_json="$OUT/audit/audit_thchs30_official_${bb}.json"
    if [ -s "$audit_json" ]; then
      skip "$audit_json"
    else
      asr-gate audit --instances "$scored" --n-perm "$AUDIT_N_PERM" --alpha "$AUDIT_ALPHA" \
        --seed "$SEED" --out "$audit_json" || echo "WARN audit exited non-zero ($bb)" >&2
    fi
    if python3 - "$audit_json" "$N_OFFICIAL" <<'PYEOF'
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
# Digest: reproduce the manuscript's tab:landscape THCHS derivation
# (compute_numbers.accepted_set_stats/summarise) so the returned cells are
# directly readable -- full-set macro-CER, and per-alpha 20-reseed mean
# acceptance / accepted-set macro-CER / #violations, and the tightest-cert alpha.
# ---------------------------------------------------------------------------
stage_digest() {
  step "Digest (per-backbone: full-set CER + per-alpha attainment + tightest cert alpha)"
  python3 - "$OUT" "$N_RESEEDS" "$ALPHA_LABELS" "$DIGEST_JSON" "$BACKBONES" <<'PYEOF'
import glob, json, os, sys
import numpy as np
OUT, N_RESEEDS = sys.argv[1], int(sys.argv[2])
LABELS = sys.argv[3].split()
DIGEST = sys.argv[4]
BACKBONES = sys.argv[5].split()

def load(p):
    with open(p) as f: return json.load(f)
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def cer_map(scored):
    return {r["utt_id"]: r["cer"] for r in load_jsonl(scored)}

def accepted_set_stats(applied, cmap, alpha):
    a = load(applied); dec = a["decisions"]
    n_total = a["n"]; n_accept = a["n_accept"]
    acc = [d["utt_id"] for d in dec if d["action"] == "ACCEPT"]
    cers = [cmap[u] for u in acc if u in cmap]
    frac = len(acc)/n_total if n_total else 0.0
    macro = float(np.mean(cers)) if cers else None
    return {"acc_fraction": frac, "acc_set_macro_cer": macro,
            "violation": bool(cers and macro > alpha), "vacuous": len(acc) == 0}

out = {"n_official": None, "backbones": {}}
for bb in BACKBONES:
    bb_dir = os.path.join(OUT, f"backbone_{bb}")
    scored = os.path.join(bb_dir, "thchs30_official_test_scored.jsonl")
    audit = os.path.join(OUT, "audit", f"audit_thchs30_official_{bb}.json")
    cell = {"full_set_macro_cer": None, "full_set_micro_cer": None, "n": None, "per_alpha": {}}
    if os.path.exists(audit):
        d = load(audit); cell["full_set_macro_cer"] = d.get("macro_cer")
        mc = d.get("micro_cer"); cell["full_set_micro_cer"] = mc.get("point") if isinstance(mc, dict) else mc
        cell["n"] = d.get("n"); out["n_official"] = d.get("n")
    if os.path.exists(scored):
        cmap = cer_map(scored)
        tightest = None
        for label in LABELS:
            alpha = float(label)
            per = []
            for r in range(N_RESEEDS):
                ap = os.path.join(bb_dir, f"reseed_{r}", f"applied_thchs30_official_alpha{label}.json")
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
print("=== THCHS OFFICIAL (n={}) landscape-cell digest ===".format(out["n_official"]))
print(f"{'backbone':12} {'full-CER':>8} {'tight-a':>8} {'accept':>8} {'acc-CER':>8}")
for bb in BACKBONES:
    c = out["backbones"][bb]
    fc = c["full_set_macro_cer"]; ta = c.get("tightest_cert_alpha")
    ac = c.get("cert_accept_mean"); acc = c.get("cert_acc_cer_mean")
    print("{:12} {:>8} {:>8} {:>8} {:>8}".format(
        bb,
        f"{fc*100:.2f}%" if fc is not None else "-",
        (f"{float(ta)*100:g}%" if ta else "VAC."),
        f"{ac*100:.1f}%" if ac is not None else "-",
        f"{acc*100:.2f}%" if acc is not None else "-"))
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
    { echo "PARTIAL $(date -u +%FT%TZ)"; printf '%s\n' "${FAILED_MARKERS[@]}"; } > "$OUT/THCHS_OFFICIAL_PARTIAL"
    echo "THCHS_OFFICIAL_PARTIAL -- failed: ${FAILED_MARKERS[*]}"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
stage_prologue
stage_build_pool || { echo "POOL BUILD FAILED -- aborting" >&2; stage_epilogue; exit 1; }

if [ "$SMOKE_ONLY" -eq 1 ]; then
  stage_smoke
  echo "smoke-only done ($(date -u +%FT%TZ)); failed markers: ${FAILED_MARKERS[*]:-none}"
  exit $([ "${#FAILED_MARKERS[@]}" -eq 0 ] && echo 0 || echo 1)
fi

stage_pipeline
stage_digest
stage_epilogue
