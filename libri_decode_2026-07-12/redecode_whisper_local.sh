#!/bin/bash
# Whisper re-decode with the sequences/transition_scores alignment fix
# (2026-07-12): the box-run JSONLs lost the last ~4 generated tokens of
# every utterance. wav2vec2 files unaffected (different code path).
set -uo pipefail
A=${REPO_ROOT}
cd $A/orchestration
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_ENDPOINT=https://hf-mirror.com WHISPER_MAX_NEW_TOKENS=440
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for sub in test-clean test-other; do
  python3 decode_librispeech.py --backend whisper --subset $sub \
    --libri-root $A/libri_staging/LibriSpeech \
    --out $A/libri_decode_2026-07-12/whisper_${sub}_fixed.jsonl \
    || echo "WARN: $sub non-zero"
done
wc -l $A/libri_decode_2026-07-12/whisper_*_fixed.jsonl
echo WHISPER_REDECODE_DONE
