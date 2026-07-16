#!/bin/bash
# S1 PREP — rclone-sync the exact registry best/ checkpoints for the 5 S1
# matched pairs (gpt2 s/m/l, pythia 410m/1b). Inference files only — excludes
# optimizer/scheduler/rng/training-args state (the bulk of each checkpoint's
# size, irrelevant to from_pretrained loading).
# See gen-gec-review-specs/50-dispatch-log.md dispatch #2.
set -uo pipefail

REMOTE_ROOT="i:_p/artificial-learners/models"
LOCAL_ROOT="PLACEHOLDER_MODELS_ROOT"
LOG="./outputs/s1-checkpoint-sync.log"
mkdir -p "$(dirname "$LOG")"

CKPTS=(
  "gpt2/gpt2-small-all-data/best/checkpoint-7596"
  "gpt2/gpt2-medium-all-data/best/checkpoint-5625"
  "gpt2/gpt2-large-all-data/best/checkpoint-6750"
  "pythia/pythia-410m-all-data/best/checkpoint-21476"
  "pythia/pythia-1b-all-data/best/checkpoint-21036"
)

{
  echo "=== S1 checkpoint sync — $(date -Is) ==="
  df -h PLACEHOLDER_DATA_STORE
  echo ""
} > "$LOG"

RC=0
for ckpt in "${CKPTS[@]}"; do
  echo ">>> Syncing $ckpt" | tee -a "$LOG"
  rclone copy "$REMOTE_ROOT/$ckpt/" "$LOCAL_ROOT/$ckpt/" \
    --include "config.json" \
    --include "generation_config.json" \
    --include "model.safetensors" \
    --include "tokenizer.json" \
    --include "tokenizer_config.json" \
    --progress --stats-one-line --log-file "$LOG" --log-level INFO
  this_rc=$?
  if [ $this_rc -ne 0 ]; then RC=$this_rc; fi
  echo ">>> Done $ckpt (rc=$this_rc)" | tee -a "$LOG"
done

echo "" >> "$LOG"
df -h PLACEHOLDER_DATA_STORE >> "$LOG"
echo "DONE rc=$RC" >> "$LOG"
exit $RC
