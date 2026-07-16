#!/bin/bash
# S1 exploratory pilot — gpt2-small pair. NOT confirmatory — see the header
# in configs/s1-pilot/*.yaml and gen-gec-review-specs/50-dispatch-log.md
# dispatch #4a/#5. Runs each named config sequentially.
#
# Usage: run_s1_pilot.sh [config-name ...]   (default: gpt2-small ft-gpt2-small)
# e.g.:  run_s1_pilot.sh ft-gpt2-small       (rerun just one, skip the rest)
set -uo pipefail

cd .
source .venv/bin/activate

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("gpt2-small" "ft-gpt2-small")
fi

for name in "${CONFIGS[@]}"; do
  OUT_DIR="outputs/s1-pilot/${name}"
  mkdir -p "$OUT_DIR"
  CONFIG="configs/s1-pilot/${name}.yaml"
  CMD="python -m gen_gec_errant.pipeline --config ${CONFIG}"
  MAX_SENTENCES=$(grep -m1 'max_sentences:' "$CONFIG" | awk '{print $2}')

  echo "$CMD" > "$OUT_DIR/command.txt"
  {
    echo "=== PROVENANCE ($name) ==="
    echo "date: $(date -Is)"
    echo "git_sha: $(git rev-parse HEAD)"
    echo "config: $CONFIG"
    echo "python: $(python --version 2>&1)"
    echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "cuda_available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
    echo "gec_model: grammarly/coedit-large"
    echo "max_sentences: ${MAX_SENTENCES} (EXPLORATORY, NOT CONFIRMATORY)"
    echo "seed: 42"
    echo "=================="
  } > "$OUT_DIR/run.log"

  echo ">>> [$(date -Is)] Starting $name"
  $CMD >> "$OUT_DIR/run.log" 2>&1
  RC=$?
  echo "DONE rc=$RC" >> "$OUT_DIR/run.log"
  echo ">>> [$(date -Is)] Finished $name rc=$RC"

  if [ $RC -ne 0 ]; then
    echo ">>> $name FAILED — stopping pilot (see $OUT_DIR/run.log)"
    exit $RC
  fi
done

echo ">>> S1 pilot complete — all requested configs DONE rc=0 (${CONFIGS[*]})"
