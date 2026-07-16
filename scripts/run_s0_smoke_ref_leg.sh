#!/bin/bash
# S0 smoke run — reference-annotation (G3) leg. See configs/dummies/s0-smoke-ref-leg.yaml
# and gen-gec-review-specs/KICKOFF-executor.md task 1.3.
set -uo pipefail

cd .
source .venv/bin/activate

OUT_DIR=outputs/s0-smoke-ref-leg
mkdir -p "$OUT_DIR"

CMD="python -m gen_gec_errant.pipeline --config configs/dummies/s0-smoke-ref-leg.yaml"
echo "$CMD" > "$OUT_DIR/command.txt"

{
  echo "=== PROVENANCE ==="
  echo "date: $(date -Is)"
  echo "git_sha: $(git rev-parse HEAD)"
  echo "config: configs/dummies/s0-smoke-ref-leg.yaml"
  echo "python: $(python --version 2>&1)"
  echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
  echo "cuda_available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
  echo "device: cpu (no CUDA on this host)"
  echo "gec_model: grammarly/coedit-large"
  echo "generator: ft-gpt2-small"
  echo "generator_checkpoint: PLACEHOLDER_MODELS_ROOT/2026-02-23-model/gpt2-small-all-data-resume/final"
  echo "generator_checkpoint_NOTE: SUBSTITUTED for the registry/remote path .../gpt2/gpt2-small-all-data/best/checkpoint-7596, which does not exist on this local machine (not yet rclone-synced; see config header comment and 90-open-questions.md)"
  echo "seed: 42"
  echo "include_learner_baseline: true (G3 reference-annotation leg)"
  echo "=================="
} > "$OUT_DIR/run.log"

$CMD >> "$OUT_DIR/run.log" 2>&1
RC=$?
echo "DONE rc=$RC" >> "$OUT_DIR/run.log"
exit $RC
