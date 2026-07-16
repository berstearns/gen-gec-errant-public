#!/bin/bash
set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────
CONFIGS_DIR="configs/pipeline/per-model"
OUTPUT_ROOT="outputs"
DEVICE=""
BATCH_SIZE=""
SEED=""
MAX_SENTENCES=""

# ── Arg parsing ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs-dir)   CONFIGS_DIR="$2";   shift 2 ;;
    --output-root)   OUTPUT_ROOT="$2";   shift 2 ;;
    --device)        DEVICE="$2";        shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
    --seed)          SEED="$2";          shift 2 ;;
    --max-sentences) MAX_SENTENCES="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--configs-dir DIR] [--output-root DIR]"
      echo "          [--device auto|cpu|cuda] [--batch-size N]"
      echo "          [--seed N] [--max-sentences N]"
      echo ""
      echo "Runs pipeline for each YAML in configs-dir."
      echo "CLI overrides (--device, etc.) override values in the YAML."
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Resolve paths ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
ENFORCER_DIR="${SCRIPT_DIR}/deploy/enforcers"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
PASSED=0; FAILED=0; SKIPPED=0

CONFIGS=("${CONFIGS_DIR}"/*.yaml)
if [[ ${#CONFIGS[@]} -eq 0 ]] || [[ ! -f "${CONFIGS[0]}" ]]; then
  echo "No YAML configs found in ${CONFIGS_DIR}"; exit 1
fi

echo "=== run_pipeline.sh ==="
echo "  Configs:  ${CONFIGS_DIR} (${#CONFIGS[@]} files)"
echo "  Output:   ${OUTPUT_ROOT}"
echo "  Start:    $(date)"
echo ""

# ── Enforcer helper ──────────────────────────────────────────
run_enforcers() {
  local phase="$1"; shift
  if [[ ! -f "$ENFORCER_DIR/run_all_enforcers.sh" ]]; then
    echo "[WARN] Enforcers not found at $ENFORCER_DIR — skipping $phase checks"
    return 0
  fi
  echo ""
  echo "── Enforcers: $phase ──"
  export PROJECT_DIR="$SCRIPT_DIR"
  export CONFIGS_DIR
  if bash "$ENFORCER_DIR/run_all_enforcers.sh" "$@" 2>&1; then
    echo "[ENFORCERS] $phase: all passed"
  else
    echo "[ENFORCERS] $phase: some checks failed (continuing)"
  fi
  echo ""
}

# ── Pre-flight enforcers ─────────────────────────────────────
run_enforcers "pre-flight" --pre-flight

# ── Helper: write per-model YAML config (for ad-hoc use) ─────
write_config() {
  local name="$1" hf_id="$2" family="$3" outdir="$4" dest="$5"
  local data="${6:-data/splits/norm-CELVA-SP.csv}"
  local max_sent="${7:-10}" bs="${8:-2}" dev="${9:-auto}" sd="${10:-42}"
  local gec="${11:-grammarly/coedit-large}"
  cat > "$dest" <<YAML
data_loader:
  data_path: ${data}
  text_column: text
  max_sentences: ${max_sent}
  min_words: 10
  max_words: 500
  prompt_ratio: 0.5
  min_prompt_words: 5

generation:
  max_new_tokens: 50
  min_new_tokens: 10
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  do_sample: true
  repetition_penalty: 1.2

gec:
  method: dedicated
  model_id: ${gec}
  batch_size: 4
  device: auto

annotation:
  lang: en

analysis:
  skip_plots: false
  top_n_error_types: 10

models:
  - name: ${name}
    hf_model_id: ${hf_id}
    model_family: ${family}

batch_size: ${bs}
device: ${dev}
seed: ${sd}
output_dir: ${outdir}
skip_plots: false
YAML
}

# ── Per-config loop ──────────────────────────────────────────
for config_file in "${CONFIGS[@]}"; do
  NAME=$(basename "$config_file" .yaml)

  # Check for local model weights (parse hf_model_id from YAML)
  HF_ID=$(grep 'hf_model_id:' "$config_file" | head -1 | sed 's/.*hf_model_id: *//')
  if [[ "$HF_ID" == /* ]] && ! ls "$HF_ID"/*.safetensors "$HF_ID"/model*.bin &>/dev/null; then
    echo "[SKIP] $NAME (no weights in $HF_ID)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Build CLI overrides
  OVERRIDES=""
  [[ -n "$OUTPUT_ROOT" ]] && OVERRIDES="$OVERRIDES output_dir=${OUTPUT_ROOT}/${NAME}-${TIMESTAMP}"
  [[ -n "$DEVICE" ]]      && OVERRIDES="$OVERRIDES device=${DEVICE}"
  [[ -n "$BATCH_SIZE" ]]   && OVERRIDES="$OVERRIDES batch_size=${BATCH_SIZE}"
  [[ -n "$SEED" ]]         && OVERRIDES="$OVERRIDES seed=${SEED}"
  [[ -n "$MAX_SENTENCES" ]] && OVERRIDES="$OVERRIDES data_loader.max_sentences=${MAX_SENTENCES}"

  MODEL_OUTPUT="${OUTPUT_ROOT}/${NAME}-${TIMESTAMP}"

  echo ""
  echo "================================================================"
  echo "[RUN] $NAME  ($(date '+%H:%M:%S'))"
  echo "  config: $config_file"
  echo "  output: $MODEL_OUTPUT"
  [[ -n "$OVERRIDES" ]] && echo "  overrides:$OVERRIDES"
  echo "================================================================"

  # Pre-experiment enforcers
  export MODEL_NAME="$NAME"
  export OUTPUT_DIR="$MODEL_OUTPUT"
  run_enforcers "pre-experiment ($NAME)" --pre-experiment

  if $PYTHON -m gen_gec_errant.pipeline --config "$config_file" $OVERRIDES; then
    PASSED=$((PASSED + 1))
    echo "[DONE] $NAME  ($(date '+%H:%M:%S'))"
  else
    FAILED=$((FAILED + 1))
    echo "[FAIL] $NAME  ($(date '+%H:%M:%S'))"
  fi

  # Post-experiment enforcers
  export OUTPUT_DIR="$MODEL_OUTPUT"
  run_enforcers "post-experiment ($NAME)" --post-experiment
done

echo ""
echo "=== SUMMARY ==="
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  End:     $(date)"
