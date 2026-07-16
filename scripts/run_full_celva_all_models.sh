#!/bin/bash
set -uo pipefail

# Run the full CELVA dataset pipeline sequentially for each model.
# Overrides max_sentences to use all data (configs default to 10).

# ── Defaults ──────────────────────────────────────────────────
CONFIGS_DIR="configs/full-celva"
OUTPUT_ROOT="outputs"
DEVICE="auto"
BATCH_SIZE=""
SEED=""
DRY_RUN=false

# ── Arg parsing ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)   OUTPUT_ROOT="$2";   shift 2 ;;
    --device)        DEVICE="$2";        shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
    --seed)          SEED="$2";          shift 2 ;;
    --dry-run)       DRY_RUN=true;       shift ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Run the full CELVA pipeline for all models sequentially."
      echo ""
      echo "Options:"
      echo "  --output-root DIR    Output root directory (default: outputs)"
      echo "  --device DEVICE      Device: auto|cpu|cuda (default: auto)"
      echo "  --batch-size N       Override batch size"
      echo "  --seed N             Override random seed"
      echo "  --dry-run            Print commands without executing"
      echo "  -h, --help           Show this help"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Resolve paths ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] Python not found at $PYTHON"
  exit 1
fi

# ── Collect configs in explicit order ─────────────────────────
#    Pythia (base → ft, small → large), then SmolLM2, then GPT-2
CONFIG_ORDER=(
  # Pythia base (ascending size)
  full-celva-base-pythia-70m
  full-celva-base-pythia-160m
  full-celva-base-pythia-410m
  full-celva-base-pythia-1b
  full-celva-base-pythia-1.4b
  # Pythia fine-tuned (ascending size)
  full-celva-ft-pythia-70m
  full-celva-ft-pythia-160m
  full-celva-ft-pythia-410m
  full-celva-ft-pythia-1b
  full-celva-ft-pythia-1.4b
  # SmolLM2 base (ascending size)
  full-celva-base-smollm2-135m
  full-celva-base-smollm2-360m
  full-celva-base-smollm2-1.7b
  # SmolLM2 fine-tuned (ascending size)
  full-celva-ft-smollm2-135m
  full-celva-ft-smollm2-360m
  full-celva-ft-smollm2-1.7b
  # GPT-2 base (ascending size)
  full-celva-base-gpt2-small
  full-celva-base-gpt2-medium
  full-celva-base-gpt2-large
  # GPT-2 fine-tuned (ascending size)
  full-celva-ft-gpt2-small
  full-celva-ft-gpt2-medium
  full-celva-ft-gpt2-large
)

CONFIGS=()
for name in "${CONFIG_ORDER[@]}"; do
  cfg="${CONFIGS_DIR}/${name}.yaml"
  if [[ -f "$cfg" ]]; then
    CONFIGS+=("$cfg")
  else
    echo "[WARN] Config not found: $cfg"
  fi
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "[ERROR] No YAML configs resolved"
  exit 1
fi

TOTAL=${#CONFIGS[@]}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
PASSED=0; FAILED=0; SKIPPED=0; CURRENT=0

echo "============================================="
echo "  Full CELVA Pipeline — All Models"
echo "============================================="
echo "  Configs:  ${CONFIGS_DIR} (${TOTAL} models)"
echo "  Output:   ${OUTPUT_ROOT}"
echo "  Device:   ${DEVICE}"
echo "  Start:    $(date)"
echo "============================================="
echo ""

# ── Per-config loop ───────────────────────────────────────────
for config_file in "${CONFIGS[@]}"; do
  NAME=$(basename "$config_file" .yaml)

  CURRENT=$((CURRENT + 1))

  # Check for local model weights
  HF_ID=$(grep 'hf_model_id:' "$config_file" | head -1 | sed 's/.*hf_model_id: *//')
  if [[ "$HF_ID" == /* ]] && ! ls "$HF_ID"/*.safetensors "$HF_ID"/model*.bin &>/dev/null; then
    echo "[SKIP] $NAME — no weights in $HF_ID"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Build CLI overrides
  MODEL_OUTPUT="${OUTPUT_ROOT}/${NAME}"
  OVERRIDES="output_dir=${MODEL_OUTPUT} data_loader.max_sentences=none"
  [[ -n "$DEVICE" ]]     && OVERRIDES="$OVERRIDES device=${DEVICE}"
  [[ -n "$BATCH_SIZE" ]] && OVERRIDES="$OVERRIDES batch_size=${BATCH_SIZE}"
  [[ -n "$SEED" ]]       && OVERRIDES="$OVERRIDES seed=${SEED}"

  echo "================================================================"
  echo "[${CURRENT}/${TOTAL}] $NAME  ($(date '+%H:%M:%S'))"
  echo "  config:    $config_file"
  echo "  output:    $MODEL_OUTPUT"
  echo "  overrides: $OVERRIDES"
  echo "================================================================"

  CMD="$PYTHON -m gen_gec_errant.pipeline --config $config_file $OVERRIDES"

  if $DRY_RUN; then
    echo "[DRY-RUN] $CMD"
    echo ""
    continue
  fi

  START_TIME=$SECONDS
  if $CMD; then
    ELAPSED=$(( SECONDS - START_TIME ))
    PASSED=$((PASSED + 1))
    echo "[DONE] $NAME  (${ELAPSED}s, $(date '+%H:%M:%S'))"
  else
    ELAPSED=$(( SECONDS - START_TIME ))
    FAILED=$((FAILED + 1))
    echo "[FAIL] $NAME  (${ELAPSED}s, $(date '+%H:%M:%S'))"
  fi
  echo ""
done

echo "============================================="
echo "  SUMMARY"
echo "============================================="
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  End:     $(date)"
echo "============================================="
