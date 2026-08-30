#!/bin/bash
################################################################################
# HICO-DET Action Referring Evaluation — Qwen3VL Tool-Use Agent (training-prompt aligned).
# Uses the VERBATIM SFTv2/GRPO training prompt -> works for BOTH the SFT and the
# GRPO checkpoint (point CHECKPOINT_PATH / VLLM_URL at whichever model).
#
# Args:  bash run_hico_action_sftgrpo_eval.sh [GPU_ID] [VLLM_URL] [CHECKPOINT_PATH]
#
# ---- HOW TO RUN ----------------------------------------------------------
# 1) SMOKE TEST FIRST (verbose, 1 sample) — confirm the model emits
#    <think>/<tool_call>/<answer> and the prompt renders correctly:
#       MAX_IMAGES=1 VERBOSE=1 CHECKPOINT_PATH=/path/to/checkpoint \
#           bash run_hico_action_sftgrpo_eval.sh 0
#
# 2) FULL RUN:
#       CHECKPOINT_PATH=/path/to/checkpoint bash run_hico_action_sftgrpo_eval.sh 0
#
# 3) Against an already-running vLLM server (skip auto-start):
#       bash run_hico_action_sftgrpo_eval.sh 0 http://localhost:8000
#
# 4) Verbose on a subset / single image / resume:
#       MAX_IMAGES=20 VERBOSE=1 bash run_hico_action_sftgrpo_eval.sh 0
#       IMAGE_ID=HICO_test2015_00000001.jpg VERBOSE=1 bash run_hico_action_sftgrpo_eval.sh 0
#       RESUME=1 WANDB=1 VERBOSE=1 bash run_hico_action_sftgrpo_eval.sh 0
#
# ---- ENV VARS ------------------------------------------------------------
#   VERBOSE=1         Show per-sample results (prompt, tool calls, answer) — recommended
#   MAX_IMAGES=N      Limit to first N samples (use for smoke tests)
#   CHECKPOINT_PATH   Path to checkpoint (SFT or GRPO) for auto-starting vLLM
#   VLLM_URL          Use an existing vLLM server instead of auto-start
#   MAX_TURNS=N       Max tool-call turns (default: 5, matches training)
#   WANDB=1           Enable Weights & Biases logging
#   RESUME=1          Resume from the most recent partial checkpoint in OUTPUT_DIR
#   IMAGE_ID=<file>   Run only samples matching this image filename
#   OUTPUT_DIR=<dir>  Override output dir (default: results-sftgrpo/...)
#
# ---- ZOOM ABLATION (reviewer request) ------------------------------------
#   ZOOM_MODE=<mode>  adaptive (default, stock SAHA) | never | always | random
#                       never  : every zoom_in is denied -> no-tool lower bound
#                       always : model must zoom >=1x, it still picks the region
#                                -> ablates *when* to zoom (the gate)
#                       random : model must zoom >=1x, but the region is replaced
#                                by a random crop -> ablates *where* to zoom
#                     Non-adaptive modes write to $OUTPUT_DIR/zoom_<mode>/ so the
#                     four conditions never clobber each other.
#   ZOOM_SEED=N       RNG seed for ZOOM_MODE=random (default: 0). Re-run with 2-3
#                     seeds and report mean +/- std.
#   ZOOM_EXTRA="..."  Extra flags passed through, e.g.
#                     ZOOM_EXTRA="--random-zoom-max-iou 0.0"   (force a disjoint region)
#                     ZOOM_EXTRA="--random-zoom-free-size"     (randomise box size too)
#                     ZOOM_EXTRA="--never-zoom-prompt-hint"    (also disable tools in prompt)
#
#   e.g.  ZOOM_MODE=never  bash <this script> 0
#         ZOOM_MODE=random ZOOM_SEED=1 bash <this script> 0
################################################################################

set -eo pipefail

GPU_ID="${1:-0}"
VLLM_URL="${2:-http://vllm-qwen-inference:8000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/raid/scratch/shaun_sit/hoi/AdaTooler-V/verltool/checkpoints/SAHA-CF/SAHA-CF-a0.6-newsft-resampled/global_step_1000/actor/huggingface}"
# GPU server (RTX 6000 Ada): CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/d/Work/latest_checkpoints/sft-checkpoints/qwen3VL-4B}"
# OUTPUT_DIR="${OUTPUT_DIR:-results-sftgrpo/hico_action_grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-results-sftgrpo/hico_action_grpo_4b_step1000}"
MAX_TURNS="${MAX_TURNS:-6}"

# ---- Zoom-policy ablation --------------------------------------------------
ZOOM_MODE="${ZOOM_MODE:-adaptive}"
ZOOM_SEED="${ZOOM_SEED:-0}"
case "$ZOOM_MODE" in
    adaptive|never|always|random) ;;
    *) echo "ERROR: ZOOM_MODE must be adaptive|never|always|random (got '$ZOOM_MODE')"; exit 1 ;;
esac
ZOOM_FLAGS="--zoom-mode $ZOOM_MODE --zoom-seed $ZOOM_SEED"
[ -n "$ZOOM_EXTRA" ] && ZOOM_FLAGS="$ZOOM_FLAGS $ZOOM_EXTRA"
if [ "$ZOOM_MODE" != "adaptive" ]; then
    # keep each ablation condition in its own dir (results + resume partials)
    OUTPUT_DIR="${OUTPUT_DIR%/}/zoom_${ZOOM_MODE}"
    [ "$ZOOM_MODE" = "random" ] && OUTPUT_DIR="${OUTPUT_DIR}_seed${ZOOM_SEED}"
fi

if [[ "$GPU_ID" == cuda:* ]]; then
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_sft_evaluation_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/hico_action_sft_results_${TIMESTAMP}.json"

ANN_FILE="/workspace/Groma/groma_data/benchmarks_simplified/hico_action_referring_test_simplified.json"
IMG_PREFIX="/workspace/data/hico_20160224_det/images/test2015"
PROPOSALS_DIR="/workspace/hoi-tool-use-checkpoints/test_proposals"
# GPU server (RTX 6000 Ada):
# ANN_FILE="/mnt/d/Work/data/benchmarks_simplified/hico_action_referring_test_simplified.json"
# IMG_PREFIX="/mnt/d/Work/data/hico_20160224_det/images/test2015"
# PROPOSALS_DIR="/mnt/d/Work/data/test_proposals"

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (SFT/GRPO training-prompt — Qwen3VL)"
echo "========================================================================"
echo "GPU:        $GPU_ID"
echo "vLLM URL:   $VLLM_URL"
echo "Output:     $OUTPUT_DIR"
echo "Zoom mode:  $ZOOM_MODE${ZOOM_EXTRA:+ ($ZOOM_EXTRA)}"
echo "========================================================================"
echo ""

if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "vLLM server not running at $VLLM_URL"
    if [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
        echo "Starting vLLM server..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
            --model "$CHECKPOINT_PATH" \
            --port 8000 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code &
        VLLM_PID=$!
        sleep 60
        if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            echo "ERROR: vLLM server failed to start."
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "✓ vLLM server started"
    else
        echo "ERROR: Set CHECKPOINT_PATH or start vLLM manually."
        exit 1
    fi
else
    echo "✓ vLLM server running"
fi

MODEL_NAME=$(curl -s "${VLLM_URL}/v1/models" | python3 -c "import json,sys; models=json.load(sys.stdin)['data']; print(models[0]['id'])" 2>/dev/null || echo "qwen3VL-4B")
echo "Model: $MODEL_NAME"
echo ""

if [ ! -d "$PROPOSALS_DIR" ]; then
    echo "WARNING: Proposals directory not found: $PROPOSALS_DIR"
    echo "Run generate_test_proposals.sh first!"
fi

VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
RESUME_FLAG=""
IMAGE_FLAG=""

[ ! -z "$VERBOSE" ] && VERBOSE_FLAG="--verbose"
[ ! -z "$MAX_IMAGES" ] && MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
[ ! -z "$WANDB" ] && WANDB_FLAG="--wandb"
[ ! -z "$IMAGE_ID" ] && IMAGE_FLAG="--image $IMAGE_ID"

# Resume: find the most recent partial checkpoint and reuse its output file
if [ ! -z "$RESUME" ]; then
    PARTIAL_FILE=$(ls -t "$OUTPUT_DIR"/*.json.partial.jsonl 2>/dev/null | head -1)
    if [ -n "$PARTIAL_FILE" ]; then
        PRED_FILE="${PARTIAL_FILE%.partial.jsonl}"
        LOG_FILE="${PRED_FILE//_results_/_evaluation_}.log"
        RESUME_FLAG="--resume"
        echo "Resuming from partial checkpoint: $PARTIAL_FILE"
        echo "Output file: $PRED_FILE"
    else
        echo "No partial checkpoint found in $OUTPUT_DIR, starting fresh"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[ ! -x "$PYTHON" ] && PYTHON="python3"

EVAL_CMD="$PYTHON eval_hico_action_referring_sftgrpo_qwen3vl.py \
    --vllm-url $VLLM_URL \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --proposals-dir $PROPOSALS_DIR \
    --pred-file $PRED_FILE \
    --max-turns $MAX_TURNS \
    $ZOOM_FLAGS"

[ ! -z "$VERBOSE_FLAG" ] && EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
[ ! -z "$MAX_IMAGES_FLAG" ] && EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
[ ! -z "$WANDB_FLAG" ] && EVAL_CMD="$EVAL_CMD $WANDB_FLAG"
[ ! -z "$RESUME_FLAG" ] && EVAL_CMD="$EVAL_CMD $RESUME_FLAG"
[ ! -z "$IMAGE_FLAG" ] && EVAL_CMD="$EVAL_CMD $IMAGE_FLAG"

eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Predictions: $PRED_FILE"
    echo "Metrics:     ${PRED_FILE//.json/_metrics.json}"
    echo "Log:         $LOG_FILE"
    echo ""
    echo "BERTScore (run separately):"
    echo "  python calculate_bertscore.py --pred-file $PRED_FILE --model roberta-large"
    echo "========================================================================"
else
    echo "ERROR: Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi
