#!/bin/bash
################################################################################
# HICO-DET Grounding Evaluation — Qwen3VL Tool-Use Agent (training-prompt aligned).
# Uses the VERBATIM SFTv2/GRPO training prompt -> works for BOTH the SFT and the
# GRPO checkpoint (point CHECKPOINT_PATH / VLLM_URL at whichever model).
#
# Args:  bash run_hico_ground_sftgrpo_eval.sh [GPU_ID] [VLLM_URL] [CHECKPOINT_PATH]
#
# ---- HOW TO RUN ----------------------------------------------------------
# 1) SMOKE TEST FIRST (verbose, 1 sample) — confirm the model emits
#    <think>/<tool_call>/<answer> and the prompt renders correctly:
#       MAX_IMAGES=1 VERBOSE=1 CHECKPOINT_PATH=/path/to/checkpoint \
#           bash run_hico_ground_sftgrpo_eval.sh 0
#
# 2) FULL RUN:
#       CHECKPOINT_PATH=/path/to/checkpoint bash run_hico_ground_sftgrpo_eval.sh 0
#
# 3) Against an already-running vLLM server (skip auto-start):
#       bash run_hico_ground_sftgrpo_eval.sh 0 http://localhost:8000
#
# 4) Verbose on a subset / single image / resume:
#       MAX_IMAGES=20 VERBOSE=1 bash run_hico_ground_sftgrpo_eval.sh 0
#       IMAGE_ID=HICO_test2015_00000001.jpg VERBOSE=1 bash run_hico_ground_sftgrpo_eval.sh 0
#       RESUME=1 WANDB=1 VERBOSE=1 bash run_hico_ground_sftgrpo_eval.sh 0
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
################################################################################

set -e

GPU_ID="${1:-0}"
VLLM_URL="${2:-http://vllm-qwen-inference:8000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/raid/scratch/shaun_sit/hoi/AdaTooler-V/verltool/checkpoints/SAHA-CF/SAHA-CF-a0.6-newsft-resampled/global_step_1000/actor/huggingface}"
# GPU server (RTX 6000 Ada): CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/d/Work/latest_checkpoints/sft-checkpoints/qwen3VL-4B}"
# OUTPUT_DIR="${OUTPUT_DIR:-results-sftgrpo/hico_ground_grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-results-sftgrpo/hico_ground_grpo_4b_step1000}"
MAX_TURNS="${MAX_TURNS:-6}"

if [[ "$GPU_ID" == cuda:* ]]; then
    DEVICE_ARG="$GPU_ID"
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    DEVICE_ARG="cuda:$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_sft_evaluation_${TIMESTAMP}.log"
RESULT_FILE="$OUTPUT_DIR/hico_ground_sft_results_${TIMESTAMP}.json"

ANN_FILE="${ANN_FILE:-/workspace/Groma/groma_data/benchmarks_simplified/hico_ground_test_simplified.json}"
IMG_PREFIX="${IMG_PREFIX:-/workspace/data/hico_20160224_det/images/test2015}"
PROPOSALS_DIR="${PROPOSALS_DIR:-/workspace/hoi-tool-use-checkpoints/test_proposals}"
# GPU server (RTX 6000 Ada):
# ANN_FILE="/mnt/d/Work/data/benchmarks_simplified/hico_ground_test_simplified.json"
# IMG_PREFIX="/mnt/d/Work/data/hico_20160224_det/images/test2015"
# PROPOSALS_DIR="/mnt/d/Work/data/test_proposals"

echo "========================================================================"
echo "HICO-DET Grounding Evaluation (SFT/GRPO training-prompt — Qwen3VL)"
echo "========================================================================"
echo "GPU:           $GPU_ID"
echo "vLLM URL:      $VLLM_URL"
echo "Annotation:    $ANN_FILE"
echo "Images:        $IMG_PREFIX"
echo "Proposals:     $PROPOSALS_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Max turns:     $MAX_TURNS"
echo "========================================================================"
echo ""

# Check if vLLM server is running
if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "vLLM server not running at $VLLM_URL"
    if [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
        echo "Starting vLLM server with checkpoint: $CHECKPOINT_PATH"
        echo "NOTE: Server startup takes ~30-60 seconds."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
            --model "$CHECKPOINT_PATH" \
            --port 8000 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"
        echo "Waiting for server to start..."
        sleep 60
        if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            echo "ERROR: vLLM server failed to start. Check logs."
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "✓ vLLM server started"
    else
        echo "ERROR: No checkpoint path set and server not running."
        echo "Please start vLLM server manually:"
        echo "  vllm serve $CHECKPOINT_PATH --port 8000 --max-model-len 8192"
        exit 1
    fi
else
    echo "✓ vLLM server is running at $VLLM_URL"
fi

# Get model name from vLLM
MODEL_NAME=$(curl -s "${VLLM_URL}/v1/models" | python3 -c "import json,sys; models=json.load(sys.stdin)['data']; print(models[0]['id'])" 2>/dev/null || echo "qwen3VL-4B")
echo "Model name: $MODEL_NAME"
echo ""

if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found: $ANN_FILE"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found: $IMG_PREFIX"
    exit 1
fi

if [ ! -d "$PROPOSALS_DIR" ]; then
    echo "WARNING: Proposals directory not found: $PROPOSALS_DIR"
    echo "Run generate_test_proposals.sh first!"
fi

VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
RESUME_FLAG=""
IMAGE_FLAG=""

[ ! -z "$VERBOSE" ] && VERBOSE_FLAG="--verbose" && echo "✓ Verbose mode enabled"
[ ! -z "$MAX_IMAGES" ] && MAX_IMAGES_FLAG="--max-images $MAX_IMAGES" && echo "✓ Limiting to $MAX_IMAGES images"
[ ! -z "$WANDB" ] && WANDB_FLAG="--wandb" && echo "✓ WandB logging enabled"
[ ! -z "$IMAGE_ID" ] && IMAGE_FLAG="--image $IMAGE_ID" && echo "✓ Image filter: $IMAGE_ID"

# Resume: find the most recent partial checkpoint and reuse its output file
if [ ! -z "$RESUME" ]; then
    PARTIAL_FILE=$(ls -t "$OUTPUT_DIR"/*.json.partial.jsonl 2>/dev/null | head -1)
    if [ -n "$PARTIAL_FILE" ]; then
        RESULT_FILE="${PARTIAL_FILE%.partial.jsonl}"
        LOG_FILE="${RESULT_FILE//_results_/_evaluation_}.log"
        RESUME_FLAG="--resume"
        echo "✓ Resuming from partial checkpoint: $PARTIAL_FILE"
        echo "  Output file: $RESULT_FILE"
    else
        echo "No partial checkpoint found in $OUTPUT_DIR, starting fresh"
    fi
fi

echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
[ ! -x "$PYTHON" ] && PYTHON="python3"

EVAL_CMD="$PYTHON eval_hico_ground_sftgrpo_qwen3vl.py \
    --vllm-url $VLLM_URL \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --proposals-dir $PROPOSALS_DIR \
    --result-file $RESULT_FILE \
    --max-turns $MAX_TURNS"

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
    echo "Results: $RESULT_FILE"
    echo "Metrics: ${RESULT_FILE//.json/_metrics.json}"
    echo "Log:     $LOG_FILE"
    echo "========================================================================"
else
    echo "ERROR: Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi
