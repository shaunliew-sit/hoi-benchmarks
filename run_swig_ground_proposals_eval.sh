#!/bin/bash
################################################################################
# SWIG-HOI Grounding Evaluation with Object Proposals (Qwen3VL baseline)
# Ablation: injects pre-computed object proposals into the prompt without tool use.
# Uses the baseline Qwen3VL model (no SFT checkpoint, no zoom_in/zoom_out).
#
# Task: Detect person-object pairs with bounding boxes + proposals hint
# Metrics: COCO-style AR (Average Recall)
#
# Usage:
#   bash run_swig_ground_proposals_eval.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   bash run_swig_ground_proposals_eval.sh 0
#   bash run_swig_ground_proposals_eval.sh 0 "Qwen/Qwen3-VL-8B-Instruct"
#   VLLM_URL=http://vllm-server:8000  VERBOSE=1 bash run_swig_ground_proposals_eval.sh 0
#   VLLM_URL=http://vllm-server:8000  MAX_IMAGES=10 VERBOSE=1 bash run_swig_ground_proposals_eval.sh 0
#   VLLM_URL=http://vllm-server:8000  WANDB=1 VERBOSE=1 bash run_swig_ground_proposals_eval.sh 0
#   VLLM_URL=http://vllm-server:8000  RESUME=1 bash run_swig_ground_proposals_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1         Show per-image results
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   VLLM_URL          URL of a running vLLM server (required)
#   RESUME=1          Resume from the latest partial checkpoint in OUTPUT_DIR
#   IMAGE_ID=<file>   Run only samples matching this image filename
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: swig-grounding-proposals)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#   PROPOSALS_DIR     Override proposals directory
#
# Output files:
#   {output_dir}/swig_ground_proposals_results_{timestamp}.json
#   {output_dir}/swig_ground_proposals_results_{timestamp}_metrics.json
#   {output_dir}/swig_ground_proposals_results_{timestamp}_action_stats.json
#   {output_dir}/swig_ground_proposals_results_{timestamp}_thinking.jsonl
#   {output_dir}/swig_ground_proposals_evaluation_{timestamp}.log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_NAME="${2:-Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${3:-results-proposals/swig_ground_proposals}"

# Set GPU (handle both "0" and "cuda:0" formats)
if [[ "$GPU_ID" == cuda:* ]]; then
    DEVICE_ARG="$GPU_ID"
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    DEVICE_ARG="cuda:$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/swig_ground_proposals_evaluation_${TIMESTAMP}.log"

# SWIG dataset paths
SWIG_ROOT="/workspace/hoi/data/swig_hoi"
IMG_PREFIX="${SWIG_ROOT}/images_512"
ANN_FILE="/workspace/hoi/data/benchmarks_simplified/swig_ground_test_simplified.json"
RESULT_FILE="${OUTPUT_DIR}/swig_ground_proposals_results_${TIMESTAMP}.json"

# Proposals directory
PROPOSALS_DIR="${PROPOSALS_DIR:-/workspace/hoi/checkpoints/test_proposals}"

if [ -z "$VLLM_URL" ]; then
    echo "ERROR: VLLM_URL is required (for example: http://localhost:8000)"
    exit 1
fi

if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not reachable at $VLLM_URL"
    echo "Start vLLM manually, then rerun this script."
    exit 1
fi

# GPU availability check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null) || true
    if [ -n "$GPU_INFO" ]; then
        printf '%s\n' "$GPU_INFO" | nl -v 0
    else
        echo "  GPU info unavailable in this container"
    fi
    echo ""
fi

echo "========================================================================"
echo "SWIG-HOI Grounding Evaluation (Qwen3VL + Proposals)"
echo "========================================================================"
echo "GPU:           $GPU_ID (Device: $DEVICE_ARG)"
echo "Model:         $MODEL_NAME"
echo "Annotation:    $ANN_FILE"
echo "Images:        $IMG_PREFIX"
echo "Proposals dir: $PROPOSALS_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Log file:      $LOG_FILE"
echo "Result file:   $RESULT_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found at $ANN_FILE"
    echo "Please ensure the benchmark file has been generated"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found at $IMG_PREFIX"
    echo "Please check the path to SWIG images_512"
    exit 1
fi

# Warn if proposals directory is missing (non-fatal — inference proceeds without proposals)
if [ ! -d "$PROPOSALS_DIR" ]; then
    echo "WARNING: Proposals directory not found at $PROPOSALS_DIR"
    echo "  Evaluation will proceed without proposals (equivalent to baseline)"
fi

# Count number of test images
NUM_IMAGES=$(ls -1 "$IMG_PREFIX"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

echo "Starting evaluation..."
echo ""

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
RESUME_FLAG=""
IMAGE_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-image results)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES images"
fi

if [ ! -z "$IMAGE_ID" ]; then
    IMAGE_FLAG="--image $IMAGE_ID"
    echo "✓ Image filter: $IMAGE_ID"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: swig-grounding-proposals (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

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

# Build evaluation command
EVAL_CMD="python3 eval_swig_ground_proposals_qwen3vl.py \
    --model-name \"$MODEL_NAME\" \
    --device $DEVICE_ARG \
    --vllm-url \"$VLLM_URL\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE \
    --proposals-dir $PROPOSALS_DIR"

if [ ! -z "$VERBOSE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
fi

if [ ! -z "$MAX_IMAGES_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
fi

if [ ! -z "$WANDB_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $WANDB_FLAG"
fi

if [ ! -z "$RESUME_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $RESUME_FLAG"
fi

if [ ! -z "$IMAGE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $IMAGE_FLAG"
fi

# Execute the command
eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Results saved to:"
    echo "  Predictions:    $RESULT_FILE"
    echo "  Metrics:        ${RESULT_FILE//.json/_metrics.json}"
    echo "  Action stats:   ${RESULT_FILE//.json/_action_stats.json}"
    THINKING_FILE="${RESULT_FILE//.json/_thinking.jsonl}"
    if [ -f "$THINKING_FILE" ]; then
        echo "  Thinking:       $THINKING_FILE  (thinking model output)"
    fi
    echo "  Log:            $LOG_FILE"
    echo ""

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
    echo ""
    echo "Note: This dataset includes person-person interactions"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Evaluation failed!"
    echo "========================================================================"
    echo "Check the log file for details:"
    echo "  $LOG_FILE"
    echo "========================================================================"
    exit 1
fi
