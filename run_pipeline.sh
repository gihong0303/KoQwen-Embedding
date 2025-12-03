#!/bin/bash
# ============================================================================
# Korean Embedding Training Pipeline
# 6-Stage Pipeline with Proper DDP Configuration
# ============================================================================
#
# Usage:
#   ./run_pipeline.sh              # Run all stages (1-6)
#   ./run_pipeline.sh --stage 1    # Run specific stage
#   ./run_pipeline.sh --stage 6    # Run Stage 6 (Supervised Retrieval)
#   ./run_pipeline.sh --eval       # Run evaluation only
#   ./run_pipeline.sh --eval-parallel  # Run parallel evaluation (multi-GPU)
#   ./run_pipeline.sh --resume 3   # Resume from stage 3
#
# Stage 6: Supervised Retrieval Contrastive Learning
#   - Uses MIRACL Korean (query-document pairs)
#   - Uses KorNLI entailment pairs
#   - Optimizes retrieval performance directly
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# GPU Configuration (1-8)
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
NUM_GPUS=8

# NCCL Configuration for stable DDP
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Tokenizer
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# Config file
CONFIG="configs/pipeline_config.yaml"

# Master port (change if port conflict)
MASTER_PORT=29500

# ============================================================================
# Parse Arguments
# ============================================================================

STAGE=""
EVAL_ONLY=false
EVAL_PARALLEL=false
PREPARE_ONLY=false
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --eval)
            EVAL_ONLY=true
            shift
            ;;
        --eval-parallel)
            EVAL_PARALLEL=true
            shift
            ;;
        --prepare)
            PREPARE_ONLY=true
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage N       Run only stage N (1-6)"
            echo "  --eval          Run sequential evaluation (single GPU)"
            echo "  --eval-parallel Run parallel evaluation (multi-GPU)"
            echo "  --prepare       Run preparation only (token difficulty)"
            echo "  --resume N      Resume from stage N"
            echo "  --config F      Use config file F"
            echo "  --help          Show this help"
            echo ""
            echo "Stages:"
            echo "  1: Easy tokens (curriculum learning)"
            echo "  2: Medium tokens (curriculum learning)"
            echo "  3: Hard tokens (curriculum learning)"
            echo "  4: Full vocabulary harmonization"
            echo "  5: LoRA fine-tuning (NLI)"
            echo "  6: Supervised retrieval (MIRACL + KorNLI)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

log_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_stage() {
    local stage_num=$1
    local script=$2
    local model_path=$3
    local checkpoint_dir=$4

    log_header "STAGE $stage_num: Running $script"

    # Check if already completed
    if [ -d "$checkpoint_dir/final" ]; then
        log_info "Stage $stage_num already completed. Skipping."
        return 0
    fi

    # Run with torchrun
    log_info "Starting Stage $stage_num with $NUM_GPUS GPUs..."

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$((MASTER_PORT + stage_num)) \
        $script \
        --config $CONFIG \
        ${model_path:+--model_path "$model_path"}

    log_info "Stage $stage_num completed!"

    # Wait for GPU memory to be released
    log_info "Waiting for GPU cleanup..."
    sleep 10

    # Force GPU memory cleanup
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 5
}

run_evaluation() {
    local model_path=$1
    local parallel=$2

    log_header "MTEB Korean Retrieval Evaluation"

    if [ "$parallel" = "true" ]; then
        log_info "Running parallel evaluation (multi-GPU)..."
        python scripts/evaluate_mteb_parallel.py \
            --model_path "$model_path" \
            --output_dir "evaluation_results" \
            --parallel
    else
        log_info "Running sequential evaluation (single GPU)..."
        python scripts/evaluate_mteb_parallel.py \
            --model_path "$model_path" \
            --output_dir "evaluation_results" \
            --device "cuda:0"
    fi
}

run_comparison() {
    local model_path=$1
    local baseline=$2

    log_header "Model Comparison"

    python scripts/evaluate_mteb_parallel.py \
        --model_path "$model_path" \
        --output_dir "evaluation_results" \
        --compare "$baseline"
}

# ============================================================================
# Main Execution
# ============================================================================

log_header "Korean Embedding Training Pipeline"
echo "Configuration:"
echo "  - GPUs: $NUM_GPUS ($CUDA_VISIBLE_DEVICES)"
echo "  - Config: $CONFIG"
echo "  - Mode: ${STAGE:-all stages}"
echo ""

# ============================================================================
# Preparation: Token Difficulty for Curriculum Learning
# ============================================================================

prepare_curriculum() {
    log_header "PREPARATION: Token Difficulty Calculation"

    if [ -f "outputs/token_difficulty/token_categories.json" ]; then
        log_info "Token categories already exist. Skipping preparation."
        return 0
    fi

    log_info "Calculating token difficulty scores..."

    python scripts/prepare_curriculum.py \
        --tokenizer_path "outputs/koqwen-expanded" \
        --output_dir "outputs/token_difficulty"

    log_info "Token difficulty preparation complete!"
}

# Prepare only mode
if [ "$PREPARE_ONLY" = true ]; then
    prepare_curriculum
    exit 0
fi

# Evaluation only mode
if [ "$EVAL_ONLY" = true ] || [ "$EVAL_PARALLEL" = true ]; then
    # Find the best available model (stage6 > stage5 > stage4)
    FINAL_MODEL="checkpoints/stage6/final"
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="checkpoints/stage5/final"
    fi
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="checkpoints/stage4/final"
    fi
    if [ ! -d "$FINAL_MODEL" ]; then
        echo "Error: No trained model found"
        exit 1
    fi

    if [ "$EVAL_PARALLEL" = true ]; then
        run_evaluation "$FINAL_MODEL" "true"
    else
        run_evaluation "$FINAL_MODEL" "false"
    fi
    exit 0
fi

# Determine starting stage
START_STAGE=1
if [ -n "$RESUME_FROM" ]; then
    START_STAGE=$RESUME_FROM
fi

# Run preparation if needed (only for stages 1-3 which use curriculum)
if [ -z "$STAGE" ] || [ "$STAGE" -le 3 ]; then
    prepare_curriculum
fi

# Run specific stage or all stages
if [ -n "$STAGE" ]; then
    # Single stage mode
    case $STAGE in
        1)
            run_stage 1 "scripts/stage1_curriculum.py" "" "checkpoints/stage1"
            ;;
        2)
            run_stage 2 "scripts/stage2_curriculum.py" "checkpoints/stage1/final" "checkpoints/stage2"
            ;;
        3)
            run_stage 3 "scripts/stage3_curriculum.py" "checkpoints/stage2/final" "checkpoints/stage3"
            ;;
        4)
            run_stage 4 "scripts/stage4.py" "checkpoints/stage3/final" "checkpoints/stage4"
            ;;
        5)
            run_stage 5 "scripts/stage5.py" "checkpoints/stage4/final" "checkpoints/stage5"
            ;;
        6)
            run_stage 6 "scripts/stage6_retrieval.py" "checkpoints/stage5/final" "checkpoints/stage6"
            ;;
        *)
            echo "Invalid stage: $STAGE (valid: 1-6)"
            exit 1
            ;;
    esac
else
    # Run all stages (1-6)
    for ((i=START_STAGE; i<=6; i++)); do
        case $i in
            1)
                run_stage 1 "scripts/stage1_curriculum.py" "" "checkpoints/stage1"
                ;;
            2)
                run_stage 2 "scripts/stage2_curriculum.py" "checkpoints/stage1/final" "checkpoints/stage2"
                ;;
            3)
                run_stage 3 "scripts/stage3_curriculum.py" "checkpoints/stage2/final" "checkpoints/stage3"
                ;;
            4)
                run_stage 4 "scripts/stage4.py" "checkpoints/stage3/final" "checkpoints/stage4"
                ;;
            5)
                run_stage 5 "scripts/stage5.py" "checkpoints/stage4/final" "checkpoints/stage5"
                ;;
            6)
                run_stage 6 "scripts/stage6_retrieval.py" "checkpoints/stage5/final" "checkpoints/stage6"
                ;;
        esac
    done
fi

# Final evaluation
log_header "PIPELINE COMPLETE!"

# Find the best available model
FINAL_MODEL="checkpoints/stage6/final"
if [ ! -d "$FINAL_MODEL" ]; then
    FINAL_MODEL="checkpoints/stage5/final"
fi
if [ ! -d "$FINAL_MODEL" ]; then
    FINAL_MODEL="checkpoints/stage4/final"
fi

if [ -d "$FINAL_MODEL" ]; then
    echo "Final model: $FINAL_MODEL"
    echo ""
    echo "To run evaluation:"
    echo "  ./run_pipeline.sh --eval          # Sequential (single GPU)"
    echo "  ./run_pipeline.sh --eval-parallel # Parallel (multi-GPU, faster)"
    echo ""
    echo "Or compare with baseline:"
    echo "  python scripts/evaluate_mteb_parallel.py \\"
    echo "      --model_path $FINAL_MODEL \\"
    echo "      --compare Qwen/Qwen3-Embedding-0.6B"
    echo ""
    echo "Stage 6 uses supervised retrieval contrastive learning:"
    echo "  - MIRACL Korean (query-document pairs)"
    echo "  - KorNLI entailment pairs"
    echo "  - Optimizes all 6 MTEB Korean retrieval tasks"
else
    echo "Warning: No final model found"
fi
