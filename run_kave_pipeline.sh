#!/bin/bash
# ============================================================================
# KAVE Pipeline: Korean Adaptive Vocabulary Expansion
# ============================================================================
#
# Complete pipeline for vocabulary expansion with retrieval preservation
#
# Usage:
#   ./run_kave_pipeline.sh                    # Run all stages
#   ./run_kave_pipeline.sh --stage 1          # Run specific stage (1-6)
#   ./run_kave_pipeline.sh --eval             # Run evaluation only
#   ./run_kave_pipeline.sh --resume 3         # Resume from stage 3
#
# Stages:
#   1: WSA Initialization (Weighted Semantic Averaging)
#   2: Easy token learning (curriculum phase 1)
#   3: Medium token learning (curriculum phase 2)
#   4: Hard token learning (curriculum phase 3)
#   5: Unified training with PEU (Progressive Embedding Unfreezing)
#   6: Retrieval fine-tuning (RAT - Retrieval-Aware Training)
#
# ============================================================================

set -e

# ============================================================================
# GPU Configuration (GPU 1-8)
# ============================================================================
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
NUM_GPUS=8

# ============================================================================
# NCCL Configuration for Stable DDP
# Reference: PyTorch DDP Best Practices
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1          # Disable InfiniBand (use Ethernet)
export NCCL_P2P_DISABLE=0         # Enable P2P (faster GPU-to-GPU)
export NCCL_SHM_DISABLE=0         # Enable shared memory
export TORCH_NCCL_BLOCKING_WAIT=1 # Blocking wait for debugging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# Config file
CONFIG="configs/kave_pipeline_config.yaml"

# Master port (change if conflict)
MASTER_PORT=29500

# ============================================================================
# Parse Arguments
# ============================================================================

STAGE=""
EVAL_ONLY=false
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
            echo "KAVE Pipeline - Korean Adaptive Vocabulary Expansion"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage N     Run only stage N (1-6)"
            echo "  --eval        Run evaluation only"
            echo "  --prepare     Run preparation only (token difficulty)"
            echo "  --resume N    Resume from stage N"
            echo "  --config F    Use config file F"
            echo "  --help        Show this help"
            echo ""
            echo "Stages:"
            echo "  1: WSA Initialization"
            echo "  2: Easy token learning"
            echo "  3: Medium token learning"
            echo "  4: Hard token learning"
            echo "  5: Unified training + PEU"
            echo "  6: Retrieval fine-tuning"
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
    echo " $1"
    echo "============================================================================"
    echo ""
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_checkpoint() {
    local dir=$1
    if [ -d "$dir/final" ]; then
        return 0  # Exists
    fi
    return 1  # Not exists
}

# ============================================================================
# Stage Functions
# ============================================================================

# Stage 0: Prepare token difficulty for curriculum learning
prepare_curriculum() {
    log_header "PREPARATION: Token Difficulty Calculation"

    if [ -f "outputs/token_difficulty/token_categories.json" ]; then
        log_info "Token categories already exist. Skipping."
        return 0
    fi

    log_info "Calculating token difficulty scores..."

    python scripts/prepare_curriculum.py \
        --tokenizer_path "outputs/koqwen-expanded" \
        --output_dir "outputs/token_difficulty"

    log_info "Token difficulty preparation complete!"
}

# Stage 1: WSA Initialization
run_stage1_init() {
    log_header "STAGE 1: WSA Token Initialization"

    if [ -d "outputs/kave-initialized" ]; then
        log_info "WSA initialization already exists. Skipping."
        return 0
    fi

    log_info "Running Weighted Semantic Averaging initialization..."

    python scripts/kave_stage1_init.py \
        --config "$CONFIG"

    log_info "Stage 1 (WSA Init) complete!"
}

# Stage 2-6: Training stages
run_training_stage() {
    local stage_num=$1
    local checkpoint_dir=$2

    log_header "STAGE $stage_num: KAVE Training"

    if check_checkpoint "$checkpoint_dir"; then
        log_info "Stage $stage_num already completed. Skipping."
        return 0
    fi

    log_info "Starting Stage $stage_num with $NUM_GPUS GPUs..."

    # Run with torchrun for distributed training
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$((MASTER_PORT + stage_num)) \
        scripts/kave_train_stage.py \
        --stage $stage_num \
        --config "$CONFIG"

    log_info "Stage $stage_num complete!"
}

# Evaluation
run_evaluation() {
    local model_path=$1

    log_header "MTEB Korean Retrieval Evaluation"

    python scripts/evaluate_mteb.py \
        --model_path "$model_path" \
        --output_dir "evaluation_results" \
        --batch_size 64 \
        --device "cuda:0"
}

# ============================================================================
# Main Execution
# ============================================================================

log_header "KAVE Pipeline: Korean Adaptive Vocabulary Expansion"
echo "Configuration:"
echo "  - GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Config: $CONFIG"
echo "  - Mode: ${STAGE:-all stages}"
echo ""
echo "KAVE Components:"
echo "  - WSA: Weighted Semantic Averaging (init)"
echo "  - CTA: Contextual Token Alignment (loss)"
echo "  - PEU: Progressive Embedding Unfreezing (gradient)"
echo "  - RAT: Retrieval-Aware Training (auxiliary)"
echo ""

# Evaluate only mode
if [ "$EVAL_ONLY" = true ]; then
    FINAL_MODEL="checkpoints/kave_stage6/final"
    if [ ! -d "$FINAL_MODEL" ]; then
        FINAL_MODEL="checkpoints/kave_stage5/final"
    fi
    if [ ! -d "$FINAL_MODEL" ]; then
        echo "Error: No trained model found"
        exit 1
    fi
    run_evaluation "$FINAL_MODEL"
    exit 0
fi

# Prepare only mode
if [ "$PREPARE_ONLY" = true ]; then
    prepare_curriculum
    exit 0
fi

# Determine starting stage
START_STAGE=1
if [ -n "$RESUME_FROM" ]; then
    START_STAGE=$RESUME_FROM
fi

# Single stage mode
if [ -n "$STAGE" ]; then
    case $STAGE in
        1)
            prepare_curriculum
            run_stage1_init
            ;;
        2|3|4|5|6)
            # Ensure previous stages are done
            if [ "$STAGE" -ge 2 ] && [ ! -d "outputs/kave-initialized" ]; then
                log_info "Running Stage 1 first..."
                prepare_curriculum
                run_stage1_init
            fi

            # Get checkpoint dir from stage number
            case $STAGE in
                2) checkpoint_dir="checkpoints/kave_stage2" ;;
                3) checkpoint_dir="checkpoints/kave_stage3" ;;
                4) checkpoint_dir="checkpoints/kave_stage4" ;;
                5) checkpoint_dir="checkpoints/kave_stage5" ;;
                6) checkpoint_dir="checkpoints/kave_stage6" ;;
            esac

            run_training_stage $STAGE "$checkpoint_dir"
            ;;
        *)
            echo "Invalid stage: $STAGE (valid: 1-6)"
            exit 1
            ;;
    esac
else
    # Run all stages
    log_info "Running complete KAVE pipeline..."

    # Preparation
    if [ "$START_STAGE" -le 1 ]; then
        prepare_curriculum
    fi

    # Stage 1: WSA Init
    if [ "$START_STAGE" -le 1 ]; then
        run_stage1_init
    fi

    # Stage 2: Easy tokens
    if [ "$START_STAGE" -le 2 ]; then
        run_training_stage 2 "checkpoints/kave_stage2"
    fi

    # Stage 3: Medium tokens
    if [ "$START_STAGE" -le 3 ]; then
        run_training_stage 3 "checkpoints/kave_stage3"
    fi

    # Stage 4: Hard tokens
    if [ "$START_STAGE" -le 4 ]; then
        run_training_stage 4 "checkpoints/kave_stage4"
    fi

    # Stage 5: Unified + PEU
    if [ "$START_STAGE" -le 5 ]; then
        run_training_stage 5 "checkpoints/kave_stage5"
    fi

    # Stage 6: Retrieval fine-tuning
    if [ "$START_STAGE" -le 6 ]; then
        run_training_stage 6 "checkpoints/kave_stage6"
    fi
fi

# ============================================================================
# Final Summary
# ============================================================================

log_header "KAVE PIPELINE COMPLETE!"

FINAL_MODEL="checkpoints/kave_stage6/final"
if [ ! -d "$FINAL_MODEL" ]; then
    FINAL_MODEL="checkpoints/kave_stage5/final"
fi

if [ -d "$FINAL_MODEL" ]; then
    echo "Final model: $FINAL_MODEL"
    echo ""
    echo "To run evaluation:"
    echo "  ./run_kave_pipeline.sh --eval"
    echo ""
    echo "To compare with baseline:"
    echo "  python scripts/evaluate_mteb.py \\"
    echo "      --model_path $FINAL_MODEL \\"
    echo "      --compare Qwen/Qwen3-Embedding-0.6B"
else
    echo "Warning: No final model found"
fi

echo ""
echo "KAVE Framework Summary:"
echo "  - WSA: Semantic-weighted token initialization"
echo "  - CTA: MLM + Contrastive + Alignment loss"
echo "  - PEU: Progressive old embedding unfreezing"
echo "  - RAT: Query-document retrieval preservation"
