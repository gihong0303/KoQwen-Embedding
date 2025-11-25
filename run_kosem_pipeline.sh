#!/bin/bash
# ============================================================================
# KOSEM Pipeline Runner
# Korean Universal Representation Enhancement
# ============================================================================

set -e

# Configuration
CONFIG="configs/kosem_config.yaml"
NUM_GPUS=8
MASTER_PORT=29500

# GPU Selection (use GPUs 1-8, skip GPU 0)
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================================"
    echo "  $1"
    echo "============================================================================"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found!"
        exit 1
    fi
    log_success "Python found: $(python --version)"

    # Check PyTorch
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
        log_error "PyTorch not installed!"
        exit 1
    }

    # Check transformers
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
        log_error "Transformers not installed!"
        exit 1
    }

    # Check GPUs
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    log_info "Available GPUs: $GPU_COUNT"

    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        log_warning "Fewer GPUs available than configured. Using $GPU_COUNT GPUs."
        NUM_GPUS=$GPU_COUNT
    fi
}

# Preparation step
prepare() {
    print_header "Stage 0: Preparation"

    log_info "Generating bilingual dictionary..."
    python scripts/stage0_clsa/bilingual_dictionary.py \
        --output outputs/bilingual_dictionary.json \
        --max_tokens 70000 \
        || log_warning "Bilingual dictionary generation skipped (may already exist)"

    log_info "Computing morpheme analysis cache..."
    python -c "
from utils.morpheme_curriculum import MorphemeAnalyzer
analyzer = MorphemeAnalyzer(use_mecab=True, cache_path='outputs/morpheme_analysis/cache.json')
print('Morpheme analyzer initialized')
" || log_warning "Morpheme analysis skipped"

    log_success "Preparation complete!"
}

# Run Stage 0 (CLSA) - uses separate script
run_stage0() {
    print_header "Running Stage 0 - CLSA"

    log_info "Starting Stage 0 (CLSA) training..."
    log_info "Config: $CONFIG"
    log_info "GPUs: $NUM_GPUS"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        scripts/stage0_clsa.py \
        --config $CONFIG

    log_success "Stage 0 completed!"
}

# Run a single stage (stages 1-7)
run_stage() {
    local stage=$1
    local prev_stage=$2

    print_header "Running $stage"

    # Stage 0 uses separate script
    if [ "$stage" = "stage0" ]; then
        run_stage0
        return
    fi

    # Determine model path
    if [ -z "$prev_stage" ]; then
        MODEL_PATH=""
    else
        MODEL_PATH="checkpoints/kosem_${prev_stage}/final"
        if [ ! -d "$MODEL_PATH" ]; then
            log_error "Previous stage checkpoint not found: $MODEL_PATH"
            exit 1
        fi
    fi

    log_info "Starting $stage training..."
    log_info "Config: $CONFIG"
    log_info "Model path: ${MODEL_PATH:-'(base model)'}"
    log_info "GPUs: $NUM_GPUS"

    # Run training
    if [ -n "$MODEL_PATH" ]; then
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            scripts/kosem_trainer.py \
            --config $CONFIG \
            --stage $stage \
            --model_path $MODEL_PATH
    else
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            scripts/kosem_trainer.py \
            --config $CONFIG \
            --stage $stage
    fi

    log_success "$stage completed!"
}

# Main pipeline
run_pipeline() {
    print_header "KOSEM Pipeline"

    log_info "Starting 8-stage KOSEM training pipeline..."
    log_info "Estimated time: 20-24 hours"

    START_TIME=$(date +%s)

    # Stage 0: CLSA
    run_stage "stage0" ""

    # Stage 1: Easy tokens
    run_stage "stage1" "stage0"

    # Stage 2: Medium tokens
    run_stage "stage2" "stage1"

    # Stage 3: Hard tokens
    run_stage "stage3" "stage2"

    # Stage 4: Full vocab harmonization
    run_stage "stage4" "stage3"

    # Stage 5: LoRA transformer
    run_stage "stage5" "stage4"

    # Stage 6: Advanced contrastive
    run_stage "stage6" "stage5"

    # Stage 7: Final refinement
    run_stage "stage7" "stage6"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    print_header "Pipeline Complete!"
    log_success "Total training time: ${HOURS}h ${MINUTES}m"
    log_success "Final model: checkpoints/kosem_stage7/final"
}

# Parse arguments
case "${1:-}" in
    "prepare")
        check_prerequisites
        prepare
        ;;
    "stage0"|"stage1"|"stage2"|"stage3"|"stage4"|"stage5"|"stage6"|"stage7")
        check_prerequisites
        prev=""
        case "$1" in
            "stage1") prev="stage0" ;;
            "stage2") prev="stage1" ;;
            "stage3") prev="stage2" ;;
            "stage4") prev="stage3" ;;
            "stage5") prev="stage4" ;;
            "stage6") prev="stage5" ;;
            "stage7") prev="stage6" ;;
        esac
        run_stage "$1" "$prev"
        ;;
    "all"|"")
        check_prerequisites
        prepare
        run_pipeline
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  prepare    - Run preparation only (bilingual dict, morpheme cache)"
        echo "  stage0     - Run Stage 0 (CLSA) only"
        echo "  stage1     - Run Stage 1 (Easy tokens) only"
        echo "  stage2     - Run Stage 2 (Medium tokens) only"
        echo "  stage3     - Run Stage 3 (Hard tokens) only"
        echo "  stage4     - Run Stage 4 (Full vocab) only"
        echo "  stage5     - Run Stage 5 (LoRA) only"
        echo "  stage6     - Run Stage 6 (Advanced) only"
        echo "  stage7     - Run Stage 7 (Final) only"
        echo "  all        - Run full pipeline (default)"
        echo "  help       - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
