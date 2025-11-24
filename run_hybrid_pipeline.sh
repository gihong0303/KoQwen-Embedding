#!/bin/bash
# ============================================================================
# Hybrid Pipeline Runner: CLSA + JLCE + MCL
# Runs all 7 stages with Korean-specific innovations
# ============================================================================

set -e

echo "============================================================================"
echo "  Hybrid Pipeline: CLSA + JLCE + MCL"
echo "  7-Stage Korean Embedding Enhancement"
echo "============================================================================"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES="4,5,6,7,8,9"
export TOKENIZERS_PARALLELISM="false"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET

NUM_GPUS=6
CONFIG="configs/pipeline_config.yaml"

echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config: $CONFIG"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# =============================================================================
# Preparation Phase
# =============================================================================
echo "============================================================================"
echo "PREPARATION PHASE"
echo "============================================================================"
echo ""

# Check for bilingual dictionary
if [ ! -f "outputs/bilingual_dictionary.json" ]; then
    echo "Generating bilingual dictionary..."
    python scripts/stage0_clsa/bilingual_dictionary.py \
        --model_path "Qwen/Qwen3-Embedding-0.6B" \
        --tokenizer_path "outputs/koqwen-expanded" \
        --output_path "outputs/bilingual_dictionary.json" \
        --max_tokens 70000
fi

# Check for morphological curriculum
if [ ! -f "outputs/morphological_curriculum/token_categories.json" ]; then
    echo "Generating morphological curriculum..."
    mkdir -p outputs/morphological_curriculum
    python -c "
import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from utils.morphological_curriculum import create_morphological_curriculum_config
tokenizer = AutoTokenizer.from_pretrained('outputs/koqwen-expanded', trust_remote_code=True)
create_morphological_curriculum_config(tokenizer, 'outputs/morphological_curriculum', use_mecab=False)
print('Morphological curriculum created successfully!')
"
fi

echo "Preparation complete!"
echo ""

# =============================================================================
# Stage 0: Cross-lingual Semantic Anchoring (CLSA)
# =============================================================================
echo "============================================================================"
echo "STAGE 0: Cross-lingual Semantic Anchoring (CLSA)"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage0/final" ]; then
    echo "Running Stage 0..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        scripts/stage0_clsa.py \
        --config $CONFIG
    echo "Stage 0 complete!"
else
    echo "Stage 0 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 1-3: JLCE + MCL (Korean-specific learning)
# =============================================================================
for STAGE in 1 2 3; do
    echo "============================================================================"
    case $STAGE in
        1) DESC="Easy Tokens (Stems)" ;;
        2) DESC="Medium Tokens (Stem+Affix)" ;;
        3) DESC="Hard Tokens (Complex Forms)" ;;
    esac
    echo "STAGE $STAGE: $DESC (JLCE + MCL)"
    echo "============================================================================"
    echo ""

    CHECKPOINT_DIR="checkpoints/stage${STAGE}_jlce_mcl/final"
    PREV_CHECKPOINT="checkpoints/stage$((STAGE-1))/final"
    if [ $STAGE -eq 1 ]; then
        PREV_CHECKPOINT="checkpoints/stage0/final"
    else
        PREV_CHECKPOINT="checkpoints/stage$((STAGE-1))_jlce_mcl/final"
    fi

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Running Stage $STAGE..."
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$((29500 + STAGE)) \
            scripts/jlce_mcl_trainer.py \
            --config $CONFIG \
            --stage $STAGE \
            --model_path $PREV_CHECKPOINT
        echo "Stage $STAGE complete!"
    else
        echo "Stage $STAGE checkpoint found. Skipping."
    fi

    echo ""
done

# =============================================================================
# Stage 4: Full Vocabulary Harmonization
# =============================================================================
echo "============================================================================"
echo "STAGE 4: Full Vocabulary Harmonization"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage4/final" ]; then
    echo "Running Stage 4..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29504 \
        scripts/stage4.py \
        --config $CONFIG \
        --model_path checkpoints/stage3_jlce_mcl/final
    echo "Stage 4 complete!"
else
    echo "Stage 4 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 5: LoRA Transformer Enhancement
# =============================================================================
echo "============================================================================"
echo "STAGE 5: LoRA Transformer Enhancement"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage5/final" ]; then
    echo "Running Stage 5..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29505 \
        scripts/stage5.py \
        --config $CONFIG \
        --model_path checkpoints/stage4/final
    echo "Stage 5 complete!"
else
    echo "Stage 5 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 6: Advanced Contrastive Learning
# =============================================================================
echo "============================================================================"
echo "STAGE 6: Advanced Contrastive Learning"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage6/final" ]; then
    echo "Running Stage 6..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29506 \
        scripts/stage6.py \
        --config $CONFIG \
        --model_path checkpoints/stage5/final
    echo "Stage 6 complete!"
else
    echo "Stage 6 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Complete!
# =============================================================================
echo "============================================================================"
echo "ALL STAGES COMPLETE!"
echo "============================================================================"
echo ""
echo "Final model location: checkpoints/stage6/final"
echo ""
echo "Pipeline Summary:"
echo "  Stage 0: CLSA (Cross-lingual Semantic Anchoring)"
echo "  Stage 1: JLCE+MCL Easy Tokens (Stems)"
echo "  Stage 2: JLCE+MCL Medium Tokens (Stem+Affix)"
echo "  Stage 3: JLCE+MCL Hard Tokens (Complex)"
echo "  Stage 4: Full Vocabulary Harmonization"
echo "  Stage 5: LoRA Transformer Enhancement"
echo "  Stage 6: Advanced Contrastive Learning"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: python scripts/comprehensive_evaluation.py"
echo "  2. Compare with baseline"
echo ""
