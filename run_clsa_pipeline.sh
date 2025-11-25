#!/bin/bash
# ============================================================================
# Complete CLSA Pipeline Runner
# Runs all 7 stages with CLSA + Token Curriculum
# ============================================================================

set -e

echo "============================================================================"
echo "  CLSA + Token Curriculum Pipeline"
echo "  7-Stage Enhanced Training for Korean Embedding"
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
CONFIG="configs/pipeline_config_clsa.yaml"

echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config: $CONFIG"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# =============================================================================
# Preparation: Bilingual Dictionary + Token Difficulty
# =============================================================================
echo "============================================================================"
echo "PREPARATION PHASE"
echo "============================================================================"
echo ""

if [ ! -f "outputs/bilingual_dictionary.json" ] || [ ! -f "outputs/token_difficulty/token_categories.json" ]; then
    echo "⚠️  Missing preparation files. Running prepare_clsa.sh..."
    bash prepare_clsa.sh
else
    echo "✅ Preparation files found. Skipping preparation."
fi

echo ""

# =============================================================================
# Stage 0: Cross-lingual Semantic Anchoring
# =============================================================================
echo "============================================================================"
echo "STAGE 0: Cross-lingual Semantic Anchoring (CLSA)"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage0/final" ]; then
    echo "▶️  Running Stage 0..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        scripts/stage0_clsa.py \
        --config $CONFIG
    echo "✅ Stage 0 complete!"
else
    echo "✅ Stage 0 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 1: Easy Tokens (Curriculum)
# =============================================================================
echo "============================================================================"
echo "STAGE 1: Easy Tokens (Curriculum Learning)"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage1_curriculum/final" ]; then
    echo "▶️  Running Stage 1..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29501 \
        scripts/stage1_curriculum.py \
        --config $CONFIG \
        --model_path checkpoints/stage0/final
    echo "✅ Stage 1 complete!"
else
    echo "✅ Stage 1 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 2: Medium Tokens (Curriculum)
# =============================================================================
echo "============================================================================"
echo "STAGE 2: Medium Tokens (Curriculum Learning)"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage2_curriculum/final" ]; then
    echo "▶️  Running Stage 2..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29502 \
        scripts/stage2_curriculum.py \
        --config $CONFIG \
        --model_path checkpoints/stage1_curriculum/final
    echo "✅ Stage 2 complete!"
else
    echo "✅ Stage 2 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 3: Hard Tokens (Curriculum)
# =============================================================================
echo "============================================================================"
echo "STAGE 3: Hard Tokens (Curriculum Learning)"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage3_curriculum/final" ]; then
    echo "▶️  Running Stage 3..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29503 \
        scripts/stage3_curriculum.py \
        --config $CONFIG \
        --model_path checkpoints/stage2_curriculum/final
    echo "✅ Stage 3 complete!"
else
    echo "✅ Stage 3 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Stage 4: Full Vocabulary Harmonization
# =============================================================================
echo "============================================================================"
echo "STAGE 4: Full Vocabulary Harmonization"
echo "============================================================================"
echo ""

if [ ! -d "checkpoints/stage4/final" ]; then
    echo "▶️  Running Stage 4..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29504 \
        scripts/stage4.py \
        --config $CONFIG \
        --model_path checkpoints/stage3_curriculum/final
    echo "✅ Stage 4 complete!"
else
    echo "✅ Stage 4 checkpoint found. Skipping."
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
    echo "▶️  Running Stage 5..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29505 \
        scripts/stage5.py \
        --config $CONFIG \
        --model_path checkpoints/stage4/final
    echo "✅ Stage 5 complete!"
else
    echo "✅ Stage 5 checkpoint found. Skipping."
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
    echo "▶️  Running Stage 6..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29506 \
        scripts/stage6.py \
        --config $CONFIG \
        --model_path checkpoints/stage5/final
    echo "✅ Stage 6 complete!"
else
    echo "✅ Stage 6 checkpoint found. Skipping."
fi

echo ""

# =============================================================================
# Complete!
# =============================================================================
echo "============================================================================"
echo "✅ ALL STAGES COMPLETE!"
echo "============================================================================"
echo ""
echo "Final model location: checkpoints/stage6/final"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: python scripts/comprehensive_evaluation.py"
echo "  2. Compare with baseline"
echo ""
echo "Training summary:"
echo "  Stage 0: Cross-lingual Semantic Anchoring"
echo "  Stage 1: Easy Tokens (30% easiest)"
echo "  Stage 2: Medium Tokens (40% medium)"
echo "  Stage 3: Hard Tokens (30% hardest)"
echo "  Stage 4: Full Vocabulary Harmonization"
echo "  Stage 5: LoRA Transformer Enhancement"
echo "  Stage 6: Advanced Contrastive Learning"
echo ""
