#!/bin/bash
# ============================================================================
# Stage 0: Cross-lingual Semantic Anchoring (CLSA)
# ============================================================================

set -e

echo "=========================================="
echo "Stage 0: Cross-lingual Semantic Anchoring"
echo "=========================================="

# Environment variables
export CUDA_VISIBLE_DEVICES="4,5,6,7,8,9"
export TOKENIZERS_PARALLELISM="false"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET

# Number of GPUs
NUM_GPUS=6

# Config path
CONFIG_PATH="configs/pipeline_config_clsa.yaml"

echo ""
echo "📋 Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config: $CONFIG_PATH"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Run with torchrun (DDP)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/stage0_clsa.py \
    --config $CONFIG_PATH

echo ""
echo "✅ Stage 0 (CLSA) training complete!"
echo ""
