#!/usr/bin/env bash
# Stage 3: New Token Refinement

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET

MODEL_PATH="checkpoints/stage2/final"
NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 3: New Token Refinement"
echo "================================================================================"
echo ""
echo "학습 대상: embed_tokens (새 토큰만)"
echo "데이터: KOREAN-SyntheticText (200K)"
echo "입력: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "배치: 12 × 5 × 8 = 480 (effective batch size)"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29533 \
  scripts/stage3.py \
  --config configs/pipeline_config.yaml \
  --model_path "${MODEL_PATH}" \
  --seed 42

echo ""
echo "✅ Stage 3 완료! → checkpoints/stage3/final"
echo ""
