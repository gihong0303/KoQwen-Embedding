#!/usr/bin/env bash
# Stage 2: New Token Alignment

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

MODEL_PATH="checkpoints/stage1/final"
NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 2: New Token Alignment"
echo "================================================================================"
echo ""
echo "학습 대상: embed_tokens (새 토큰만, 계속)"
echo "데이터: KOREAN-WEBTEXT (300K)"
echo "입력: ${MODEL_PATH}"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29532 \
  scripts/stage2.py \
  --config configs/pipeline_config.yaml \
  --model_path "${MODEL_PATH}" \
  --seed 42

echo ""
echo "✅ Stage 2 완료! → checkpoints/stage2/final"
echo ""
