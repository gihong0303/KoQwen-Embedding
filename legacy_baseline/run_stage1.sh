#!/usr/bin/env bash
# Stage 1: New Token Input Embeddings

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

NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 1: New Token Input Embeddings (최적화 - 8 GPUs)"
echo "================================================================================"
echo ""
echo "학습 대상: embed_tokens (새 토큰 67,762개만)"
echo "데이터: KOREAN-WEBTEXT (300K)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "배치: 20 × 3 × 8 = 480 (effective batch size)"
echo "예상 시간: ~2-2.5시간 (기존 4시간 → 50% 단축)"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29531 \
  scripts/stage1.py \
  --config configs/pipeline_config.yaml \
  --seed 42

echo ""
echo "✅ Stage 1 완료! → checkpoints/stage1/final"
echo ""
