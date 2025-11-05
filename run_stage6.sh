#!/usr/bin/env bash
# Stage 6: Advanced Contrastive Learning

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

MODEL_PATH="checkpoints/stage5/final"
NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 6: Advanced Contrastive Learning"
echo "================================================================================"
echo ""
echo "학습 대상: Transformer layers (LoRA r=32)"
echo "데이터: K2-Feedback (150K, score=5)"
echo "입력: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "배치: 10 × 6 × 8 = 480 (effective batch size)"
echo ""
echo "⚠️  고품질 피드백 데이터로 최종 대조 학습!"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29536 \
  scripts/stage6.py \
  --config configs/pipeline_config.yaml \
  --model_path "${MODEL_PATH}" \
  --seed 42

echo ""
echo "✅ Stage 6 완료! → checkpoints/stage6/final"
echo ""
echo "🎉 전체 6단계 파이프라인 완료!"
echo ""
