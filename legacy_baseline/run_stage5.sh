#!/usr/bin/env bash
# Stage 5: Transformer Enhancement with LoRA

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

MODEL_PATH="checkpoints/stage4/final"
NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 5: Transformer Enhancement with LoRA"
echo "================================================================================"
echo ""
echo "학습 대상: Transformer layers (LoRA r=64)"
echo "데이터: Mixed Dataset (200K)"
echo "  - HAE-RAE-COT: 100K"
echo "  - HR-Instruct-Math: 100K"
echo "입력: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "배치: 8 × 8 × 8 = 512 (effective batch size)"
echo ""
echo "⚠️  주의: LoRA로 Transformer 레이어 학습!"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29535 \
  scripts/stage5.py \
  --config configs/pipeline_config.yaml \
  --model_path "${MODEL_PATH}" \
  --seed 42

echo ""
echo "✅ Stage 5 완료! → checkpoints/stage5/final"
echo ""
