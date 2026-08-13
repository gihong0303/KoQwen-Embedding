#!/usr/bin/env bash
# Stage 4: Full Vocabulary Harmonization

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET

MODEL_PATH="checkpoints/stage3/final"
NUM_GPUS=8

echo "================================================================================"
echo "🎯 Stage 4: Full Vocabulary Harmonization"
echo "================================================================================"
echo ""
echo "학습 대상: embed_tokens (전체 토큰!)"
echo "데이터: Mixed Dataset (200K)"
echo "  - KOREAN-WEBTEXT: 100K"
echo "  - KOREAN-SyntheticText: 80K"
echo "  - KoSimpleEval: 20K"
echo "입력: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "배치: 10 × 6 × 8 = 480 (effective batch size)"
echo ""
echo "⚠️  주의: 이 단계부터 기존 토큰도 학습합니다!"
echo ""
echo "================================================================================"
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=29534 \
  scripts/stage4.py \
  --config configs/pipeline_config.yaml \
  --model_path "${MODEL_PATH}" \
  --seed 42

echo ""
echo "✅ Stage 4 완료! → checkpoints/stage4/final"
echo ""
