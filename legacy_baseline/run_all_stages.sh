#!/usr/bin/env bash
# 6-Stage Pipeline - Complete Execution

set -euo pipefail

echo "================================================================================"
echo "🚀 Korean Embedding Expansion - 6-Stage Pipeline"
echo "================================================================================"
echo ""
echo "목표: Qwen3-Embedding-0.6B 한국어 토큰 확장 (논문 기반 6단계)"
echo ""
echo "파이프라인:"
echo "  Stage 1: New Token Input Embeddings"
echo "  Stage 2: New Token Alignment"
echo "  Stage 3: New Token Refinement"
echo "  Stage 4: Full Vocabulary Harmonization"
echo "  Stage 5: Transformer Enhancement (LoRA)"
echo "  Stage 6: Advanced Contrastive Learning"
echo ""
echo "================================================================================"
echo ""

./run_stage1.sh
./run_stage2.sh
./run_stage3.sh
./run_stage4.sh
./run_stage5.sh
./run_stage6.sh

echo ""
echo "================================================================================"
echo "🎉 전체 6단계 파이프라인 완료!"
echo "================================================================================"
echo ""
echo "최종 모델: checkpoints/stage6/final"
echo ""
echo "================================================================================"
