#!/bin/bash
# ============================================================================
# Preparation Script for CLSA Pipeline
# 1. Extract Bilingual Dictionary (Korean-English-Chinese)
# 2. Compute Token Difficulty Scores
# ============================================================================

set -e

echo "============================================"
echo "CLSA Pipeline Preparation"
echo "============================================"
echo ""

# =============================================================================
# Step 1: Extract Bilingual Dictionary
# =============================================================================
echo "Step 1: Extracting Bilingual Dictionary..."
echo "-------------------------------------------"

python scripts/stage0_clsa/bilingual_dictionary.py \
    --base_model "Qwen/Qwen3-Embedding-0.6B" \
    --korean_tokenizer "outputs/koqwen-expanded" \
    --vocab_diff_path "tokenizer/vocab_diff.json" \
    --output "outputs/bilingual_dictionary.json" \
    --top_k 5 \
    --min_similarity 0.3 \
    --max_tokens 70000

echo ""
echo "✅ Bilingual dictionary extracted!"
echo ""

# =============================================================================
# Step 2: Compute Token Difficulty Scores
# =============================================================================
echo "Step 2: Computing Token Difficulty Scores..."
echo "---------------------------------------------"

python scripts/stage0_clsa/token_difficulty.py \
    --base_tokenizer "Qwen/Qwen3-Embedding-0.6B" \
    --korean_tokenizer "outputs/koqwen-expanded" \
    --vocab_diff_path "tokenizer/vocab_diff.json" \
    --dataset "HAERAE-HUB/KOREAN-WEBTEXT" \
    --max_samples 100000 \
    --output_dir "outputs/token_difficulty"

echo ""
echo "✅ Token difficulty scores computed!"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "============================================"
echo "✅ CLSA Preparation Complete!"
echo "============================================"
echo ""
echo "Generated files:"
echo "  📁 outputs/bilingual_dictionary.json"
echo "  📁 outputs/token_difficulty/token_difficulty_scores.json"
echo "  📁 outputs/token_difficulty/token_categories.json"
echo "  📁 outputs/token_difficulty/difficulty_statistics.json"
echo ""
echo "Next step: Run Stage 0 training with:"
echo "  bash run_stage0_clsa.sh"
echo ""
