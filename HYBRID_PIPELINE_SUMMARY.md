# Hybrid Pipeline Summary: CLSA + JLCE + MCL

## Overview

Korean Embedding Enhancement using a hybrid approach combining:
- **Stage 0**: Cross-lingual Semantic Anchoring (CLSA)
- **Stage 1-3**: Jamo-Level Compositional Embedding (JLCE) + Morphological Curriculum Learning (MCL)
- **Stage 4-6**: Standard Contrastive Learning with LoRA

---

## Stage-by-Stage Configuration

### Stage 0: Cross-lingual Semantic Anchoring (CLSA)

**Purpose**: Align new Korean tokens to English/Chinese semantic space

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `CLSALoss` |
| | - Alignment Loss: `1 - cosine_similarity(korean_emb, anchor_center)` |
| | - Diversity Loss: Pairwise similarity regularization (weight=0.1) |
| **Optimizer** | AdamW |
| | - Learning Rate: 5e-4 (higher for fast initialization) |
| | - Weight Decay: 0.01 |
| | - Warmup: 20% |
| **Dataset** | Bilingual Dictionary |
| | - Korean-English-Chinese token mappings |
| | - ~50K token pairs |
| **Training** | |
| | - Epochs: 3 |
| | - Batch Size: 128 |
| | - Grad Accum: 2 |

**Evidence**: CrossInit (ACL 2024), Trans-Tokenization (Aug 2024)

---

### Stage 1: Easy Tokens (JLCE + MCL)

**Purpose**: Train high-frequency, simple morphological structure tokens

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `JLCEMCLLoss` (stage=1) |
| | - Jamo Weight: 0.5 (emphasize compositional learning) |
| | - Curriculum Weight: 0.2 |
| | - Contrastive Weight: 0.3 |
| **Optimizer** | AdamW |
| | - Learning Rate: 3e-4 |
| | - Weight Decay: 0.01 |
| | - Warmup: 10% |
| **Dataset** | |
| | - KOREAN-WEBTEXT: 150K samples |
| | - KorNLI: 100K samples |
| | - **Curriculum Mode**: easy (단일 어근) |
| **Training** | |
| | - Epochs: 2 |
| | - Batch Size: 24 |
| | - Grad Accum: 3 |

**Token Examples**: 집, 학교, 먹다, 좋다 (simple stems)

---

### Stage 2: Medium Tokens (JLCE + MCL)

**Purpose**: Train stem + affix combination tokens

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `JLCEMCLLoss` (stage=2) |
| | - Jamo Weight: 0.4 (balanced) |
| | - Curriculum Weight: 0.3 |
| | - Contrastive Weight: 0.3 |
| **Optimizer** | AdamW |
| | - Learning Rate: 2e-4 |
| | - Weight Decay: 0.01 |
| | - Warmup: 10% |
| **Dataset** | |
| | - KOREAN-WEBTEXT: 150K samples |
| | - KorNLI: 100K samples |
| | - **Curriculum Mode**: medium (어근+조사) |
| **Training** | |
| | - Epochs: 1 |
| | - Batch Size: 16 |
| | - Grad Accum: 4 |

**Token Examples**: 집에, 학교로, 먹고, 커서 (stem + particle)

---

### Stage 3: Hard Tokens (JLCE + MCL)

**Purpose**: Train complex morphological structure, low-frequency tokens

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `JLCEMCLLoss` (stage=3) |
| | - Jamo Weight: 0.3 (emphasize curriculum) |
| | - Curriculum Weight: 0.4 |
| | - Contrastive Weight: 0.3 |
| **Optimizer** | AdamW |
| | - Learning Rate: 1e-4 (careful learning) |
| | - Weight Decay: 0.01 |
| | - Warmup: 15% |
| | - Grad Clipping: 0.5 (smaller) |
| **Dataset** | |
| | - KOREAN-SyntheticText: 100K samples |
| | - KorNLI: 50K samples |
| | - **Curriculum Mode**: hard (복합 형태) |
| **Training** | |
| | - Epochs: 1 |
| | - Batch Size: 12 |
| | - Grad Accum: 5 |

**Token Examples**: 학교에서부터, 먹지않았다 (complex morphology)

---

### Stage 4: Full Vocabulary Harmonization

**Purpose**: Harmonize old and new token embeddings

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `StandardContrastiveLoss` |
| | - Temperature: 0.05 |
| | - SimCSE style |
| **Optimizer** | AdamW |
| | - Learning Rate: 5e-5 |
| | - Weight Decay: 0.01 |
| **Dataset** | |
| | - KOREAN-WEBTEXT: 100K |
| | - KOREAN-SyntheticText: 60K |
| | - KoSimpleEval: 20K |
| | - KorNLI: 20K |
| **Training** | |
| | - Epochs: 1 |
| | - Batch Size: 12 |
| | - train_new_tokens_only: false |

---

### Stage 5: LoRA Transformer Enhancement

**Purpose**: Fine-tune transformer layers with LoRA

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `StandardContrastiveLoss` |
| **LoRA Config** | |
| | - r: 64, alpha: 128 |
| | - Target: q_proj, k_proj, v_proj, o_proj |
| **Optimizer** | AdamW |
| | - Learning Rate: 5e-5 |
| **Dataset** | |
| | - HAE-RAE-COT: 100K (reasoning) |
| | - HR-Instruct-Math: 100K (math) |
| **Training** | |
| | - Epochs: 1 |
| | - Batch Size: 10 |
| | - Gradient Checkpointing: true |

---

### Stage 6: Advanced Contrastive Learning

**Purpose**: Final refinement with high-quality feedback data

| Component | Configuration |
|-----------|--------------|
| **Loss Function** | `StandardContrastiveLoss` |
| **LoRA Config** | |
| | - r: 32, alpha: 64 |
| **Optimizer** | AdamW |
| | - Learning Rate: 3e-5 |
| **Dataset** | |
| | - K2-Feedback: 150K (min_score >= 5) |
| **Training** | |
| | - Epochs: 2 |
| | - Batch Size: 12 |

---

## Loss Functions Summary

### 1. CLSALoss (Stage 0)
```python
loss = alignment_loss + λ * diversity_loss
     = (1 - cosine_sim(ko_emb, anchor_center)) + 0.1 * pairwise_sim(ko_embs)
```

### 2. JLCEMCLLoss (Stage 1-3)
```python
loss = w1 * contrastive_loss + w2 * jamo_loss + w3 * curriculum_loss

# Stage 1: w1=0.3, w2=0.5, w3=0.2
# Stage 2: w1=0.3, w2=0.4, w3=0.3
# Stage 3: w1=0.3, w2=0.3, w3=0.4
```

### 3. StandardContrastiveLoss (Stage 4-6)
```python
loss = CrossEntropy(sim_matrix / temperature, labels)
# SimCSE style with InfoNCE
```

---

## Optimization Summary

| Stage | LR | Warmup | Scheduler | Grad Clip |
|-------|-----|--------|-----------|-----------|
| 0 | 5e-4 | 20% | cosine | 1.0 |
| 1 | 3e-4 | 10% | cosine | 1.0 |
| 2 | 2e-4 | 10% | cosine | 1.0 |
| 3 | 1e-4 | 15% | cosine_restarts | 0.5 |
| 4 | 5e-5 | 10% | cosine | 1.0 |
| 5 | 5e-5 | 5% | cosine | 1.0 |
| 6 | 3e-5 | 10% | linear | 1.0 |

---

## Dataset Summary

| Stage | Dataset | Source | Samples | Purpose |
|-------|---------|--------|---------|---------|
| 0 | Bilingual Dict | Generated | ~50K | Cross-lingual anchors |
| 1 | WEBTEXT + KorNLI | HuggingFace | 250K | Easy tokens |
| 2 | WEBTEXT + KorNLI | HuggingFace | 250K | Medium tokens |
| 3 | SyntheticText + KorNLI | Mixed | 150K | Hard tokens |
| 4 | Mixed (4 datasets) | Mixed | 200K | Harmonization |
| 5 | COT + Math | Local | 200K | Reasoning |
| 6 | K2-Feedback | Local | 150K | Quality refinement |

**Total**: ~1.25M samples

---

## Expected Performance

| Metric | Baseline | CLSA Only | Hybrid (CLSA+JLCE+MCL) |
|--------|----------|-----------|------------------------|
| Ko-StrategyQA | +12.0% | +18% | **+22-28%** |
| PublicHealthQA | -5.5% | +5% | **+8-12%** |
| Avg NDCG@10 | +2.1% | +8% | **+12-18%** |
| Training Time | 9-10h | 16-18h | **18-22h** |

---

## Key Innovations

### 1. JLCE (Jamo-Level Compositional Embedding)
- 68 jamo embeddings instead of 68K token embeddings
- **Parameter reduction**: 99.9%
- Zero-shot generalization to unseen Hangul combinations

### 2. MCL (Morphological Curriculum Learning)
- Train by morphological complexity: stem → stem+affix → complex
- Aligned with Korean language acquisition patterns
- Progressive difficulty prevents catastrophic forgetting

### 3. Hybrid Pipeline
- Stage 0 (CLSA): Cross-lingual initialization
- Stage 1-3 (JLCE+MCL): Korean-specific compositional learning
- Stage 4-6: Standard fine-tuning

---

## Files Created

```
utils/
├── jamo_embedding.py         # JLCE implementation
├── morphological_curriculum.py  # MCL implementation
├── hybrid_loss.py            # Loss functions for all stages

scripts/
├── jlce_mcl_trainer.py       # Stage 1-3 trainer

configs/
├── pipeline_config.yaml      # Updated hybrid config
```

---

## Running the Pipeline

```bash
# 1. Preparation
./prepare.sh

# 2. Stage 0: CLSA
./run_stage0_clsa.sh

# 3. Stage 1-3: JLCE + MCL
torchrun --nproc_per_node=6 scripts/jlce_mcl_trainer.py --stage 1 --config configs/pipeline_config.yaml
torchrun --nproc_per_node=6 scripts/jlce_mcl_trainer.py --stage 2 --config configs/pipeline_config.yaml
torchrun --nproc_per_node=6 scripts/jlce_mcl_trainer.py --stage 3 --config configs/pipeline_config.yaml

# 4. Stage 4-6: Standard pipeline
torchrun --nproc_per_node=6 scripts/stage4.py --config configs/pipeline_config.yaml
torchrun --nproc_per_node=6 scripts/stage5.py --config configs/pipeline_config.yaml
torchrun --nproc_per_node=6 scripts/stage6.py --config configs/pipeline_config.yaml
```

---

## Academic Justification

| Method | Evidence |
|--------|----------|
| CLSA | CrossInit (ACL 2024), Trans-Tokenization (2024) |
| JLCE | CharacterBERT (2020), Subword Regularization (Google 2018) |
| MCL | Morphological Analysis (ACL papers), Language Acquisition Theory |
| Contrastive | SimCSE (EMNLP 2021), InfoNCE (van den Oord 2018) |

---

## Contact

For questions: GitHub Issues
