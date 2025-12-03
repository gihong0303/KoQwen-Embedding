# Korean Embedding Expansion for Qwen3-Embedding-0.6B

**6-Stage Curriculum Learning + Supervised Retrieval Contrastive Training**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

This project extends **Qwen3-Embedding-0.6B** with **68,029 Korean-specific tokens** through a **6-stage curriculum learning pipeline**.

### Key Features

- **6-Stage Pipeline**: Curriculum learning (Stage 1-3) + Vocabulary harmonization (Stage 4) + NLI (Stage 5) + **Supervised Retrieval** (Stage 6)
- **8 GPU DDP Training**: Optimized for multi-GPU distributed training with PyTorch DDP
- **MTEB Evaluation**: Built-in parallel evaluation on 6 Korean retrieval tasks
- **Supervised Retrieval**: Stage 6 uses MIRACL + KorNLI for direct retrieval optimization

---

## 6-Stage Pipeline

| Stage | Focus | Trainable | Dataset | Key Feature |
|-------|-------|-----------|---------|-------------|
| **Stage 1** | Easy tokens (30%) | New tokens | WEBTEXT + KorNLI | Curriculum (easy first) |
| **Stage 2** | Medium tokens (40%) | New tokens | WEBTEXT + KorNLI | Curriculum (medium) |
| **Stage 3** | Hard tokens (30%) | New tokens | WEBTEXT + SyntheticText | Curriculum (hard) |
| **Stage 4** | Full vocab alignment | All tokens | Mixed (4 datasets) | Harmonization |
| **Stage 5** | Semantic fine-tuning | LoRA (r=64) | KorNLI + COT | NLI understanding |
| **Stage 6** | Retrieval optimization | LoRA (r=32) | **MIRACL + KorNLI** | **Supervised Contrastive** |

### Stage 6: Supervised Retrieval Contrastive

**Key difference from old SimCSE approach:**

```python
# OLD (SimCSE Unsupervised) - Same text twice, different dropout
outputs1 = model(text)  # dropout mask A
outputs2 = model(text)  # dropout mask B <- 자기 자신만 positive
loss = contrastive_loss(outputs1, outputs2)  # DESTROYS retrieval!

# NEW (Supervised Retrieval) - Real query-document pairs
query_emb = model(query)
pos_doc_emb = model(positive_document)   # 실제 관련 문서
neg_doc_emb = model(negative_document)   # Hard negative (native speaker annotated)
loss = retrieval_contrastive_loss(query_emb, pos_doc_emb, neg_doc_emb)
```

**Data Sources:**
- **MIRACL Korean**: Query-document pairs with hard negatives (native speaker annotated)
- **KorNLI**: Entailment pairs as pseudo-retrieval (premise→hypothesis)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Hardware: 8x GPUs (A5000/A100) with CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
```

### Run Training Pipeline

```bash
# Run all 6 stages
./run_pipeline.sh

# Run specific stage
./run_pipeline.sh --stage 6    # Run Stage 6 only

# Resume from stage
./run_pipeline.sh --resume 5   # Resume from Stage 5
```

### Evaluate Model

```bash
# Sequential evaluation (single GPU)
./run_pipeline.sh --eval

# Parallel evaluation (multi-GPU, faster for MIRACL/MrTidy)
./run_pipeline.sh --eval-parallel

# Compare with baseline
python scripts/evaluate_mteb_parallel.py \
    --model_path checkpoints/stage6/final \
    --compare Qwen/Qwen3-Embedding-0.6B
```

---

## Expected Results (Stage 6)

### Target: All 6 Tasks Improved

Stage 6의 Supervised Retrieval은 두 브랜치의 장점을 결합합니다:

| Task | main (SimCSE) | main-v2 (NLI) | Stage 6 (Expected) |
|------|--------------|---------------|-------------------|
| Ko-StrategyQA | +12.0% | +4.62% | ✅ Best of both |
| MrTidyRetrieval | +8.6% | -9.93% | ✅ MIRACL optimizes |
| BelebeleRetrieval | +3.3% | +7.93% | ✅ Both improved |
| MIRACLRetrieval | +2.7% | -7.81% | ✅ MIRACL optimizes |
| AutoRAGRetrieval | -2.1% | +0.40% | ✅ NLI preserves |
| PublicHealthQA | -5.5% | +1.58% | ✅ NLI preserves |

**Why Stage 6 should work:**
- MIRACL directly optimizes MrTidy/MIRACL performance (same domain)
- KorNLI preserves PublicHealthQA/AutoRAG performance
- Low learning rate (2e-5) prevents catastrophic forgetting

---

## Project Structure

```
KoQwen-Embedding/
├── configs/
│   └── pipeline_config.yaml      # 6-stage configuration
├── scripts/
│   ├── base_trainer.py           # Base DDP trainer
│   ├── stage1_curriculum.py      # Stage 1: Easy tokens
│   ├── stage2_curriculum.py      # Stage 2: Medium tokens
│   ├── stage3_curriculum.py      # Stage 3: Hard tokens
│   ├── stage4.py                 # Stage 4: Full vocab
│   ├── stage5.py                 # Stage 5: LoRA + NLI
│   ├── stage6_retrieval.py       # Stage 6: Supervised Retrieval
│   └── evaluate_mteb_parallel.py # Parallel MTEB evaluation
├── utils/
│   ├── contrastive_loss.py       # SimCSE loss (Stage 1-5)
│   ├── retrieval_loss.py         # Supervised retrieval loss (Stage 6)
│   ├── retrieval_dataset.py      # MIRACL + KorNLI loader
│   └── curriculum_dataset.py     # Curriculum learning
├── run_pipeline.sh               # Main pipeline script
└── README.md
```

---

## Configuration

Edit `configs/pipeline_config.yaml` to customize:

```yaml
# Stage 6 Configuration
stage6:
  use_lora: true
  lora_config:
    r: 32                         # Lower rank for fine refinement
    lora_alpha: 64

  retrieval:
    loss_type: "mnrl"             # Multiple Negatives Ranking Loss
    temperature: 0.05
    use_hard_negatives: true      # MIRACL hard negatives

  dataset:
    miracl_samples: 50000         # MIRACL Korean
    kornli_samples: 100000        # KorNLI entailment

  training:
    learning_rate: 2e-5           # Very low to preserve abilities
    batch_size: 24
```

---

## DDP Training

PyTorch DistributedDataParallel with best practices:

- **torchrun** for process management
- **NCCL** backend for GPU communication
- **30-minute timeout** for large model loading
- **Barrier synchronization** for checkpoint saving
- **find_unused_parameters=False** for efficiency

```bash
# Environment variables (set by run_pipeline.sh)
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

---

## MTEB Parallel Evaluation

For faster evaluation of slow tasks (MIRACL, MrTidy):

```python
# GPU mapping for parallel evaluation
TASK_GPU_MAPPING = {
    0: ["Ko-StrategyQA", "AutoRAGRetrieval", "PublicHealthQA", "BelebeleRetrieval"],
    1: ["MIRACLRetrieval"],    # Slow task on dedicated GPU
    2: ["MrTidyRetrieval"],    # Slow task on dedicated GPU
}
```

---

## Requirements

```txt
torch>=2.0
transformers>=4.40
peft>=0.10
datasets>=2.19
accelerate>=0.30
sentence-transformers>=2.7
mteb>=1.12
safetensors
tqdm
pyyaml
pandas
```

---

## Citation

```bibtex
@misc{koqwen-embedding-2024,
  title={Korean Embedding Expansion with Curriculum Learning and Supervised Retrieval},
  author={gihong0303},
  year={2024},
  howpublished={\url{https://github.com/gihong0303/KoQwen-Embedding}}
}
```

---

## Acknowledgments

- **Qwen Team**: Qwen3-Embedding-0.6B base model
- **MIRACL Team**: Multilingual retrieval dataset with hard negatives
- **HAERAE-HUB**: Korean datasets
- **Kakaobrain**: KorNLI dataset

---

## License

MIT License - see [LICENSE](LICENSE) for details.
