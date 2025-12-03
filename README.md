# Korean Embedding Expansion for Qwen3-Embedding-0.6B

**Curriculum Learning + Contrastive Training for Korean Embedding Enhancement**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

This project extends **Qwen3-Embedding-0.6B** with **68,029 Korean-specific tokens** through a **5-stage curriculum learning pipeline**.

### Key Features

- **5-Stage Curriculum Pipeline**: Progressive learning from easy to hard tokens
- **8 GPU DDP Training**: Optimized for multi-GPU distributed training
- **MTEB Evaluation**: Built-in evaluation on 6 Korean retrieval tasks
- **Retrieval-Safe Design**: Stage 6 removed to preserve retrieval performance

---

## 5-Stage Pipeline

| Stage | Focus | Trainable | Dataset | Samples |
|-------|-------|-----------|---------|---------|
| **Stage 1** | Easy tokens (30%) | New tokens | WEBTEXT + KorNLI | 500K |
| **Stage 2** | Medium tokens (40%) | New tokens | WEBTEXT + KorNLI | 500K |
| **Stage 3** | Hard tokens (30%) | New tokens | WEBTEXT + SyntheticText + KorNLI | 500K |
| **Stage 4** | Full vocab alignment | All tokens | Mixed (4 datasets) | 500K |
| **Stage 5** | Semantic fine-tuning | LoRA (r=64) | KorNLI + COT | 500K |

**Note**: Stage 6 was removed because SimCSE unsupervised loss destroys retrieval performance (97%+ NDCG drop observed).

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Hardware: 8x GPUs (A100 recommended) with CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
```

### Run Training Pipeline

```bash
# Run all stages
./run_pipeline.sh

# Run specific stage
./run_pipeline.sh --stage 3

# Resume from stage
./run_pipeline.sh --resume 4
```

### Evaluate Model

```bash
# Run MTEB Korean Retrieval evaluation
./run_pipeline.sh --eval

# Or run manually
python scripts/evaluate_mteb.py --model_path checkpoints/stage5/final

# Compare with baseline
python scripts/evaluate_mteb.py \
    --model_path checkpoints/stage5/final \
    --compare Qwen/Qwen3-Embedding-0.6B
```

---

## MTEB Evaluation Results

### Korean Retrieval Benchmark (nDCG@10)

| Task | Baseline (Qwen3-Embedding-0.6B) | Trained | Change |
|------|--------------------------------|---------|--------|
| AutoRAGRetrieval | 0.7452 | 0.7482 | **+0.40%** |
| BelebeleRetrieval | 0.6039 | 0.6517 | **+7.93%** |
| Ko-StrategyQA | 0.5772 | 0.6039 | **+4.62%** |
| PublicHealthQA | 0.7426 | 0.7543 | **+1.58%** |
| MIRACLRetrieval | 0.3469 | 0.3198 | -7.81% |
| MrTidyRetrieval | 0.2803 | 0.2525 | -9.93% |
| **Average** | **0.5494** | **0.5551** | **+1.04%** |

### Summary

- **4/6 tasks improved**: AutoRAG, Belebele, Ko-StrategyQA, PublicHealthQA
- **Best improvement**: BelebeleRetrieval (+7.93%)
- **Overall average**: +1.04% improvement

---

## MTEB Evaluation Tasks

The model is evaluated on 6 Korean retrieval tasks:

| Task | Type | Description |
|------|------|-------------|
| Ko-StrategyQA | Retrieval | Korean strategy QA |
| AutoRAGRetrieval | Retrieval | Korean RAG benchmark |
| BelebeleRetrieval | Retrieval | Multilingual reading comprehension |
| PublicHealthQA | Retrieval | Korean health QA |
| MIRACLRetrieval | Retrieval | Multilingual retrieval |
| MrTidyRetrieval | Retrieval | Multilingual retrieval |

---

## Project Structure

```
KoQwen-Embedding/
├── configs/
│   └── pipeline_config.yaml      # Main configuration
├── scripts/
│   ├── base_trainer.py           # Base trainer with DDP
│   ├── enhanced_trainer.py       # Curriculum-aware trainer
│   ├── stage1_curriculum.py      # Stage 1: Easy tokens
│   ├── stage2_curriculum.py      # Stage 2: Medium tokens
│   ├── stage3_curriculum.py      # Stage 3: Hard tokens
│   ├── stage4.py                 # Stage 4: Full vocab
│   ├── stage5.py                 # Stage 5: LoRA fine-tuning
│   └── evaluate_mteb.py          # MTEB evaluation script
├── utils/
│   ├── contrastive_loss.py       # SimCSE contrastive loss
│   ├── curriculum_dataset.py     # Curriculum learning
│   └── local_dataset_loader.py   # Dataset loading
├── run_pipeline.sh               # Main pipeline script
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `configs/pipeline_config.yaml` to customize:

```yaml
# Hardware (adjust for your setup)
hardware:
  num_gpus: 8
  gpu_ids: [1, 2, 3, 4, 5, 6, 7, 8]

# Training (batch sizes already 2x optimized)
stage1:
  training:
    batch_size: 48
    gradient_accumulation_steps: 2
    learning_rate: 3e-4
```

---

## DDP Training

The pipeline uses PyTorch's DistributedDataParallel (DDP) with:

- **torchrun** for process management
- **NCCL** backend for GPU communication
- **30-minute timeout** for large model loading
- **Barrier synchronization** for checkpoint saving

```bash
# Environment variables set by run_pipeline.sh
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7,8"
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
```

---

## Stage 6 Removal Rationale

Stage 6 used SimCSE unsupervised loss with K2-Feedback data:

```python
# SimCSE: Same text twice with different dropout
outputs1 = model(input_ids)  # dropout mask A
outputs2 = model(input_ids)  # dropout mask B
loss = contrastive_loss(outputs1, outputs2)
```

**Problem**: This trains the model to recognize identical texts, NOT to match queries with relevant documents.

**Impact**: 97-98% NDCG drop on retrieval tasks:
- Ko-StrategyQA: 0.5766 → 0.0144 (-97.5%)
- AutoRAGRetrieval: 0.7470 → 0.0137 (-98.2%)

**Solution**: Remove Stage 6 to preserve retrieval capability.

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
  title={Korean Embedding Expansion with Curriculum Learning},
  author={gihong0303},
  year={2024},
  howpublished={\url{https://github.com/gihong0303/KoQwen-Embedding}}
}
```

---

## Acknowledgments

- **Qwen Team**: Qwen3-Embedding-0.6B base model
- **HAERAE-HUB**: Korean datasets
- **Kakaobrain**: KorNLI dataset

---

## License

MIT License - see [LICENSE](LICENSE) for details.
