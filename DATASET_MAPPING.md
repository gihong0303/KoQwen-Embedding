# 6-Stage Pipeline Local Dataset Mapping

## Overview
All datasets are now loaded from `~/haerae_dataset` using the `LocalDatasetLoader` class.

## Dataset Mapping by Stage

### Stage 1: New Token Input Embeddings
- **Dataset**: HuggingFace Hub (HAERAE-HUB/KOREAN-WEBTEXT)
- **Purpose**: Initial new token learning
- **Size**: 300K samples
- **Rationale**: Use original HuggingFace dataset for first stage, then switch to local

### Stage 2: New Token Alignment
- **Dataset**: `KOREAN-WEBTEXT` (local)
- **Purpose**: Continue new token learning with general Korean text
- **Size**: 300K samples
- **Rationale**: Same domain as Stage 1, but from local cache

### Stage 3: New Token Refinement
- **Dataset**: `KOREAN-SyntheticText` (local)
- **Purpose**: Refine new tokens with high-quality synthetic text
- **Size**: 200K samples
- **Rationale**: High-quality generated text for better token representations

### Stage 4: Full Vocabulary Harmonization
- **Dataset**: Mixed (local)
  - `KOREAN-WEBTEXT`: 100K samples
  - `KOREAN-SyntheticText`: 80K samples
  - `KoSimpleEval`: 20K samples
- **Purpose**: Harmonize old and new tokens with diverse data
- **Total**: 200K samples
- **Rationale**: Diversity helps bridge old and new token spaces

### Stage 5: Transformer Enhancement (LoRA)
- **Dataset**: Mixed (local)
  - `HAE-RAE-COT`: 100K samples (Question + CoT_Rationale)
  - `HR-Instruct-Math`: 100K samples (instruction + response)
- **Purpose**: Enhance transformer layers with reasoning data
- **Total**: 200K samples
- **Rationale**: Reasoning tasks improve semantic understanding

### Stage 6: Advanced Contrastive Learning
- **Dataset**: `K2-Feedback` (local, score=5 only)
- **Purpose**: Final refinement with high-quality feedback data
- **Size**: 150K samples
- **Rationale**: Quality-focused data for final polishing

## Local Dataset Files

```
~/haerae_dataset/
├── KOREAN-WEBTEXT/default/train/*.parquet (18 files)
├── KOREAN-SyntheticText/default/train/*.parquet (13 files)
├── KoSimpleEval/[24 subsets]/test/*.parquet
├── HAE-RAE-COT/default/train/*.parquet (4 files)
├── HR-Instruct-Math/default/train/*.parquet (1 file)
└── K2-Feedback/default/train/*.parquet (2 files)
```

## Implementation Details

### LocalDatasetLoader Features
- Automatic text column detection
- Support for specialized loaders (COT, Math, Feedback, KoSimpleEval)
- Mixed dataset support with shuffling
- Max samples control for memory efficiency
- Field combination (Question+CoT, instruction+response)

### Configuration Format
```yaml
dataset:
  local: true
  local_path: "~/haerae_dataset"
  name: "KOREAN-WEBTEXT"
  max_samples: 300000
```

For mixed datasets:
```yaml
dataset:
  local: true
  local_path: "~/haerae_dataset"
  mixed:
    - name: "KOREAN-WEBTEXT"
      max_samples: 100000
    - name: "KOREAN-SyntheticText"
      max_samples: 80000
```

## Benefits
1. **No Download Time**: All data is pre-downloaded
2. **Fast Loading**: Direct parquet file reading
3. **Flexibility**: Easy to adjust sample sizes
4. **Reproducibility**: Fixed local datasets ensure consistency
5. **Mixed Datasets**: Combine multiple sources for diversity
