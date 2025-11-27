"""
KAVE: Korean Adaptive Vocabulary Expansion Framework

A novel approach for vocabulary expansion that preserves retrieval performance.

Components:
- WSA (Weighted Semantic Averaging): Better token initialization
- CTA (Contextual Token Alignment): Hybrid MLM + Contrastive loss
- PEU (Progressive Embedding Unfreezing): Gradual learning
- RAT (Retrieval-Aware Training): Preserve retrieval performance
"""

from .kave_framework import (
    KAVEConfig,
    KAVEFramework,
    WeightedSemanticAveraging,
    ContextualTokenAlignment,
    ProgressiveEmbeddingUnfreezing,
    RetrievalAwareTraining,
    create_kave_config
)

from .kave_trainer import (
    KAVETrainingArgs,
    KAVETrainer
)

__all__ = [
    'KAVEConfig',
    'KAVEFramework',
    'WeightedSemanticAveraging',
    'ContextualTokenAlignment',
    'ProgressiveEmbeddingUnfreezing',
    'RetrievalAwareTraining',
    'create_kave_config',
    'KAVETrainingArgs',
    'KAVETrainer'
]

__version__ = '1.0.0'
