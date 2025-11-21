#!/usr/bin/env python3
"""
Curriculum Learning Dataset Wrapper
토큰 난이도 기반 커리큘럼 학습
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumDatasetWrapper:
    """
    Curriculum learning wrapper for datasets

    Filters/prioritizes samples based on token difficulty

    Usage:
        Stage 1: Easy tokens (30%)
        Stage 2: Medium tokens (40%)
        Stage 3: Hard tokens (30%)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer,
        difficulty_categories_path: str,
        curriculum_mode: str = 'easy',  # 'easy', 'medium', 'hard', 'all'
        priority_weight: float = 3.0     # How much to oversample priority tokens
    ):
        """
        Args:
            base_dataset: Base HuggingFace dataset
            tokenizer: Tokenizer for analyzing tokens
            difficulty_categories_path: Path to token_categories.json
            curriculum_mode: Which difficulty level to focus on
            priority_weight: Oversampling weight for priority tokens
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.curriculum_mode = curriculum_mode
        self.priority_weight = priority_weight

        # Load difficulty categories
        with open(difficulty_categories_path, 'r', encoding='utf-8') as f:
            self.categories = json.load(f)

        # Build token sets
        self.easy_tokens = set(self.categories.get('easy', []))
        self.medium_tokens = set(self.categories.get('medium', []))
        self.hard_tokens = set(self.categories.get('hard', []))

        # Select priority tokens based on curriculum mode
        if curriculum_mode == 'easy':
            self.priority_tokens = self.easy_tokens
        elif curriculum_mode == 'medium':
            self.priority_tokens = self.medium_tokens
        elif curriculum_mode == 'hard':
            self.priority_tokens = self.hard_tokens
        else:  # 'all'
            self.priority_tokens = set()

        logger.info(f"CurriculumDataset initialized:")
        logger.info(f"  Mode: {curriculum_mode}")
        logger.info(f"  Priority tokens: {len(self.priority_tokens):,}")
        logger.info(f"  Base dataset size: {len(base_dataset):,}")

        # Analyze dataset and compute sample weights
        self._compute_sample_weights()

    def _count_priority_tokens(self, text: str) -> int:
        """Count how many priority tokens appear in text"""
        if not self.priority_tokens:
            return 0

        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        # Count priority tokens
        count = sum(1 for token in tokens if token in self.priority_tokens)

        return count

    def _compute_sample_weights(self):
        """Compute sampling weights for each example in dataset"""
        logger.info("Computing sample weights for curriculum learning...")

        self.sample_weights = []

        for idx in range(min(len(self.base_dataset), 10000)):  # Sample for speed
            example = self.base_dataset[idx]
            text = example.get('text', '')

            if not text:
                self.sample_weights.append(1.0)
                continue

            # Count priority tokens
            priority_count = self._count_priority_tokens(text)

            # Weight: higher if contains more priority tokens
            if priority_count > 0:
                weight = 1.0 + (priority_count / 10.0) * self.priority_weight
            else:
                weight = 1.0

            self.sample_weights.append(weight)

        # Extend weights to full dataset (approximate)
        if len(self.sample_weights) < len(self.base_dataset):
            avg_weight = sum(self.sample_weights) / len(self.sample_weights)
            remaining = len(self.base_dataset) - len(self.sample_weights)
            self.sample_weights.extend([avg_weight] * remaining)

        logger.info(f"Sample weights computed:")
        logger.info(f"  Mean weight: {sum(self.sample_weights) / len(self.sample_weights):.2f}")
        logger.info(f"  Max weight: {max(self.sample_weights):.2f}")

    def get_weighted_indices(self, num_samples: int) -> List[int]:
        """
        Sample indices with replacement based on weights

        Args:
            num_samples: Number of samples to draw

        Returns:
            List of indices
        """
        indices = random.choices(
            range(len(self.base_dataset)),
            weights=self.sample_weights,
            k=num_samples
        )
        return indices

    def get_curriculum_subset(self, max_samples: Optional[int] = None) -> Dataset:
        """
        Get a curriculum-weighted subset of the dataset

        Args:
            max_samples: Maximum number of samples (None = full dataset)

        Returns:
            Filtered dataset
        """
        if max_samples is None:
            max_samples = len(self.base_dataset)

        # Sample with replacement based on weights
        indices = self.get_weighted_indices(max_samples)

        # Create subset
        subset = self.base_dataset.select(indices)

        logger.info(f"Curriculum subset created: {len(subset):,} samples")

        return subset

    @staticmethod
    def create_for_stage(
        base_dataset: Dataset,
        tokenizer,
        difficulty_categories_path: str,
        stage: int,
        priority_weight: float = 3.0
    ) -> Dataset:
        """
        Factory method: create curriculum dataset for a specific stage

        Stage 1: Easy tokens (top 30%)
        Stage 2: Medium tokens (middle 40%)
        Stage 3: Hard tokens (bottom 30%)

        Args:
            base_dataset: Base dataset
            tokenizer: Tokenizer
            difficulty_categories_path: Path to token categories
            stage: Stage number (1, 2, or 3)
            priority_weight: Oversampling weight

        Returns:
            Curriculum-weighted dataset
        """
        mode_map = {
            1: 'easy',
            2: 'medium',
            3: 'hard'
        }

        mode = mode_map.get(stage, 'all')

        wrapper = CurriculumDatasetWrapper(
            base_dataset=base_dataset,
            tokenizer=tokenizer,
            difficulty_categories_path=difficulty_categories_path,
            curriculum_mode=mode,
            priority_weight=priority_weight
        )

        return wrapper.get_curriculum_subset()


# Utility function for easy integration
def apply_curriculum(
    dataset: Dataset,
    tokenizer,
    difficulty_categories_path: str,
    stage: int,
    max_samples: Optional[int] = None,
    priority_weight: float = 3.0
) -> Dataset:
    """
    Apply curriculum learning to a dataset

    Args:
        dataset: Base dataset
        tokenizer: Tokenizer
        difficulty_categories_path: Path to token_categories.json
        stage: Stage number (1, 2, 3, or 4+)
        max_samples: Max samples (None = all)
        priority_weight: Oversampling factor for priority tokens

    Returns:
        Curriculum-filtered dataset
    """
    if stage >= 4:
        # Stage 4+: No curriculum, return as-is
        if max_samples:
            return dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    # Apply curriculum for stages 1-3
    wrapper = CurriculumDatasetWrapper(
        base_dataset=dataset,
        tokenizer=tokenizer,
        difficulty_categories_path=difficulty_categories_path,
        curriculum_mode={1: 'easy', 2: 'medium', 3: 'hard'}[stage],
        priority_weight=priority_weight
    )

    return wrapper.get_curriculum_subset(max_samples)
