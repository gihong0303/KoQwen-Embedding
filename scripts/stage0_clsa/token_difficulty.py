#!/usr/bin/env python3
"""
Token Difficulty Scoring System
토큰별 학습 난이도 계산
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm
import logging

from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenDifficultyScorer:
    """
    Calculate difficulty scores for tokens based on:
    1. Subword decomposition complexity (서브워드 개수)
    2. Corpus frequency (빈도 - 낮을수록 어려움)
    3. Semantic ambiguity (다의성 - 여러 문맥에서 사용되면 어려움)
    """

    def __init__(
        self,
        base_tokenizer_path: str,
        korean_tokenizer_path: str,
        vocab_diff_path: str
    ):
        logger.info("Initializing TokenDifficultyScorer...")

        # Load tokenizers
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_tokenizer_path,
            trust_remote_code=True
        )
        self.korean_tokenizer = AutoTokenizer.from_pretrained(
            korean_tokenizer_path,
            trust_remote_code=True
        )

        # Load new Korean tokens
        with open(vocab_diff_path, 'r', encoding='utf-8') as f:
            self.vocab_diff = json.load(f)

        self.new_korean_tokens = list(self.vocab_diff.keys())
        logger.info(f"Loaded {len(self.new_korean_tokens):,} new Korean tokens")

    def compute_subword_complexity(self, token: str) -> float:
        """
        서브워드 분해 복잡도 (0-1, 높을수록 복잡)

        Examples:
            "안녕" → 1 subword → 0.1
            "국제연합평화유지군" → 8 subwords → 0.9
        """
        subtokens = self.base_tokenizer.tokenize(token)

        if not subtokens:
            return 1.0  # Cannot decompose = very complex

        num_subtokens = len(subtokens)

        # Normalize: 1 subtoken = 0.0, 10+ subtokens = 1.0
        complexity = min(num_subtokens - 1, 10) / 10.0

        return complexity

    def compute_corpus_frequency(
        self,
        dataset_name: str = "HAERAE-HUB/KOREAN-WEBTEXT",
        max_samples: int = 100000,
        local_path: str = None
    ) -> Dict[str, int]:
        """
        코퍼스에서 토큰 빈도 계산

        Returns:
            {token: frequency_count}
        """
        logger.info(f"Computing corpus frequency from {dataset_name}...")

        # Load dataset
        if local_path:
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)
        else:
            dataset = load_dataset(dataset_name, split='train', streaming=True)

        token_counts = Counter()

        # Count token occurrences
        for idx, example in enumerate(tqdm(dataset, total=max_samples, desc="Counting tokens")):
            if idx >= max_samples:
                break

            text = example.get('text', '')
            if not text:
                continue

            # Tokenize with Korean tokenizer
            token_ids = self.korean_tokenizer.encode(text, add_special_tokens=False)

            # Convert to tokens
            tokens = self.korean_tokenizer.convert_ids_to_tokens(token_ids)

            # Count
            token_counts.update(tokens)

        logger.info(f"Counted {len(token_counts):,} unique tokens")

        return dict(token_counts)

    def compute_frequency_score(
        self,
        token: str,
        frequency_dict: Dict[str, int]
    ) -> float:
        """
        빈도 기반 난이도 (0-1, 높을수록 어려움)

        Low frequency = difficult
        High frequency = easy
        """
        freq = frequency_dict.get(token, 0)

        if freq == 0:
            return 1.0  # Never seen = very difficult

        # Log scale normalization
        # Assume: 1 occurrence = difficult (1.0), 10000+ = easy (0.0)
        log_freq = np.log10(freq + 1)
        max_log_freq = 5.0  # log10(100000)

        score = max(0.0, 1.0 - log_freq / max_log_freq)

        return score

    def compute_semantic_ambiguity(
        self,
        token: str,
        frequency_dict: Dict[str, int]
    ) -> float:
        """
        의미적 모호성 (다의성)

        Simple heuristic:
        - 문자 길이가 짧고 빈도가 높으면 다의어일 가능성 높음
        - 예: "배" (fruit/ship/stomach) vs "사과나무" (apple tree - unambiguous)
        """
        freq = frequency_dict.get(token, 0)
        token_clean = token.replace('▁', '').replace('Ġ', '')

        if len(token_clean) == 0:
            return 0.0

        # Short tokens with high frequency tend to be ambiguous
        length = len(token_clean)
        freq_score = min(np.log10(freq + 1) / 4.0, 1.0)  # 0-1

        # Combine: short + frequent = ambiguous
        if length <= 2 and freq_score > 0.5:
            ambiguity = 0.7  # High ambiguity
        elif length <= 3 and freq_score > 0.4:
            ambiguity = 0.5  # Medium ambiguity
        else:
            ambiguity = 0.2  # Low ambiguity

        return ambiguity

    def compute_difficulty_scores(
        self,
        frequency_dict: Dict[str, int],
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> Dict[str, float]:
        """
        전체 난이도 점수 계산

        Args:
            frequency_dict: Token frequency dictionary
            weights: (complexity_weight, frequency_weight, ambiguity_weight)

        Returns:
            {token: difficulty_score} (0-1, higher = more difficult)
        """
        logger.info("Computing difficulty scores for all tokens...")

        w_complexity, w_frequency, w_ambiguity = weights

        difficulty_scores = {}

        for token in tqdm(self.new_korean_tokens, desc="Computing difficulty"):
            complexity = self.compute_subword_complexity(token)
            frequency = self.compute_frequency_score(token, frequency_dict)
            ambiguity = self.compute_semantic_ambiguity(token, frequency_dict)

            # Weighted sum
            difficulty = (
                w_complexity * complexity +
                w_frequency * frequency +
                w_ambiguity * ambiguity
            )

            difficulty_scores[token] = difficulty

        logger.info(f"Computed difficulty scores for {len(difficulty_scores):,} tokens")

        return difficulty_scores

    def categorize_by_difficulty(
        self,
        difficulty_scores: Dict[str, float],
        percentiles: Tuple[float, float] = (0.33, 0.67)
    ) -> Dict[str, List[str]]:
        """
        난이도별로 토큰 분류

        Args:
            difficulty_scores: {token: difficulty}
            percentiles: (easy_threshold, medium_threshold)

        Returns:
            {
                'easy': [...],
                'medium': [...],
                'hard': [...]
            }
        """
        scores = list(difficulty_scores.values())
        p33, p67 = np.percentile(scores, [percentiles[0] * 100, percentiles[1] * 100])

        categories = {
            'easy': [],
            'medium': [],
            'hard': []
        }

        for token, score in difficulty_scores.items():
            if score <= p33:
                categories['easy'].append(token)
            elif score <= p67:
                categories['medium'].append(token)
            else:
                categories['hard'].append(token)

        logger.info(f"Token categorization:")
        logger.info(f"  Easy: {len(categories['easy']):,} tokens")
        logger.info(f"  Medium: {len(categories['medium']):,} tokens")
        logger.info(f"  Hard: {len(categories['hard']):,} tokens")

        return categories

    def save_results(
        self,
        difficulty_scores: Dict[str, float],
        categories: Dict[str, List[str]],
        output_dir: str
    ):
        """Save difficulty scores and categories"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save difficulty scores
        scores_path = output_dir / "token_difficulty_scores.json"
        with open(scores_path, 'w', encoding='utf-8') as f:
            json.dump(difficulty_scores, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved difficulty scores to: {scores_path}")

        # Save categories
        categories_path = output_dir / "token_categories.json"
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved categories to: {categories_path}")

        # Statistics
        stats = {
            'total_tokens': len(difficulty_scores),
            'easy_count': len(categories['easy']),
            'medium_count': len(categories['medium']),
            'hard_count': len(categories['hard']),
            'mean_difficulty': float(np.mean(list(difficulty_scores.values()))),
            'std_difficulty': float(np.std(list(difficulty_scores.values()))),
            'min_difficulty': float(min(difficulty_scores.values())),
            'max_difficulty': float(max(difficulty_scores.values()))
        }

        stats_path = output_dir / "difficulty_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to: {stats_path}")

        logger.info(f"\n📊 Difficulty Statistics:")
        logger.info(f"  Mean: {stats['mean_difficulty']:.3f}")
        logger.info(f"  Std: {stats['std_difficulty']:.3f}")
        logger.info(f"  Range: [{stats['min_difficulty']:.3f}, {stats['max_difficulty']:.3f}]")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Compute token difficulty scores")
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base tokenizer path"
    )
    parser.add_argument(
        "--korean_tokenizer",
        type=str,
        default="outputs/koqwen-expanded",
        help="Korean-expanded tokenizer path"
    )
    parser.add_argument(
        "--vocab_diff_path",
        type=str,
        default="tokenizer/vocab_diff.json",
        help="Vocabulary difference file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HAERAE-HUB/KOREAN-WEBTEXT",
        help="Dataset for frequency analysis"
    )
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default=None,
        help="Local dataset path (if using local data)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum samples for frequency analysis"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/token_difficulty",
        help="Output directory"
    )

    args = parser.parse_args()

    # Initialize scorer
    scorer = TokenDifficultyScorer(
        base_tokenizer_path=args.base_tokenizer,
        korean_tokenizer_path=args.korean_tokenizer,
        vocab_diff_path=args.vocab_diff_path
    )

    # Compute corpus frequency
    frequency_dict = scorer.compute_corpus_frequency(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        local_path=args.local_dataset_path
    )

    # Compute difficulty scores
    difficulty_scores = scorer.compute_difficulty_scores(frequency_dict)

    # Categorize
    categories = scorer.categorize_by_difficulty(difficulty_scores)

    # Save
    scorer.save_results(difficulty_scores, categories, args.output_dir)

    logger.info("✅ Token difficulty scoring complete!")


if __name__ == "__main__":
    main()
