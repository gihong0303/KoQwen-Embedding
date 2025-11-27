#!/usr/bin/env python3
"""
Token Difficulty Calculator for Curriculum Learning

Categorizes new Korean tokens into easy/medium/hard based on:
- Subword complexity (number of subwords when tokenized by base tokenizer)
- Character length
- Morphological complexity (jamo composition)

Usage:
    python scripts/prepare_curriculum.py --tokenizer_path outputs/koqwen-expanded

Output:
    outputs/token_difficulty/token_categories.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed")
    sys.exit(1)


def is_korean(char: str) -> bool:
    """Check if character is Korean (Hangul)"""
    return '\uac00' <= char <= '\ud7a3' or '\u1100' <= char <= '\u11ff' or '\u3130' <= char <= '\u318f'


def count_jamo(text: str) -> int:
    """Count total jamo (consonants + vowels) in Korean text"""
    count = 0
    for char in text:
        if '\uac00' <= char <= '\ud7a3':
            # Decompose Hangul syllable
            code = ord(char) - 0xac00
            # Each syllable = initial + medial + (optional final)
            count += 2 if code % 28 == 0 else 3
        elif '\u1100' <= char <= '\u11ff':
            count += 1
    return count


def calculate_token_difficulty(
    token: str,
    base_tokenizer,
    new_tokenizer
) -> Tuple[float, Dict]:
    """
    Calculate difficulty score for a token

    Returns:
        (difficulty_score, detail_dict)
    """
    # Skip non-Korean tokens
    korean_chars = sum(1 for c in token if is_korean(c))
    if korean_chars == 0:
        return 0.0, {'skip': True, 'reason': 'non-korean'}

    # 1. Subword complexity (how many pieces in base tokenizer)
    base_tokens = base_tokenizer.tokenize(token)
    subword_count = len(base_tokens)
    subword_score = min(subword_count / 5.0, 1.0)  # Normalize to 0-1

    # 2. Character length
    char_length = len(token.replace('▁', '').replace('Ġ', ''))
    length_score = min(char_length / 10.0, 1.0)  # Normalize to 0-1

    # 3. Jamo complexity
    jamo_count = count_jamo(token)
    jamo_score = min(jamo_count / 15.0, 1.0)  # Normalize to 0-1

    # 4. Special character penalty
    special_chars = sum(1 for c in token if not is_korean(c) and not c.isalnum())
    special_score = min(special_chars / 3.0, 1.0)

    # Combined difficulty score (weighted average)
    difficulty = (
        0.35 * subword_score +
        0.25 * length_score +
        0.25 * jamo_score +
        0.15 * special_score
    )

    details = {
        'subword_count': subword_count,
        'char_length': char_length,
        'jamo_count': jamo_count,
        'special_chars': special_chars,
        'difficulty': difficulty
    }

    return difficulty, details


def categorize_tokens(
    tokenizer_path: str,
    base_model: str = "Qwen/Qwen3-Embedding-0.6B",
    original_vocab_size: int = 151669,
    output_dir: str = "outputs/token_difficulty"
) -> Dict[str, List[str]]:
    """
    Categorize all new tokens into easy/medium/hard

    Args:
        tokenizer_path: Path to expanded tokenizer
        base_model: Base model for comparison
        original_vocab_size: Original vocabulary size
        output_dir: Output directory

    Returns:
        Dictionary with 'easy', 'medium', 'hard' token lists
    """
    print("=" * 60)
    print("Token Difficulty Calculator")
    print("=" * 60)

    # Load tokenizers
    print(f"\nLoading tokenizers...")
    print(f"  Base: {base_model}")
    print(f"  Expanded: {tokenizer_path}")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    new_vocab_size = len(new_tokenizer)
    num_new_tokens = new_vocab_size - original_vocab_size

    print(f"\nVocabulary:")
    print(f"  Original: {original_vocab_size:,}")
    print(f"  Expanded: {new_vocab_size:,}")
    print(f"  New tokens: {num_new_tokens:,}")

    # Calculate difficulty for each new token
    print(f"\nCalculating difficulty scores...")

    token_difficulties = []

    for token_id in range(original_vocab_size, new_vocab_size):
        token = new_tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            continue

        difficulty, details = calculate_token_difficulty(token, base_tokenizer, new_tokenizer)

        if not details.get('skip'):
            token_difficulties.append({
                'token': token,
                'token_id': token_id,
                'difficulty': difficulty,
                **details
            })

    print(f"  Processed {len(token_difficulties):,} Korean tokens")

    # Sort by difficulty
    token_difficulties.sort(key=lambda x: x['difficulty'])

    # Categorize into easy/medium/hard (30/40/30 split)
    n = len(token_difficulties)
    easy_cutoff = int(n * 0.3)
    hard_cutoff = int(n * 0.7)

    categories = {
        'easy': [t['token'] for t in token_difficulties[:easy_cutoff]],
        'medium': [t['token'] for t in token_difficulties[easy_cutoff:hard_cutoff]],
        'hard': [t['token'] for t in token_difficulties[hard_cutoff:]]
    }

    print(f"\nCategories:")
    print(f"  Easy (30%): {len(categories['easy']):,} tokens")
    print(f"  Medium (40%): {len(categories['medium']):,} tokens")
    print(f"  Hard (30%): {len(categories['hard']):,} tokens")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save categories
    categories_file = output_path / "token_categories.json"
    with open(categories_file, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {categories_file}")

    # Save detailed scores
    details_file = output_path / "token_difficulty_details.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(token_difficulties, f, ensure_ascii=False, indent=2)
    print(f"Saved: {details_file}")

    # Print examples
    print(f"\nExamples:")
    print(f"  Easy tokens: {categories['easy'][:5]}")
    print(f"  Medium tokens: {categories['medium'][:5]}")
    print(f"  Hard tokens: {categories['hard'][:5]}")

    return categories


def main():
    parser = argparse.ArgumentParser(description="Calculate token difficulty for curriculum learning")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="outputs/koqwen-expanded",
        help="Path to expanded tokenizer"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Base model name"
    )
    parser.add_argument(
        "--original_vocab_size",
        type=int,
        default=151669,
        help="Original vocabulary size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/token_difficulty",
        help="Output directory"
    )

    args = parser.parse_args()

    categorize_tokens(
        tokenizer_path=args.tokenizer_path,
        base_model=args.base_model,
        original_vocab_size=args.original_vocab_size,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print("Done! Token categories saved to outputs/token_difficulty/")
    print("=" * 60)


if __name__ == "__main__":
    main()
