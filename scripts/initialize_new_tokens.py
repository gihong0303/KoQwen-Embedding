#!/usr/bin/env python3
"""
Subword Averaging Token Initialization (EEVE/Thunder Style)

새 토큰을 기존 subword들의 평균으로 초기화
- 랜덤 초기화보다 훨씬 좋은 시작점
- 기존 임베딩 공간에서 시작하므로 fine-tuning 용이

Usage:
    python scripts/initialize_new_tokens.py \
        --base_model Qwen/Qwen3-Embedding-0.6B \
        --tokenizer_path outputs/koqwen-expanded \
        --output_path outputs/initialized_model
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModel


def initialize_with_subword_average(
    base_model_name: str,
    expanded_tokenizer_path: str,
    output_path: str,
    original_vocab_size: int = 151669
):
    """
    Subword averaging으로 새 토큰 초기화

    새 토큰 "안녕하세요"가 base tokenizer로 ["안", "녕", "하", "세요"]로 분해되면,
    이 subword들의 임베딩 평균으로 초기화
    """
    print("=" * 70)
    print("Subword Averaging Token Initialization")
    print("=" * 70)

    # Load base model and tokenizer
    print(f"\n1. Loading base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Full precision for initialization
    )

    # Load expanded tokenizer
    print(f"2. Loading expanded tokenizer: {expanded_tokenizer_path}")
    expanded_tokenizer = AutoTokenizer.from_pretrained(expanded_tokenizer_path, trust_remote_code=True)

    new_vocab_size = len(expanded_tokenizer)
    num_new_tokens = new_vocab_size - original_vocab_size

    print(f"\n   Original vocab: {original_vocab_size:,}")
    print(f"   Expanded vocab: {new_vocab_size:,}")
    print(f"   New tokens: {num_new_tokens:,}")

    # Resize model embeddings
    print(f"\n3. Resizing model embeddings...")
    base_model.resize_token_embeddings(new_vocab_size)

    # Get embedding layer
    embed_tokens = base_model.get_input_embeddings()
    embed_weight = embed_tokens.weight.data

    # Get original embeddings (for averaging)
    original_embeddings = embed_weight[:original_vocab_size].clone()

    # Initialize new tokens with subword averaging
    print(f"\n4. Initializing new tokens with subword averaging...")

    initialized_count = 0
    fallback_count = 0

    for new_token_id in tqdm(range(original_vocab_size, new_vocab_size), desc="Initializing"):
        # Get the new token string
        new_token = expanded_tokenizer.convert_ids_to_tokens(new_token_id)

        if new_token is None:
            continue

        # Remove special prefixes (▁, Ġ, etc.)
        clean_token = new_token.replace('▁', '').replace('Ġ', '').strip()

        if not clean_token:
            # Empty token - use mean of all embeddings
            embed_weight[new_token_id] = original_embeddings.mean(dim=0)
            fallback_count += 1
            continue

        # Tokenize with base tokenizer to get subwords
        subword_ids = base_tokenizer.encode(clean_token, add_special_tokens=False)

        if len(subword_ids) == 0:
            # Can't tokenize - use mean
            embed_weight[new_token_id] = original_embeddings.mean(dim=0)
            fallback_count += 1
            continue

        # Filter valid subword IDs (within original vocab)
        valid_ids = [sid for sid in subword_ids if sid < original_vocab_size]

        if len(valid_ids) == 0:
            embed_weight[new_token_id] = original_embeddings.mean(dim=0)
            fallback_count += 1
            continue

        # Average the subword embeddings
        subword_embeddings = original_embeddings[valid_ids]
        avg_embedding = subword_embeddings.mean(dim=0)

        embed_weight[new_token_id] = avg_embedding
        initialized_count += 1

    print(f"\n   Initialized with subword avg: {initialized_count:,}")
    print(f"   Fallback to mean: {fallback_count:,}")

    # Verify embedding statistics
    print(f"\n5. Verifying embedding statistics...")

    with torch.no_grad():
        old_norms = torch.norm(embed_weight[:original_vocab_size], dim=1)
        new_norms = torch.norm(embed_weight[original_vocab_size:], dim=1)

        print(f"   Old tokens - Mean norm: {old_norms.mean():.4f}, Std: {old_norms.std():.4f}")
        print(f"   New tokens - Mean norm: {new_norms.mean():.4f}, Std: {new_norms.std():.4f}")

        # Check similarity distribution
        sample_old = embed_weight[:1000]
        sample_new = embed_weight[original_vocab_size:original_vocab_size+1000]

        old_sim = torch.mm(
            torch.nn.functional.normalize(sample_old, dim=1),
            torch.nn.functional.normalize(sample_old, dim=1).t()
        )
        new_sim = torch.mm(
            torch.nn.functional.normalize(sample_new, dim=1),
            torch.nn.functional.normalize(sample_old, dim=1).t()
        )

        print(f"   Old-Old similarity: {old_sim.mean():.4f}")
        print(f"   New-Old similarity: {new_sim.mean():.4f}")

    # Save model
    print(f"\n6. Saving initialized model to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model.save_pretrained(output_dir)
    expanded_tokenizer.save_pretrained(output_dir)

    # Save initialization info
    import json
    info = {
        "base_model": base_model_name,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "initialized_with_subword_avg": initialized_count,
        "fallback_to_mean": fallback_count,
        "method": "subword_averaging"
    }
    with open(output_dir / "initialization_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n" + "=" * 70)
    print("Initialization complete!")
    print("=" * 70)
    print(f"\nNext step: Use '{output_path}' as starting point for training")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Initialize new tokens with subword averaging")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Base model name"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="outputs/koqwen-expanded",
        help="Path to expanded tokenizer"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/initialized_model",
        help="Output path for initialized model"
    )
    parser.add_argument(
        "--original_vocab_size",
        type=int,
        default=151669,
        help="Original vocabulary size"
    )

    args = parser.parse_args()

    initialize_with_subword_average(
        base_model_name=args.base_model,
        expanded_tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        original_vocab_size=args.original_vocab_size
    )


if __name__ == "__main__":
    main()
