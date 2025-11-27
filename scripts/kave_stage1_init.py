#!/usr/bin/env python3
"""
KAVE Stage 1: WSA Token Initialization

Weighted Semantic Averaging으로 새 토큰 초기화
- 단순 평균 대신 semantic similarity 가중 평균 사용
- 기존 EEVE 방식보다 더 좋은 초기화

Usage:
    python scripts/kave_stage1_init.py --config configs/kave_pipeline_config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModel


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_semantic_weights(
    subword_embeddings: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute semantic similarity weights for subwords

    WSA (Weighted Semantic Averaging) 핵심 로직:
    - subword들 간의 pairwise similarity 계산
    - 다른 subword들과 관련성이 높은 subword에 더 높은 가중치
    """
    if len(subword_embeddings) == 1:
        return torch.ones(1, device=subword_embeddings.device)

    # Normalize
    subword_norm = F.normalize(subword_embeddings, dim=1)

    # Pairwise similarity
    pairwise_sim = torch.mm(subword_norm, subword_norm.t())

    # Self-similarity 제외한 평균
    mask = ~torch.eye(len(subword_embeddings), dtype=torch.bool, device=subword_embeddings.device)
    mean_sim = (pairwise_sim * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Softmax로 가중치 계산
    weights = F.softmax(mean_sim / temperature, dim=0)

    return weights


def initialize_with_wsa(
    base_model_name: str,
    expanded_tokenizer_path: str,
    output_path: str,
    original_vocab_size: int = 151669,
    wsa_temperature: float = 0.1,
    use_semantic_weights: bool = True
):
    """
    WSA (Weighted Semantic Averaging)로 새 토큰 초기화

    기존 EEVE의 simple average 대신:
    - subword들 간 semantic similarity 계산
    - 관련성 높은 subword에 더 높은 가중치
    """
    print("=" * 70)
    print("KAVE Stage 1: WSA Token Initialization")
    print("=" * 70)

    # Load models
    print(f"\n1. Loading base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    print(f"2. Loading expanded tokenizer: {expanded_tokenizer_path}")
    expanded_tokenizer = AutoTokenizer.from_pretrained(expanded_tokenizer_path, trust_remote_code=True)

    new_vocab_size = len(expanded_tokenizer)
    num_new_tokens = new_vocab_size - original_vocab_size

    print(f"\n   Original vocab: {original_vocab_size:,}")
    print(f"   Expanded vocab: {new_vocab_size:,}")
    print(f"   New tokens: {num_new_tokens:,}")

    # Resize embeddings
    print(f"\n3. Resizing model embeddings...")
    base_model.resize_token_embeddings(new_vocab_size)

    embed_tokens = base_model.get_input_embeddings()
    embed_weight = embed_tokens.weight.data
    original_embeddings = embed_weight[:original_vocab_size].clone()

    # Initialize with WSA
    print(f"\n4. Initializing with WSA (temperature={wsa_temperature})...")

    stats = {
        'wsa_initialized': 0,
        'single_subword': 0,
        'fallback_mean': 0,
        'total_weight_variance': []
    }

    for new_token_id in tqdm(range(original_vocab_size, new_vocab_size), desc="WSA Init"):
        token = expanded_tokenizer.convert_ids_to_tokens(new_token_id)
        if token is None:
            continue

        # Clean token
        clean_token = token.replace('▁', '').replace('Ġ', '').strip()
        if not clean_token:
            embed_weight[new_token_id] = original_embeddings.mean(dim=0)
            stats['fallback_mean'] += 1
            continue

        # Get subword IDs
        subword_ids = base_tokenizer.encode(clean_token, add_special_tokens=False)
        valid_ids = [sid for sid in subword_ids if sid < original_vocab_size]

        if len(valid_ids) == 0:
            embed_weight[new_token_id] = original_embeddings.mean(dim=0)
            stats['fallback_mean'] += 1
            continue

        if len(valid_ids) == 1:
            # Single subword - just copy
            embed_weight[new_token_id] = original_embeddings[valid_ids[0]].clone()
            stats['single_subword'] += 1
            continue

        # Get subword embeddings
        subword_embeddings = original_embeddings[valid_ids]

        if use_semantic_weights:
            # WSA: Weighted Semantic Averaging
            weights = compute_semantic_weights(subword_embeddings, wsa_temperature)
            new_embedding = (subword_embeddings * weights.unsqueeze(1)).sum(dim=0)
            stats['total_weight_variance'].append(weights.var().item())
        else:
            # Simple averaging (EEVE style)
            new_embedding = subword_embeddings.mean(dim=0)

        embed_weight[new_token_id] = new_embedding
        stats['wsa_initialized'] += 1

    print(f"\n   WSA initialized: {stats['wsa_initialized']:,}")
    print(f"   Single subword: {stats['single_subword']:,}")
    print(f"   Fallback to mean: {stats['fallback_mean']:,}")

    if stats['total_weight_variance']:
        avg_var = sum(stats['total_weight_variance']) / len(stats['total_weight_variance'])
        print(f"   Average weight variance: {avg_var:.6f}")

    # Verify statistics
    print(f"\n5. Verifying embedding statistics...")

    with torch.no_grad():
        old_norms = torch.norm(embed_weight[:original_vocab_size], dim=1)
        new_norms = torch.norm(embed_weight[original_vocab_size:], dim=1)

        print(f"   Old tokens - Mean norm: {old_norms.mean():.4f}, Std: {old_norms.std():.4f}")
        print(f"   New tokens - Mean norm: {new_norms.mean():.4f}, Std: {new_norms.std():.4f}")

        # Cosine similarity distribution
        sample_old = F.normalize(embed_weight[:1000], dim=1)
        sample_new = F.normalize(embed_weight[original_vocab_size:original_vocab_size+1000], dim=1)

        new_old_sim = torch.mm(sample_new, sample_old.t())
        print(f"   New-Old similarity: mean={new_old_sim.mean():.4f}, max={new_old_sim.max():.4f}")

    # Save
    print(f"\n6. Saving initialized model to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model.save_pretrained(output_dir)
    expanded_tokenizer.save_pretrained(output_dir)

    # Save initialization info
    info = {
        "method": "WSA (Weighted Semantic Averaging)",
        "base_model": base_model_name,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "wsa_temperature": wsa_temperature,
        "use_semantic_weights": use_semantic_weights,
        "stats": {
            "wsa_initialized": stats['wsa_initialized'],
            "single_subword": stats['single_subword'],
            "fallback_mean": stats['fallback_mean']
        }
    }
    with open(output_dir / "wsa_init_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n" + "=" * 70)
    print("WSA Initialization Complete!")
    print("=" * 70)
    print(f"\nNext: Run KAVE Stage 2 with '{output_path}' as starting point")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="KAVE Stage 1: WSA Initialization")
    parser.add_argument("--config", type=str, default="configs/kave_pipeline_config.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wsa_temperature", type=float, default=None)
    parser.add_argument("--no_semantic_weights", action="store_true")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    base_model = args.base_model or config['model']['base_model']
    tokenizer_path = args.tokenizer_path or config['model']['tokenizer_path']
    output_path = args.output_path or config['model']['initialized_model_path']
    wsa_temperature = args.wsa_temperature or config['kave']['wsa']['temperature']
    use_semantic_weights = not args.no_semantic_weights and config['kave']['wsa']['use_semantic_weights']

    initialize_with_wsa(
        base_model_name=base_model,
        expanded_tokenizer_path=tokenizer_path,
        output_path=output_path,
        original_vocab_size=config['model']['original_vocab_size'],
        wsa_temperature=wsa_temperature,
        use_semantic_weights=use_semantic_weights
    )


if __name__ == "__main__":
    main()
