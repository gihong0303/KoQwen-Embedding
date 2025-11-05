#!/usr/bin/env python3
"""
EEVE Subword-based Initialization for Embedding Models

Qwen3-Embedding 모델용 초기화
- Input embedding: subtoken 평균
- Output embedding 없음 (임베딩 모델)
"""

import torch
import torch.nn as nn
from typing import Set
from transformers import PreTrainedTokenizer, PreTrainedModel
import logging

logger = logging.getLogger(__name__)


def initialize_new_embeddings_eeve_style(
    model: PreTrainedModel,
    old_tokenizer: PreTrainedTokenizer,
    new_tokenizer: PreTrainedTokenizer,
) -> PreTrainedModel:
    """
    EEVE 방식 초기화 (임베딩 모델용)

    새 토큰 embedding = subtoken embeddings의 평균
    """
    logger.info("=" * 80)
    logger.info("EEVE-style Subword Initialization (Embedding Model)")
    logger.info("=" * 80)

    old_vocab = set(old_tokenizer.get_vocab().keys())
    new_vocab = set(new_tokenizer.get_vocab().keys())
    added_tokens = new_vocab - old_vocab

    logger.info(f"Old vocabulary: {len(old_vocab):,}")
    logger.info(f"New vocabulary: {len(new_vocab):,}")
    logger.info(f"Added tokens: {len(added_tokens):,}")

    if len(added_tokens) == 0:
        logger.warning("No new tokens to initialize!")
        return model

    # Embeddings 가져오기
    embed_tokens = model.get_input_embeddings()

    if embed_tokens is None:
        logger.error("Cannot find input embeddings!")
        return model

    initialized_count = 0
    failed_count = 0

    logger.info("\nInitializing new tokens...")

    for idx, token in enumerate(added_tokens):
        try:
            new_token_id = new_tokenizer.convert_tokens_to_ids(token)

            # Old tokenizer로 subword 분해
            subtokens = old_tokenizer.tokenize(token)

            if not subtokens:
                # Fallback: random initialization은 이미 되어 있음
                failed_count += 1
                continue

            subtoken_ids = old_tokenizer.convert_tokens_to_ids(subtokens)

            # Subtoken embedding 평균
            with torch.no_grad():
                subtoken_embeds = [embed_tokens.weight[sid] for sid in subtoken_ids]
                avg_embed = torch.stack(subtoken_embeds).mean(dim=0)
                embed_tokens.weight[new_token_id] = avg_embed

            initialized_count += 1

            if (idx + 1) % 5000 == 0:
                logger.info(f"  Progress: {idx+1}/{len(added_tokens)} tokens...")

        except Exception as e:
            failed_count += 1
            continue

    logger.info(f"\n" + "=" * 80)
    logger.info(f"Initialization Results:")
    logger.info(f"  ✓ Successfully initialized: {initialized_count:,}")
    logger.info(f"  ✗ Failed (using random): {failed_count:,}")
    logger.info(f"  Total: {len(added_tokens):,}")
    logger.info("=" * 80 + "\n")

    return model


def get_new_token_mask(
    tokenizer: PreTrainedTokenizer,
    old_vocab_size: int
) -> torch.Tensor:
    """
    새 토큰 마스크 생성

    Returns:
        Boolean tensor [vocab_size], True = 새 토큰
    """
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    if vocab_size > old_vocab_size:
        mask[old_vocab_size:] = True

    return mask


def create_gradient_mask_hook(new_token_mask: torch.Tensor):
    """
    Gradient masking hook 생성

    기존 토큰의 gradient를 0으로 만듦
    """
    def hook(grad):
        """기존 토큰 gradient 제거"""
        if grad is None:
            return None

        masked_grad = grad.clone()
        masked_grad[~new_token_mask] = 0.0
        return masked_grad

    return hook
