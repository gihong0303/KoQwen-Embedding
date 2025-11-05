"""
Model utilities for loading, expanding, and managing models
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_tokenizers(kormo_model: str, qwen_model: str, use_fast: bool = False) -> Tuple:
    """
    KORMo와 Qwen 토크나이저 로드

    Args:
        kormo_model: KORMo 모델 경로
        qwen_model: Qwen 모델 경로
        use_fast: Fast tokenizer 사용 여부

    Returns:
        (kormo_tokenizer, qwen_tokenizer)
    """
    logger.info("=" * 80)
    logger.info("토크나이저 로딩")
    logger.info("=" * 80)

    # KORMo
    logger.info(f"KORMo 토크나이저: {kormo_model}")
    kormo_tokenizer = AutoTokenizer.from_pretrained(
        kormo_model,
        trust_remote_code=True,
        use_fast=use_fast
    )
    logger.info(f"✓ KORMo vocab size: {len(kormo_tokenizer):,}")

    # Qwen
    logger.info(f"Qwen 토크나이저: {qwen_model}")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_model,
        trust_remote_code=True,
        use_fast=use_fast
    )
    logger.info(f"✓ Qwen vocab size: {len(qwen_tokenizer):,}")
    logger.info("")

    return kormo_tokenizer, qwen_tokenizer


def load_model(
    model_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> nn.Module:
    """
    모델 로드

    Args:
        model_path: 모델 경로
        torch_dtype: 데이터 타입
        device_map: 디바이스 맵 ("auto", "cpu", "cuda", None)

    Returns:
        model
    """
    logger.info("=" * 80)
    logger.info("모델 로딩")
    logger.info("=" * 80)
    logger.info(f"경로: {model_path}")
    logger.info(f"dtype: {torch_dtype}")

    # dtype 매핑
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # 모델 로드 (accelerate 없을 경우 대비)
    # config를 명시하지 않으면 checkpoint의 실제 크기 사용
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if device_map != "auto" else None
        )
    except (ValueError, ImportError):
        # accelerate 없을 때: device_map 없이 로드
        logger.warning("accelerate 없음 - CPU로 로드 후 수동으로 GPU 이동")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        # GPU 사용 가능하면 이동
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✓ 모델을 GPU로 이동")

    # use_cache 비활성화 (학습용)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    logger.info(f"✓ 모델 로드 완료")
    logger.info("")

    return model


def freeze_model_params(model: nn.Module, trainable_params: List[str]) -> int:
    """
    모델 파라미터 프리징 (trainable_params만 학습)

    Args:
        model: 모델
        trainable_params: 학습할 파라미터 이름 리스트

    Returns:
        학습 가능한 파라미터 수
    """
    logger.info("=" * 80)
    logger.info("모델 파라미터 프리징")
    logger.info("=" * 80)

    # 전체 freeze
    for param in model.parameters():
        param.requires_grad = False

    # trainable_params만 unfreeze
    trainable_count = 0
    for name, param in model.named_parameters():
        for trainable_name in trainable_params:
            if trainable_name in name:
                param.requires_grad = True
                trainable_count += param.numel()
                logger.info(f"✓ Trainable: {name} ({param.numel():,} params)")
                break

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n총 파라미터: {total_params:,}")
    logger.info(f"학습 파라미터: {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
    logger.info("")

    return trainable_count


def expand_embeddings(
    model: nn.Module,
    qwen_tokenizer,
    kormo_tokenizer,
    vocab_diff: List[str],
    init_method: str = "mean"
) -> nn.Module:
    """
    임베딩 확장 (차집합 추가 방식)

    Args:
        model: Qwen 모델
        qwen_tokenizer: Qwen 토크나이저
        kormo_tokenizer: KORMo 토크나이저
        vocab_diff: 추가할 토큰 리스트
        init_method: 초기화 방법 ("mean", "random")

    Returns:
        확장된 모델
    """
    logger.info("=" * 80)
    logger.info("임베딩 확장 (차집합 추가 방식)")
    logger.info("=" * 80)

    old_embed = model.get_input_embeddings()
    old_vocab_size = old_embed.num_embeddings
    new_vocab_size = old_vocab_size + len(vocab_diff)

    logger.info(f"기존 vocab: {old_vocab_size:,}")
    logger.info(f"추가 토큰: {len(vocab_diff):,}")
    logger.info(f"새 vocab: {new_vocab_size:,}")
    logger.info(f"초기화 방법: {init_method}")
    logger.info("")

    # 새 임베딩 생성
    new_embed = nn.Embedding(
        new_vocab_size,
        old_embed.embedding_dim,
        device=old_embed.weight.device,
        dtype=old_embed.weight.dtype
    )

    # 기존 임베딩 복사
    with torch.no_grad():
        new_embed.weight[:old_vocab_size] = old_embed.weight.clone()

    # 새 토큰 초기화
    logger.info("새 토큰 임베딩 초기화...")
    with torch.no_grad():
        for i, token in enumerate(tqdm(vocab_diff, desc="초기화")):
            new_idx = old_vocab_size + i

            if init_method == "mean":
                # Qwen 토크나이저로 분해
                subtoken_ids = qwen_tokenizer.encode(token, add_special_tokens=False)

                if len(subtoken_ids) > 0:
                    # 서브토큰 임베딩의 평균
                    subtoken_embeds = old_embed.weight[subtoken_ids]
                    new_embed.weight[new_idx] = subtoken_embeds.mean(dim=0)
                else:
                    # Fallback: 랜덤 초기화
                    torch.nn.init.normal_(new_embed.weight[new_idx:new_idx+1], mean=0.0, std=0.02)

            elif init_method == "random":
                torch.nn.init.normal_(new_embed.weight[new_idx:new_idx+1], mean=0.0, std=0.02)

    # 모델에 적용
    model.set_input_embeddings(new_embed)

    # lm_head도 확장 (있는 경우)
    if hasattr(model, 'lm_head'):
        old_lm_head = model.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features,
            new_vocab_size,
            bias=old_lm_head.bias is not None,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype
        )

        with torch.no_grad():
            new_lm_head.weight[:old_vocab_size] = old_lm_head.weight.clone()
            if old_lm_head.bias is not None:
                new_lm_head.bias[:old_vocab_size] = old_lm_head.bias.clone()

            # 새 토큰 초기화 (임베딩과 동일하게)
            for i, token in enumerate(vocab_diff):
                new_idx = old_vocab_size + i
                subtoken_ids = qwen_tokenizer.encode(token, add_special_tokens=False)

                if len(subtoken_ids) > 0:
                    subtoken_weights = old_lm_head.weight[subtoken_ids]
                    new_lm_head.weight[new_idx] = subtoken_weights.mean(dim=0)
                else:
                    torch.nn.init.normal_(new_lm_head.weight[new_idx:new_idx+1], mean=0.0, std=0.02)

        model.lm_head = new_lm_head
        logger.info("✓ lm_head도 확장됨")

    logger.info("✓ 임베딩 확장 완료")
    logger.info("")

    return model


def save_model(model: nn.Module, tokenizer, output_dir: str):
    """
    모델과 토크나이저 저장

    Args:
        model: 모델
        tokenizer: 토크나이저
        output_dir: 출력 디렉토리
    """
    logger.info("=" * 80)
    logger.info("모델 저장")
    logger.info("=" * 80)
    logger.info(f"경로: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)
    logger.info("✓ 토크나이저 저장 완료")

    # 모델 저장
    model.save_pretrained(output_path)
    logger.info("✓ 모델 저장 완료")
    logger.info("")


def create_expanded_tokenizer(qwen_tokenizer, vocab_diff: List[str], output_dir: str):
    """
    확장된 토크나이저 생성 및 저장

    Args:
        qwen_tokenizer: Qwen 토크나이저
        vocab_diff: 추가할 토큰 리스트
        output_dir: 출력 디렉토리

    Returns:
        확장된 토크나이저
    """
    logger.info("=" * 80)
    logger.info("토크나이저 확장")
    logger.info("=" * 80)
    logger.info(f"추가 토큰 수: {len(vocab_diff):,}")

    # 토큰 추가
    num_added = qwen_tokenizer.add_tokens(vocab_diff)
    logger.info(f"✓ {num_added:,}개 토큰 추가됨")
    logger.info(f"✓ 새 vocab size: {len(qwen_tokenizer):,}")

    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    qwen_tokenizer.save_pretrained(output_path)
    logger.info(f"✓ 토크나이저 저장: {output_path}")
    logger.info("")

    return qwen_tokenizer
