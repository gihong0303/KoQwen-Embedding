"""
Hybrid Loss Functions for Korean Embedding Enhancement

통합 손실 함수 모듈:
- Stage 0: CLSA Loss (Cross-lingual Semantic Anchoring)
- Stage 1-3: JLCE + MCL Loss (Jamo + Morphological Curriculum)
- Stage 4-6: Contrastive Loss (SimCSE style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Stage 0: CLSA Loss (Cross-lingual Semantic Anchoring)
# ============================================================================

class CLSALoss(nn.Module):
    """
    Cross-lingual Semantic Anchoring Loss

    새 한국어 토큰을 영어/중국어 의미 공간에 정렬

    Loss = alignment_loss + λ * diversity_loss

    alignment_loss: 한국어 토큰과 anchor center 간 거리
    diversity_loss: 토큰 임베딩 collapse 방지
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        distance_type: str = 'cosine'  # 'cosine' or 'euclidean'
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.distance_type = distance_type

    def forward(
        self,
        korean_embeddings: torch.Tensor,
        anchor_embeddings: torch.Tensor,
        anchor_weights: torch.Tensor,
        anchor_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            korean_embeddings: 한국어 토큰 임베딩 [batch_size, hidden_dim]
            anchor_embeddings: 앵커 임베딩 [batch_size, max_anchors, hidden_dim]
            anchor_weights: 앵커 가중치 [batch_size, max_anchors]
            anchor_masks: 유효 앵커 마스크 [batch_size, max_anchors]

        Returns:
            loss: 총 손실
            metrics: 개별 손실 값
        """
        batch_size = korean_embeddings.shape[0]

        # Normalize Korean embeddings
        korean_emb_norm = F.normalize(korean_embeddings, p=2, dim=1)

        # Normalize anchor embeddings
        anchor_emb_norm = F.normalize(anchor_embeddings, p=2, dim=2)

        # Masked weighted anchor center
        weights_masked = anchor_weights * anchor_masks.float()
        weights_sum = weights_masked.sum(dim=1, keepdim=True).clamp(min=1e-9)
        weights_normalized = weights_masked / weights_sum  # [B, A]

        # Weighted average of anchors
        anchor_center = torch.bmm(
            weights_normalized.unsqueeze(1),  # [B, 1, A]
            anchor_emb_norm  # [B, A, H]
        ).squeeze(1)  # [B, H]

        anchor_center = F.normalize(anchor_center, p=2, dim=1)

        # 1. Alignment Loss
        if self.distance_type == 'cosine':
            # Cosine distance (1 - similarity)
            alignment_loss = 1.0 - (korean_emb_norm * anchor_center).sum(dim=1)
        else:
            # Euclidean distance
            alignment_loss = torch.norm(korean_emb_norm - anchor_center, p=2, dim=1)

        alignment_loss = alignment_loss.mean()

        # 2. Diversity Loss (prevent token collapse)
        # Compute pairwise similarity within batch
        similarity_matrix = torch.mm(korean_emb_norm, korean_emb_norm.t())

        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        diversity_loss = similarity_matrix[mask].abs().mean()

        # Total loss
        total_loss = alignment_loss + self.diversity_weight * diversity_loss

        metrics = {
            'alignment_loss': alignment_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


# ============================================================================
# Stage 1-3: JLCE + MCL Combined Loss
# ============================================================================

class JLCEMCLLoss(nn.Module):
    """
    Jamo-Level Compositional Embedding + Morphological Curriculum Learning Loss

    3개 손실 함수 결합:
    1. Jamo Composition Loss: 자모 기반 임베딩과 타겟 임베딩 정렬
    2. Morphological Curriculum Loss: 형태소 복잡도 기반 가중치
    3. Contrastive Loss: SimCSE 스타일 대조 학습

    각 Stage별 가중치:
    - Stage 1 (Easy): jamo_weight=0.5, curriculum_weight=0.3
    - Stage 2 (Medium): jamo_weight=0.4, curriculum_weight=0.4
    - Stage 3 (Hard): jamo_weight=0.3, curriculum_weight=0.5
    """

    def __init__(
        self,
        stage: int = 1,
        temperature: float = 0.05,
        jamo_weight: float = 0.4,
        curriculum_weight: float = 0.3,
        contrastive_weight: float = 0.3
    ):
        super().__init__()
        self.stage = stage
        self.temperature = temperature

        # Stage별 가중치 자동 조정
        if stage == 1:  # Easy
            self.jamo_weight = 0.5
            self.curriculum_weight = 0.2
            self.contrastive_weight = 0.3
        elif stage == 2:  # Medium
            self.jamo_weight = 0.4
            self.curriculum_weight = 0.3
            self.contrastive_weight = 0.3
        else:  # Hard
            self.jamo_weight = 0.3
            self.curriculum_weight = 0.4
            self.contrastive_weight = 0.3

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        jamo_embeddings: Optional[torch.Tensor] = None,
        token_categories: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings1: 첫 번째 forward pass 임베딩 [batch_size, hidden_dim]
            embeddings2: 두 번째 forward pass 임베딩 [batch_size, hidden_dim]
            jamo_embeddings: JLCE로 생성된 임베딩 [batch_size, hidden_dim] (선택)
            token_categories: 토큰 카테고리 리스트 ['easy', 'medium', 'hard'] (선택)

        Returns:
            loss: 총 손실
            metrics: 개별 손실 값
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device

        # Normalize
        emb1_norm = F.normalize(embeddings1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings2, p=2, dim=1)

        # 1. Contrastive Loss (SimCSE)
        sim_matrix = torch.mm(emb1_norm, emb2_norm.t()) / self.temperature
        labels = torch.arange(batch_size, device=device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)

        # 2. Jamo Composition Loss
        jamo_loss = torch.tensor(0.0, device=device)
        if jamo_embeddings is not None:
            jamo_norm = F.normalize(jamo_embeddings, p=2, dim=1)
            # 자모 임베딩과 실제 임베딩 정렬
            jamo_loss = 1.0 - (jamo_norm * emb1_norm).sum(dim=1).mean()

        # 3. Curriculum Loss (형태소 복잡도 기반 가중치)
        curriculum_loss = torch.tensor(0.0, device=device)
        if token_categories is not None:
            stage_category = {1: 'easy', 2: 'medium', 3: 'hard'}
            target_category = stage_category.get(self.stage, 'easy')

            # 타겟 카테고리 가중치
            weights = torch.ones(batch_size, device=device)
            for i, cat in enumerate(token_categories):
                if cat == target_category:
                    weights[i] = 2.0  # 타겟 카테고리에 2배 가중치

            # 가중 대조 손실
            weighted_ce = F.cross_entropy(sim_matrix, labels, reduction='none')
            curriculum_loss = (weights * weighted_ce).mean() - contrastive_loss  # 추가 분만

        # Total Loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.jamo_weight * jamo_loss +
            self.curriculum_weight * curriculum_loss
        )

        metrics = {
            'contrastive_loss': contrastive_loss.item(),
            'jamo_loss': jamo_loss.item(),
            'curriculum_loss': curriculum_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


# ============================================================================
# Stage 4-6: Standard Contrastive Loss
# ============================================================================

class StandardContrastiveLoss(nn.Module):
    """
    표준 SimCSE 스타일 대조 손실

    Stage 4: Full vocabulary harmonization
    Stage 5-6: LoRA fine-tuning
    """

    def __init__(
        self,
        temperature: float = 0.05,
        hard_negative_weight: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings1: 첫 번째 forward pass [batch_size, hidden_dim]
            embeddings2: 두 번째 forward pass [batch_size, hidden_dim]
            hard_negatives: Hard negative 임베딩 [batch_size, hidden_dim] (선택)

        Returns:
            loss: 총 손실
            metrics: 손실 값
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device

        # Normalize
        emb1_norm = F.normalize(embeddings1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings2, p=2, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(emb1_norm, emb2_norm.t()) / self.temperature

        # Labels
        labels = torch.arange(batch_size, device=device)

        # Base contrastive loss
        base_loss = F.cross_entropy(sim_matrix, labels)

        # Hard negative loss (optional)
        hn_loss = torch.tensor(0.0, device=device)
        if hard_negatives is not None and self.hard_negative_weight > 0:
            hn_norm = F.normalize(hard_negatives, p=2, dim=1)
            # Hard negative와의 유사도를 낮춰야 함
            hn_sim = (emb1_norm * hn_norm).sum(dim=1)
            hn_loss = F.relu(hn_sim + 0.5).mean()  # Margin = 0.5

        total_loss = base_loss + self.hard_negative_weight * hn_loss

        metrics = {
            'base_loss': base_loss.item(),
            'hard_negative_loss': hn_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


# ============================================================================
# Unified Loss Factory
# ============================================================================

def get_loss_function(stage: int, config: dict = None) -> nn.Module:
    """
    Stage별 손실 함수 팩토리

    Args:
        stage: 학습 단계 (0-6)
        config: 손실 함수 설정

    Returns:
        손실 함수 모듈
    """
    config = config or {}

    if stage == 0:
        # CLSA
        return CLSALoss(
            diversity_weight=config.get('diversity_weight', 0.1),
            distance_type=config.get('distance_type', 'cosine')
        )

    elif stage in [1, 2, 3]:
        # JLCE + MCL
        return JLCEMCLLoss(
            stage=stage,
            temperature=config.get('temperature', 0.05),
            jamo_weight=config.get('jamo_weight', 0.4),
            curriculum_weight=config.get('curriculum_weight', 0.3),
            contrastive_weight=config.get('contrastive_weight', 0.3)
        )

    else:
        # Stage 4-6: Standard contrastive
        return StandardContrastiveLoss(
            temperature=config.get('temperature', 0.05),
            hard_negative_weight=config.get('hard_negative_weight', 0.0)
        )


# ============================================================================
# Optimization Configuration
# ============================================================================

def get_optimizer_config(stage: int) -> Dict:
    """
    Stage별 최적화 설정

    Args:
        stage: 학습 단계 (0-6)

    Returns:
        최적화 설정 딕셔너리
    """
    configs = {
        0: {  # CLSA - 높은 LR로 빠른 초기화
            'optimizer': 'AdamW',
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'warmup_ratio': 0.2,
            'lr_scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
        1: {  # Easy tokens - 빠른 학습
            'optimizer': 'AdamW',
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'lr_scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
        2: {  # Medium tokens - 안정적 학습
            'optimizer': 'AdamW',
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'lr_scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
        3: {  # Hard tokens - 조심스러운 학습
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_ratio': 0.15,
            'lr_scheduler': 'cosine_with_restarts',
            'max_grad_norm': 0.5,  # 더 작은 gradient clipping
            'betas': (0.9, 0.999),
        },
        4: {  # Full vocab harmonization
            'optimizer': 'AdamW',
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'lr_scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
        5: {  # LoRA fine-tuning
            'optimizer': 'AdamW',
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.05,
            'lr_scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
        6: {  # Advanced contrastive
            'optimizer': 'AdamW',
            'learning_rate': 3e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'lr_scheduler': 'linear',
            'max_grad_norm': 1.0,
            'betas': (0.9, 0.999),
        },
    }

    return configs.get(stage, configs[1])


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_dataset_config(stage: int) -> Dict:
    """
    Stage별 데이터셋 설정

    Args:
        stage: 학습 단계 (0-6)

    Returns:
        데이터셋 설정 딕셔너리
    """
    configs = {
        0: {  # CLSA
            'name': 'bilingual_dictionary',
            'source': 'generated',
            'path': 'outputs/bilingual_dictionary.json',
            'description': 'Korean-English-Chinese token mappings',
        },
        1: {  # Easy tokens
            'datasets': [
                {'name': 'HAERAE-HUB/KOREAN-WEBTEXT', 'source': 'huggingface', 'samples': 150000},
                {'name': 'kakaobrain/kor_nli', 'source': 'huggingface', 'samples': 100000},
            ],
            'curriculum_mode': 'easy',
            'total_samples': 250000,
        },
        2: {  # Medium tokens
            'datasets': [
                {'name': 'HAERAE-HUB/KOREAN-WEBTEXT', 'source': 'huggingface', 'samples': 150000},
                {'name': 'kakaobrain/kor_nli', 'source': 'huggingface', 'samples': 100000},
            ],
            'curriculum_mode': 'medium',
            'total_samples': 250000,
        },
        3: {  # Hard tokens
            'datasets': [
                {'name': 'HAERAE-HUB/KOREAN-SyntheticText', 'source': 'local', 'samples': 100000},
                {'name': 'kakaobrain/kor_nli', 'source': 'huggingface', 'samples': 50000},
            ],
            'curriculum_mode': 'hard',
            'total_samples': 150000,
        },
        4: {  # Full vocabulary
            'datasets': [
                {'name': 'HAERAE-HUB/KOREAN-WEBTEXT', 'source': 'local', 'samples': 80000},
                {'name': 'HAERAE-HUB/KOREAN-SyntheticText', 'source': 'local', 'samples': 50000},
                {'name': 'HAERAE-HUB/KoSimpleEval', 'source': 'local', 'samples': 20000},
                {'name': 'kakaobrain/kor_nli', 'source': 'huggingface', 'samples': 50000},
            ],
            'curriculum_mode': None,
            'total_samples': 200000,
        },
        5: {  # LoRA (Reasoning)
            'datasets': [
                {'name': 'HAE-RAE-COT', 'source': 'local', 'samples': 100000},
                {'name': 'HR-Instruct-Math', 'source': 'local', 'samples': 100000},
            ],
            'curriculum_mode': None,
            'total_samples': 200000,
        },
        6: {  # Advanced contrastive
            'datasets': [
                {'name': 'K2-Feedback', 'source': 'local', 'samples': 150000, 'min_score': 5},
            ],
            'curriculum_mode': None,
            'total_samples': 150000,
        },
    }

    return configs.get(stage, configs[1])
