"""
KURE Core Components

Multi-granularity Representation (MGR) + Adaptive Loss Balancing (ALB) + Validation-guided Training (VGT)

KURE 핵심 구성 요소 통합 모듈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# ============================================================================
# Multi-granularity Representation (MGR) - Matryoshka Style
# ============================================================================

class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning

    다양한 차원에서 유효한 임베딩을 학습하여
    효율성과 성능의 trade-off 제공

    Reference: Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022
    """

    # 학습할 차원들 (전체 1536 기준)
    DEFAULT_DIMENSIONS = [1536, 768, 384, 192, 96]

    def __init__(
        self,
        full_dim: int = 1536,
        dimensions: List[int] = None,
        temperature: float = 0.05,
        dimension_weights: Optional[Dict[int, float]] = None
    ):
        """
        Args:
            full_dim: 전체 임베딩 차원
            dimensions: 학습할 차원 목록
            temperature: Contrastive loss temperature
            dimension_weights: 차원별 가중치 (없으면 자동 계산)
        """
        super().__init__()
        self.full_dim = full_dim
        self.dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self.temperature = temperature

        # 차원별 가중치 (큰 차원일수록 가중치 높게)
        if dimension_weights:
            self.dimension_weights = dimension_weights
        else:
            self.dimension_weights = {
                dim: dim / self.dimensions[0]
                for dim in self.dimensions
            }

        # 정규화
        total_weight = sum(self.dimension_weights.values())
        self.dimension_weights = {
            dim: w / total_weight
            for dim, w in self.dimension_weights.items()
        }

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Matryoshka contrastive loss

        Args:
            embeddings1: [batch_size, full_dim]
            embeddings2: [batch_size, full_dim]

        Returns:
            total_loss: 가중 합산 손실
            loss_dict: 차원별 손실
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        for dim in self.dimensions:
            # 앞쪽 dim 차원만 사용
            truncated1 = embeddings1[:, :dim]
            truncated2 = embeddings2[:, :dim]

            # Normalize
            norm1 = F.normalize(truncated1, p=2, dim=1)
            norm2 = F.normalize(truncated2, p=2, dim=1)

            # Similarity matrix
            sim_matrix = torch.mm(norm1, norm2.T) / self.temperature

            # Labels
            labels = torch.arange(batch_size, device=device)

            # Cross-entropy loss
            loss = F.cross_entropy(sim_matrix, labels)

            # Weighted sum
            weight = self.dimension_weights[dim]
            total_loss = total_loss + weight * loss

            loss_dict[f'loss_dim_{dim}'] = loss.item()

        loss_dict['matryoshka_total'] = total_loss.item()

        return total_loss, loss_dict

    def get_truncated_embedding(
        self,
        embeddings: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """
        특정 차원으로 truncate된 임베딩 반환

        Args:
            embeddings: [batch_size, full_dim]
            target_dim: 목표 차원

        Returns:
            truncated: [batch_size, target_dim]
        """
        if target_dim not in self.dimensions:
            logger.warning(f"Dimension {target_dim} not in trained dimensions. Using closest.")
            target_dim = min(self.dimensions, key=lambda x: abs(x - target_dim))

        truncated = embeddings[:, :target_dim]
        return F.normalize(truncated, p=2, dim=1)


class MultiGranularityRepresentation(nn.Module):
    """
    Multi-granularity Representation (MGR)

    자모 → 음절 → 어절 → 문장 계층적 표현 학습
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        granularity_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 기본 가중치
        self.granularity_weights = granularity_weights or {
            'jamo': 0.1,
            'syllable': 0.2,
            'token': 0.3,
            'sentence': 0.4
        }

        # 계층별 projection
        self.projections = nn.ModuleDict({
            'jamo': nn.Linear(hidden_dim, hidden_dim),
            'syllable': nn.Linear(hidden_dim, hidden_dim),
            'token': nn.Linear(hidden_dim, hidden_dim),
            'sentence': nn.Linear(hidden_dim, hidden_dim)
        })

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        계층적 임베딩 통합

        Args:
            embeddings: {
                'jamo': [B, D],
                'syllable': [B, D],
                'token': [B, D],
                'sentence': [B, D]
            }

        Returns:
            unified: [B, D] 통합 임베딩
        """
        unified = torch.zeros_like(next(iter(embeddings.values())))

        for granularity, emb in embeddings.items():
            if granularity in self.projections:
                projected = self.projections[granularity](emb)
                weight = self.granularity_weights.get(granularity, 0.25)
                unified = unified + weight * projected

        return F.normalize(unified, p=2, dim=1)


# ============================================================================
# Adaptive Loss Balancing (ALB) - GradNorm Style
# ============================================================================

class GradNormBalancer(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing

    다중 손실 함수의 가중치를 gradient magnitude 기반으로 자동 조정

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
    """

    def __init__(
        self,
        num_losses: int,
        loss_names: List[str] = None,
        alpha: float = 1.5,
        learning_rate: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 10.0
    ):
        """
        Args:
            num_losses: 손실 함수 개수
            loss_names: 손실 함수 이름들
            alpha: 균형 강도 (높을수록 균형화 강함)
            learning_rate: 가중치 업데이트 학습률
            min_weight: 최소 가중치
            max_weight: 최대 가중치
        """
        super().__init__()
        self.num_losses = num_losses
        self.loss_names = loss_names or [f'loss_{i}' for i in range(num_losses)]
        self.alpha = alpha
        self.lr = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 학습 가능한 가중치 (log scale)
        self.log_weights = nn.Parameter(torch.zeros(num_losses))

        # 초기 손실값 저장
        self.initial_losses: Optional[torch.Tensor] = None
        self.step_count = 0

    @property
    def weights(self) -> torch.Tensor:
        """현재 가중치 (exp로 변환)"""
        return F.softmax(self.log_weights, dim=0)

    def compute_grad_norms(
        self,
        losses: List[torch.Tensor],
        shared_layer: nn.Module
    ) -> List[torch.Tensor]:
        """
        각 손실의 gradient norm 계산

        Args:
            losses: 손실 텐서 리스트
            shared_layer: 공유 레이어 (gradient 계산용)

        Returns:
            grad_norms: gradient norm 리스트
        """
        grad_norms = []

        for loss in losses:
            # Gradient 계산
            grads = torch.autograd.grad(
                loss,
                shared_layer.parameters(),
                retain_graph=True,
                allow_unused=True
            )

            # Flatten and compute norm
            valid_grads = [g.flatten() for g in grads if g is not None]
            if valid_grads:
                total_grad = torch.cat(valid_grads)
                grad_norm = torch.norm(total_grad)
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            grad_norms.append(grad_norm)

        return grad_norms

    def forward(
        self,
        losses: List[torch.Tensor],
        shared_layer: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        가중 합산 손실 계산 및 가중치 업데이트

        Args:
            losses: 손실 텐서 리스트
            shared_layer: 공유 레이어 (GradNorm 계산용, None이면 단순 가중 합)

        Returns:
            weighted_loss: 가중 합산 손실
            info: 가중치 및 손실 정보
        """
        assert len(losses) == self.num_losses, f"Expected {self.num_losses} losses, got {len(losses)}"

        weights = self.weights
        device = losses[0].device

        # 초기 손실 저장 (첫 번째 호출 시)
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [l.item() for l in losses],
                device=device
            )

        # 가중 손실
        weighted_loss = sum(w * l for w, l in zip(weights, losses))

        # GradNorm 업데이트 (shared_layer 제공 시)
        if shared_layer is not None and self.training:
            grad_norms = self.compute_grad_norms(losses, shared_layer)

            # 평균 gradient norm
            mean_norm = sum(grad_norms) / len(grad_norms)

            # 상대적 학습 진행률
            current_losses = torch.tensor([l.item() for l in losses], device=device)
            relative_losses = current_losses / (self.initial_losses + 1e-8)

            # 목표 gradient norm
            mean_relative = relative_losses.mean()
            target_norms = mean_norm * (relative_losses / mean_relative) ** self.alpha

            # 가중치 gradient 계산 및 업데이트
            with torch.no_grad():
                for i, (gn, tn) in enumerate(zip(grad_norms, target_norms)):
                    # 현재 norm과 목표 norm의 차이
                    if gn > 0:
                        grad_ratio = (tn / gn).clamp(0.5, 2.0)
                        self.log_weights.data[i] += self.lr * torch.log(grad_ratio)

                # Clamp weights
                self.log_weights.data = self.log_weights.data.clamp(
                    np.log(self.min_weight),
                    np.log(self.max_weight)
                )

        self.step_count += 1

        # 정보 반환
        info = {
            'weighted_loss': weighted_loss.item(),
            **{
                f'weight_{name}': w.item()
                for name, w in zip(self.loss_names, weights)
            },
            **{
                f'loss_{name}': l.item()
                for name, l in zip(self.loss_names, losses)
            }
        }

        return weighted_loss, info


class SimpleLossBalancer:
    """
    간단한 손실 균형 조정기

    GradNorm 없이 EMA 기반 적응적 가중치 조정
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        ema_decay: float = 0.99,
        adaptation_rate: float = 0.1
    ):
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.ema_decay = ema_decay
        self.adaptation_rate = adaptation_rate

        # 초기 가중치
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            self.weights = {name: 1.0 / self.num_losses for name in loss_names}

        # EMA 손실
        self.ema_losses = {name: 1.0 for name in loss_names}

    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        """
        손실 기반 가중치 업데이트

        Args:
            losses: {'loss_name': loss_value, ...}

        Returns:
            updated_weights
        """
        # EMA 업데이트
        for name, loss in losses.items():
            if name in self.ema_losses:
                self.ema_losses[name] = (
                    self.ema_decay * self.ema_losses[name] +
                    (1 - self.ema_decay) * loss
                )

        # 손실 역수 기반 가중치 (손실 큰 것에 더 높은 가중치)
        total_loss = sum(self.ema_losses.values())
        if total_loss > 0:
            for name in self.loss_names:
                target_weight = self.ema_losses[name] / total_loss
                self.weights[name] = (
                    (1 - self.adaptation_rate) * self.weights[name] +
                    self.adaptation_rate * target_weight
                )

        # 정규화
        total_weight = sum(self.weights.values())
        self.weights = {name: w / total_weight for name, w in self.weights.items()}

        return self.weights.copy()

    def get_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """가중 손실 계산"""
        weighted = sum(
            self.weights.get(name, 1.0 / self.num_losses) * loss
            for name, loss in losses.items()
        )
        return weighted


# ============================================================================
# Validation-guided Training (VGT)
# ============================================================================

@dataclass
class ValidationResult:
    """검증 결과"""
    step: int
    epoch: int
    loss: float
    metrics: Dict[str, float]
    is_best: bool


class EarlyStopping:
    """Early Stopping 구현"""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'min',  # 'min' for loss, 'max' for accuracy
        restore_best: bool = True
    ):
        """
        Args:
            patience: 개선 없이 기다리는 epoch 수
            min_delta: 개선으로 인정하는 최소 변화량
            mode: 'min' (손실) 또는 'max' (정확도)
            restore_best: 최적 체크포인트 복원 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_epoch = 0
        self.best_state_dict: Optional[Dict] = None

    def __call__(
        self,
        score: float,
        epoch: int,
        model: nn.Module = None
    ) -> Tuple[bool, bool]:
        """
        Early stopping 체크

        Args:
            score: 현재 점수 (loss 또는 metric)
            epoch: 현재 epoch
            model: 모델 (best 저장용)

        Returns:
            (should_stop, is_best)
        """
        is_best = False

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            if model and self.restore_best:
                self.best_state_dict = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
        else:
            if self.mode == 'min':
                improved = score < self.best_score - self.min_delta
            else:
                improved = score > self.best_score + self.min_delta

            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
                is_best = True
                if model and self.restore_best:
                    self.best_state_dict = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
            else:
                self.counter += 1

        should_stop = self.counter >= self.patience
        return should_stop, is_best

    def restore_best_model(self, model: nn.Module):
        """최적 모델 복원"""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best model from epoch {self.best_epoch}")


class ValidationCallback:
    """
    검증 콜백

    학습 중 주기적으로 검증을 수행하고 결과를 기록
    """

    def __init__(
        self,
        validation_fn: Callable[[nn.Module], Dict[str, float]],
        eval_steps: int = 500,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        save_best: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Args:
            validation_fn: 검증 함수 (model -> metrics dict)
            eval_steps: 검증 주기 (steps)
            metric_name: 추적할 메트릭 이름
            mode: 'min' 또는 'max'
            save_best: 최적 체크포인트 저장
            output_dir: 체크포인트 저장 경로
        """
        self.validation_fn = validation_fn
        self.eval_steps = eval_steps
        self.metric_name = metric_name
        self.mode = mode
        self.save_best = save_best
        self.output_dir = Path(output_dir) if output_dir else None

        self.history: List[ValidationResult] = []
        self.early_stopping = EarlyStopping(patience=5, mode=mode)
        self.best_metrics: Optional[Dict[str, float]] = None

    def should_evaluate(self, step: int) -> bool:
        """검증 수행 여부"""
        return step > 0 and step % self.eval_steps == 0

    def evaluate(
        self,
        model: nn.Module,
        step: int,
        epoch: int,
        tokenizer=None
    ) -> ValidationResult:
        """
        검증 수행

        Args:
            model: 모델
            step: 현재 step
            epoch: 현재 epoch
            tokenizer: 토크나이저 (저장용)

        Returns:
            ValidationResult
        """
        model.eval()

        with torch.no_grad():
            metrics = self.validation_fn(model)

        score = metrics.get(self.metric_name, 0.0)
        should_stop, is_best = self.early_stopping(score, epoch, model)

        result = ValidationResult(
            step=step,
            epoch=epoch,
            loss=metrics.get('val_loss', 0.0),
            metrics=metrics,
            is_best=is_best
        )
        self.history.append(result)

        if is_best:
            self.best_metrics = metrics.copy()
            if self.save_best and self.output_dir:
                self._save_checkpoint(model, tokenizer, metrics)

        model.train()

        return result

    def _save_checkpoint(
        self,
        model: nn.Module,
        tokenizer,
        metrics: Dict[str, float]
    ):
        """최적 체크포인트 저장"""
        if self.output_dir is None:
            return

        best_dir = self.output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(best_dir)

        if tokenizer:
            tokenizer.save_pretrained(best_dir)

        # 메트릭 저장
        with open(best_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved best checkpoint to {best_dir}")

    def get_summary(self) -> Dict[str, Any]:
        """검증 요약"""
        if not self.history:
            return {}

        return {
            'num_evaluations': len(self.history),
            'best_step': self.early_stopping.best_epoch,
            'best_metrics': self.best_metrics,
            'history': [
                {
                    'step': r.step,
                    'epoch': r.epoch,
                    'loss': r.loss,
                    'is_best': r.is_best
                }
                for r in self.history
            ]
        }


# ============================================================================
# KURE Unified Loss
# ============================================================================

class KURELoss(nn.Module):
    """
    KURE 통합 손실 함수

    모든 구성 요소를 통합:
    - PJC (Phonological Jamo Composition)
    - MGC (Morpheme-guided Curriculum)
    - HCL (Hierarchical Contrastive Learning)
    - MGR (Multi-granularity Representation)
    - ALB (Adaptive Loss Balancing)
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        use_matryoshka: bool = True,
        use_gradnorm: bool = True,
        temperature: float = 0.05,
        loss_names: List[str] = None
    ):
        """
        Args:
            hidden_dim: 임베딩 차원
            use_matryoshka: Matryoshka loss 사용
            use_gradnorm: GradNorm 가중치 조정 사용
            temperature: Contrastive temperature
            loss_names: 손실 함수 이름들
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_matryoshka = use_matryoshka
        self.use_gradnorm = use_gradnorm
        self.temperature = temperature

        # 손실 함수 이름
        self.loss_names = loss_names or [
            'contrastive', 'pjc', 'curriculum', 'matryoshka'
        ]

        # Matryoshka loss
        if use_matryoshka:
            self.matryoshka = MatryoshkaLoss(
                full_dim=hidden_dim,
                temperature=temperature
            )

        # GradNorm balancer
        if use_gradnorm:
            self.gradnorm = GradNormBalancer(
                num_losses=len(self.loss_names),
                loss_names=self.loss_names
            )
        else:
            self.balancer = SimpleLossBalancer(self.loss_names)

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        embeddings1: Optional[torch.Tensor] = None,
        embeddings2: Optional[torch.Tensor] = None,
        shared_layer: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        통합 손실 계산

        Args:
            losses: 개별 손실 딕셔너리
            embeddings1: 첫 번째 임베딩 (Matryoshka용)
            embeddings2: 두 번째 임베딩 (Matryoshka용)
            shared_layer: 공유 레이어 (GradNorm용)

        Returns:
            total_loss: 총 손실
            info: 상세 정보
        """
        info = {}

        # Matryoshka loss 추가
        if self.use_matryoshka and embeddings1 is not None and embeddings2 is not None:
            matryoshka_loss, matryoshka_info = self.matryoshka(embeddings1, embeddings2)
            losses['matryoshka'] = matryoshka_loss
            info.update(matryoshka_info)

        # 손실 리스트 구성
        loss_list = [
            losses.get(name, torch.tensor(0.0, device=next(iter(losses.values())).device))
            for name in self.loss_names
        ]

        # 가중 합산
        if self.use_gradnorm:
            total_loss, balance_info = self.gradnorm(loss_list, shared_layer)
        else:
            total_loss = self.balancer.get_weighted_loss(losses)
            balance_info = {'weighted_loss': total_loss.item()}

        info.update(balance_info)

        return total_loss, info


# ============================================================================
# Utility Functions
# ============================================================================

def create_validation_fn(
    eval_dataset,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 192,
    device: torch.device = None
) -> Callable:
    """
    검증 함수 생성

    Args:
        eval_dataset: 검증 데이터셋
        tokenizer: 토크나이저
        batch_size: 배치 크기
        max_length: 최대 길이
        device: 디바이스

    Returns:
        validation_fn: 모델을 받아 메트릭을 반환하는 함수
    """
    from torch.utils.data import DataLoader

    def collate_fn(examples):
        texts = [ex['text'] for ex in examples]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    def validation_fn(model) -> Dict[str, float]:
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True
                )

                # 간단한 reconstruction loss 또는 contrastive loss
                hidden = outputs.last_hidden_state
                # Mean pooling
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                # Variance as proxy for quality
                variance = pooled.var(dim=0).mean()
                loss = -variance  # 높은 variance가 좋음

                total_loss += loss.item() * batch['input_ids'].shape[0]
                total_samples += batch['input_ids'].shape[0]

        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        return {
            'val_loss': avg_loss,
            'num_samples': total_samples
        }

    return validation_fn
