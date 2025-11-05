"""
Training utilities for distributed training, optimization, and loss functions
"""

import os
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


def setup_distributed() -> Tuple[int, int, int]:
    """
    분산 학습 환경 설정

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def is_main_process() -> bool:
    """메인 프로세스 여부 확인"""
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    """동기화 barrier"""
    if dist.is_initialized():
        dist.barrier()


def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8
) -> AdamW:
    """
    Optimizer 설정

    Args:
        model: 모델
        learning_rate: 학습률
        weight_decay: Weight decay
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon

    Returns:
        Optimizer
    """
    # Weight decay 제외할 파라미터
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon
    )

    return optimizer


def setup_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear"
):
    """
    Learning rate scheduler 설정

    Args:
        optimizer: Optimizer
        num_training_steps: 전체 학습 스텝 수
        warmup_ratio: Warmup 비율
        scheduler_type: 스케줄러 타입 ("linear", "cosine")

    Returns:
        Scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean pooling (attention mask 고려)

    Args:
        hidden_states: [B, L, D]
        attention_mask: [B, L]

    Returns:
        [B, D]
    """
    # Expand mask
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    # Sum embeddings
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

    # Sum mask
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def simcse_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.05
) -> torch.Tensor:
    """
    SimCSE loss (InfoNCE)
    같은 문장의 서로 다른 dropout을 positive pair로 간주

    Args:
        embeddings: [2*B, D] - 각 샘플이 2번씩 forward (다른 dropout)
        temperature: Temperature

    Returns:
        loss
    """
    batch_size = embeddings.size(0) // 2

    # Normalize
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Similarity matrix: [2B, 2B]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(batch_size, device=embeddings.device)
    labels = torch.cat([labels + batch_size, labels])  # [0+B, 1+B, ..., B-1+B, 0, 1, ..., B-1]

    # CrossEntropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def contrastive_loss(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    temperature: float = 0.05
) -> torch.Tensor:
    """
    Contrastive loss (대조 학습)

    Args:
        embeddings1: [B, D]
        embeddings2: [B, D]
        temperature: Temperature

    Returns:
        loss
    """
    batch_size = embeddings1.size(0)

    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Similarity matrices
    sim_11 = torch.matmul(embeddings1, embeddings1.T) / temperature
    sim_22 = torch.matmul(embeddings2, embeddings2.T) / temperature
    sim_12 = torch.matmul(embeddings1, embeddings2.T) / temperature
    sim_21 = torch.matmul(embeddings2, embeddings1.T) / temperature

    # Labels (diagonal)
    labels = torch.arange(batch_size, device=embeddings1.device)

    # Loss
    loss_12 = F.cross_entropy(sim_12, labels)
    loss_21 = F.cross_entropy(sim_21, labels)

    loss = (loss_12 + loss_21) / 2

    return loss


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    output_dir: str
):
    """
    체크포인트 저장

    Args:
        model: 모델
        optimizer: Optimizer
        scheduler: Scheduler
        epoch: Epoch
        step: Step
        loss: Loss
        output_dir: 출력 디렉토리
    """
    if not is_main_process():
        return

    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    checkpoint_path = output_path / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None
) -> Tuple[int, int, float]:
    """
    체크포인트 로드

    Args:
        checkpoint_path: 체크포인트 경로
        model: 모델
        optimizer: Optimizer (옵션)
        scheduler: Scheduler (옵션)

    Returns:
        (epoch, step, loss)
    """
    logger.info(f"체크포인트 로드: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0.0)

    logger.info(f"✓ 로드 완료: Epoch {epoch}, Step {step}, Loss {loss:.4f}")

    return epoch, step, loss
