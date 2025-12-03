"""
Supervised Retrieval Contrastive Loss Functions (Production Version)

이 모듈은 검색 태스크에 특화된 supervised contrastive loss를 구현합니다.
MIRACL, MrTyDi 같은 데이터셋의 query-positive-negative 쌍을 활용합니다.

주요 개선사항 (Production):
- GatherLayer: DDP에서 gradient 전파 지원
- 안정적인 loss 계산
- 메모리 효율적인 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.distributed as dist


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling with attention mask

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]

    Returns:
        pooled: [batch_size, hidden_dim]
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation.

    일반적인 dist.all_gather는 gradient를 전파하지 않습니다.
    이 클래스는 autograd.Function을 상속하여 gradient 전파를 지원합니다.

    Reference: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input_tensor)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Gather all tensors
        output = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(output, input_tensor)

        # Save rank for backward
        ctx.rank = rank
        ctx.world_size = world_size

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input_tensor, = ctx.saved_tensors
        rank = ctx.rank

        # Only propagate gradient for the current rank's tensor
        grad_input = grads[rank].clone()

        return grad_input


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all GPUs while preserving gradients.

    Args:
        tensor: [batch_size, hidden_dim] - Local tensor

    Returns:
        all_tensors: [world_size * batch_size, hidden_dim] - Gathered tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Use GatherLayer for gradient support
    gathered = GatherLayer.apply(tensor)

    return torch.cat(gathered, dim=0)


def gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all GPUs without gradients (for metrics).

    Args:
        tensor: [batch_size, hidden_dim] - Local tensor

    Returns:
        all_tensors: [world_size * batch_size, hidden_dim] - Gathered tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    with torch.no_grad():
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)

    return torch.cat(output, dim=0)


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (Production Version)

    배치 내의 다른 positive들을 negative로 활용하는 효율적인 loss.
    Hard negative를 추가로 포함할 수 있습니다.

    DDP 환경에서 cross-GPU in-batch negatives를 지원합니다.

    Reference: https://arxiv.org/abs/1705.00652

    Args:
        temperature: Softmax temperature (default: 0.05)
        pooling: Pooling method ("mean" or "cls")
        use_cross_gpu_negatives: DDP에서 다른 GPU의 샘플을 negative로 사용
    """

    def __init__(
        self,
        temperature: float = 0.05,
        pooling: str = "mean",
        use_cross_gpu_negatives: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling
        self.use_cross_gpu_negatives = use_cross_gpu_negatives

    def _get_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings from model"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.pooling == "mean":
            return mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        model: nn.Module,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with Multiple Negatives Ranking Loss

        In-batch negatives + optional hard negatives + optional cross-GPU negatives

        Args:
            model: Embedding model (NOT unwrapped - should be DDP model for proper gradient sync)
            query_input_ids: [batch_size, seq_len]
            query_attention_mask: [batch_size, seq_len]
            pos_input_ids: [batch_size, seq_len]
            pos_attention_mask: [batch_size, seq_len]
            neg_input_ids: [batch_size, seq_len] (optional hard negatives)
            neg_attention_mask: [batch_size, seq_len]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size = query_input_ids.shape[0]

        # Get embeddings through the model (important: keep DDP wrapper for gradient sync)
        query_emb = self._get_embeddings(model, query_input_ids, query_attention_mask)
        pos_emb = self._get_embeddings(model, pos_input_ids, pos_attention_mask)

        # Normalize
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        # Build candidate pool
        candidates = [pos_emb]

        # Add hard negatives if provided
        if neg_input_ids is not None:
            neg_emb = self._get_embeddings(model, neg_input_ids, neg_attention_mask)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)
            candidates.append(neg_emb)

        # Concatenate local candidates
        all_candidates = torch.cat(candidates, dim=0)  # [batch + num_neg, hidden]

        # Cross-GPU gathering for larger effective batch size
        if self.use_cross_gpu_negatives and dist.is_initialized() and dist.get_world_size() > 1:
            # Gather positive embeddings from all GPUs (with gradient)
            all_pos_emb = gather_with_grad(pos_emb)

            # For hard negatives, gather without gradient (they're already negatives)
            if neg_input_ids is not None:
                all_neg_emb = gather_without_grad(neg_emb)
                all_candidates = torch.cat([all_pos_emb, all_neg_emb], dim=0)
            else:
                all_candidates = all_pos_emb

            # Similarity matrix: [local_batch, global_batch + global_neg]
            scores = torch.mm(query_emb, all_candidates.t()) / self.temperature

            # Labels: positive is at position [rank * local_batch + local_idx]
            rank = dist.get_rank()
            labels = torch.arange(batch_size, device=query_emb.device) + rank * batch_size
        else:
            # Local only
            scores = torch.mm(query_emb, all_candidates.t()) / self.temperature
            labels = torch.arange(batch_size, device=query_emb.device)

        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(scores, labels)

        # Metrics (without gradient)
        with torch.no_grad():
            accuracy = (scores.argmax(dim=1) == labels).float().mean()

            # Positive/negative scores for monitoring
            pos_scores_diag = torch.gather(
                scores, 1, labels.unsqueeze(1)
            ).squeeze(1)
            pos_score_mean = pos_scores_diag.mean()

            # Negative scores (exclude positive)
            neg_mask = torch.ones_like(scores, dtype=torch.bool)
            neg_mask.scatter_(1, labels.unsqueeze(1), False)
            neg_score_mean = scores[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score": pos_score_mean.item(),
            "neg_score": neg_score_mean.item(),
            "margin": (pos_score_mean - neg_score_mean).item(),
            "effective_batch_size": scores.shape[1]
        }

        return loss, metrics


class SupervisedRetrievalLoss(nn.Module):
    """
    Supervised Retrieval Contrastive Loss (Production Version)

    Query-Positive-Negative 쌍을 사용한 검색 최적화 loss.
    Cross-GPU in-batch negatives를 지원합니다.

    Args:
        temperature: Softmax temperature (default: 0.05)
        pooling: Pooling method ("mean" or "cls")
        use_hard_negatives: Hard negative를 사용할지 여부
        use_cross_gpu_negatives: DDP에서 cross-GPU negatives 사용
    """

    def __init__(
        self,
        temperature: float = 0.05,
        pooling: str = "mean",
        use_hard_negatives: bool = True,
        use_cross_gpu_negatives: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling
        self.use_hard_negatives = use_hard_negatives
        self.use_cross_gpu_negatives = use_cross_gpu_negatives

    def _get_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.pooling == "mean":
            return mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        model: nn.Module,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass for supervised retrieval loss"""
        batch_size = query_input_ids.shape[0]

        # Get embeddings
        query_emb = self._get_embeddings(model, query_input_ids, query_attention_mask)
        pos_emb = self._get_embeddings(model, pos_input_ids, pos_attention_mask)

        # Normalize
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        # Hard negatives
        if neg_input_ids is not None and self.use_hard_negatives:
            neg_emb = self._get_embeddings(model, neg_input_ids, neg_attention_mask)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)
        else:
            neg_emb = None

        # Cross-GPU gathering
        if self.use_cross_gpu_negatives and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # Gather with gradient for positive embeddings
            all_pos_emb = gather_with_grad(pos_emb)

            # Similarity with all positives
            pos_scores = torch.mm(query_emb, all_pos_emb.t()) / self.temperature

            # Hard negatives (gather without gradient)
            if neg_emb is not None:
                all_neg_emb = gather_without_grad(neg_emb)
                neg_scores = torch.mm(query_emb, all_neg_emb.t()) / self.temperature
                all_scores = torch.cat([pos_scores, neg_scores], dim=1)
            else:
                all_scores = pos_scores

            # Labels
            labels = torch.arange(batch_size, device=query_emb.device) + rank * batch_size
        else:
            # Local only
            pos_scores = torch.mm(query_emb, pos_emb.t()) / self.temperature

            if neg_emb is not None:
                neg_scores = torch.mm(query_emb, neg_emb.t()) / self.temperature
                all_scores = torch.cat([pos_scores, neg_scores], dim=1)
            else:
                all_scores = pos_scores

            labels = torch.arange(batch_size, device=query_emb.device)

        # InfoNCE Loss
        loss = F.cross_entropy(all_scores, labels)

        # Metrics
        with torch.no_grad():
            predictions = all_scores.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            pos_score_mean = torch.gather(
                all_scores, 1, labels.unsqueeze(1)
            ).squeeze(1).mean()

            neg_mask = torch.ones_like(all_scores, dtype=torch.bool)
            neg_mask.scatter_(1, labels.unsqueeze(1), False)
            neg_score_mean = all_scores[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score": pos_score_mean.item(),
            "neg_score": neg_score_mean.item(),
            "margin": (pos_score_mean - neg_score_mean).item(),
            "effective_batch_size": all_scores.shape[1]
        }

        return loss, metrics


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for Retrieval

    Alternative to InfoNCE, directly optimizes margin between positive and negative.

    Args:
        margin: Margin for triplet loss
        pooling: Pooling method
    """

    def __init__(self, margin: float = 0.2, pooling: str = "mean"):
        super().__init__()
        self.margin = margin
        self.pooling = pooling
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

    def _get_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.pooling == "mean":
            return mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        model: nn.Module,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # Get embeddings
        anchor = self._get_embeddings(model, query_input_ids, query_attention_mask)
        positive = self._get_embeddings(model, pos_input_ids, pos_attention_mask)
        negative = self._get_embeddings(model, neg_input_ids, neg_attention_mask)

        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Triplet loss
        loss = self.triplet_loss(anchor, positive, negative)

        # Metrics
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            satisfied = (neg_dist - pos_dist > self.margin).float().mean()

        metrics = {
            "loss": loss.item(),
            "pos_distance": pos_dist.mean().item(),
            "neg_distance": neg_dist.mean().item(),
            "margin_satisfied": satisfied.item()
        }

        return loss, metrics
