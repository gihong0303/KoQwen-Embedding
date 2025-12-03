"""
Supervised Retrieval Contrastive Loss Functions

이 모듈은 검색 태스크에 특화된 supervised contrastive loss를 구현합니다.
MIRACL, MrTidy 같은 데이터셋의 query-positive-negative 쌍을 활용합니다.

기존 SimCSE (unsupervised)와 달리, 실제 관련 문서를 positive로 사용하여
검색 성능을 직접 최적화합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
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


def gather_embeddings_from_all_gpus(embeddings: torch.Tensor) -> torch.Tensor:
    """
    DDP 환경에서 모든 GPU의 임베딩을 수집

    Args:
        embeddings: [local_batch_size, hidden_dim]

    Returns:
        all_embeddings: [global_batch_size, hidden_dim]
    """
    if not dist.is_initialized():
        return embeddings

    world_size = dist.get_world_size()
    if world_size == 1:
        return embeddings

    # Gather sizes from all processes
    local_size = torch.tensor([embeddings.shape[0]], device=embeddings.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    max_size = max(s.item() for s in sizes)

    # Pad to max size
    if embeddings.shape[0] < max_size:
        padding = torch.zeros(
            max_size - embeddings.shape[0],
            embeddings.shape[1],
            device=embeddings.device,
            dtype=embeddings.dtype
        )
        embeddings_padded = torch.cat([embeddings, padding], dim=0)
    else:
        embeddings_padded = embeddings

    # All-gather
    gathered = [torch.zeros_like(embeddings_padded) for _ in range(world_size)]
    dist.all_gather(gathered, embeddings_padded)

    # Remove padding and concatenate
    result = []
    for i, (emb, size) in enumerate(zip(gathered, sizes)):
        result.append(emb[:size.item()])

    return torch.cat(result, dim=0)


class SupervisedRetrievalLoss(nn.Module):
    """
    Supervised Retrieval Contrastive Loss

    Query-Positive-Negative 쌍을 사용한 검색 최적화 loss.
    In-batch negatives + Hard negatives를 함께 활용합니다.

    Args:
        temperature: Softmax temperature (default: 0.05)
        pooling: Pooling method ("mean" or "cls")
        use_hard_negatives: Hard negative를 사용할지 여부
        gather_with_grad: DDP에서 gradient를 유지하며 gather할지
    """

    def __init__(
        self,
        temperature: float = 0.05,
        pooling: str = "mean",
        use_hard_negatives: bool = True,
        gather_with_grad: bool = False
    ):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling
        self.use_hard_negatives = use_hard_negatives
        self.gather_with_grad = gather_with_grad

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
        Forward pass for supervised retrieval loss

        Args:
            model: Embedding model
            query_input_ids: [batch_size, seq_len] - Query tokens
            query_attention_mask: [batch_size, seq_len]
            pos_input_ids: [batch_size, seq_len] - Positive document tokens
            pos_attention_mask: [batch_size, seq_len]
            neg_input_ids: [batch_size * num_neg, seq_len] - Hard negative tokens (optional)
            neg_attention_mask: [batch_size * num_neg, seq_len]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        model.train()
        batch_size = query_input_ids.shape[0]

        # Get embeddings
        query_emb = self._get_embeddings(model, query_input_ids, query_attention_mask)
        pos_emb = self._get_embeddings(model, pos_input_ids, pos_attention_mask)

        # Normalize
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        # Hard negatives (if provided)
        if neg_input_ids is not None and self.use_hard_negatives:
            neg_emb = self._get_embeddings(model, neg_input_ids, neg_attention_mask)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)
        else:
            neg_emb = None

        # Gather from all GPUs for larger batch (in-batch negatives)
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_pos_emb = gather_embeddings_from_all_gpus(pos_emb)
            if neg_emb is not None:
                all_neg_emb = gather_embeddings_from_all_gpus(neg_emb)
            else:
                all_neg_emb = None
        else:
            all_pos_emb = pos_emb
            all_neg_emb = neg_emb

        # Compute similarity scores
        # Query-Positive similarity: [batch_size, global_batch_size]
        pos_scores = torch.mm(query_emb, all_pos_emb.t()) / self.temperature

        # Query-Negative similarity (if hard negatives provided)
        if all_neg_emb is not None:
            neg_scores = torch.mm(query_emb, all_neg_emb.t()) / self.temperature
            # Concatenate: [batch_size, global_batch_size + num_neg * global_batch_size]
            all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        else:
            all_scores = pos_scores

        # Labels: positive is at position [rank * local_batch + local_idx]
        if dist.is_initialized():
            rank = dist.get_rank()
            labels = torch.arange(batch_size, device=query_emb.device) + rank * batch_size
        else:
            labels = torch.arange(batch_size, device=query_emb.device)

        # InfoNCE Loss
        loss = F.cross_entropy(all_scores, labels)

        # Compute metrics
        with torch.no_grad():
            # Accuracy (positive in top-1)
            predictions = all_scores.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            # Mean positive score
            pos_score_mean = torch.diagonal(pos_scores).mean()

            # Mean negative score (in-batch)
            mask = torch.ones_like(pos_scores, dtype=torch.bool)
            mask[torch.arange(batch_size), labels % pos_scores.shape[1]] = False
            neg_score_mean = pos_scores[mask].mean() if mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score": pos_score_mean.item(),
            "neg_score": neg_score_mean.item(),
            "margin": (pos_score_mean - neg_score_mean).item()
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
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

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
        """
        Forward pass for triplet loss
        """
        model.train()

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

            # How many triplets satisfy the margin
            satisfied = (neg_dist - pos_dist > self.margin).float().mean()

        metrics = {
            "loss": loss.item(),
            "pos_distance": pos_dist.mean().item(),
            "neg_distance": neg_dist.mean().item(),
            "margin_satisfied": satisfied.item()
        }

        return loss, metrics


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (sentence-transformers style)

    배치 내의 다른 positive들을 negative로 활용하는 효율적인 loss.
    Hard negative를 추가로 포함할 수 있습니다.

    Reference: https://arxiv.org/abs/1705.00652
    """

    def __init__(
        self,
        temperature: float = 0.05,
        pooling: str = "mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling

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
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass

        In-batch negatives + optional hard negatives
        """
        model.train()
        batch_size = query_input_ids.shape[0]

        # Get query embeddings
        query_emb = self._get_embeddings(model, query_input_ids, query_attention_mask)
        query_emb = F.normalize(query_emb, p=2, dim=1)

        # Get positive embeddings
        pos_emb = self._get_embeddings(model, pos_input_ids, pos_attention_mask)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        # Build candidate pool
        candidates = [pos_emb]

        # Add hard negatives if provided
        if neg_input_ids is not None:
            neg_emb = self._get_embeddings(model, neg_input_ids, neg_attention_mask)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)
            candidates.append(neg_emb)

        # Concatenate all candidates
        all_candidates = torch.cat(candidates, dim=0)  # [batch + num_neg, hidden]

        # Similarity matrix: [batch, batch + num_neg]
        scores = torch.mm(query_emb, all_candidates.t()) / self.temperature

        # Labels: positive is at position i for query i
        labels = torch.arange(batch_size, device=query_emb.device)

        # Cross-entropy loss
        loss = F.cross_entropy(scores, labels)

        # Metrics
        with torch.no_grad():
            accuracy = (scores.argmax(dim=1) == labels).float().mean()
            pos_scores = torch.diagonal(scores[:, :batch_size])
            neg_scores = scores[:, :batch_size].clone()
            neg_scores.fill_diagonal_(float('-inf'))
            neg_scores_mean = neg_scores[neg_scores != float('-inf')].mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_score": pos_scores.mean().item(),
            "neg_score": neg_scores_mean.item()
        }

        return loss, metrics
