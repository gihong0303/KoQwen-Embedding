"""
Hierarchical Contrastive Learning (HCL)

계층적 대조 학습 + Hard Negative Mining
- Token-level Contrastive
- Phrase-level Contrastive with Hard Negatives
- Sentence-level Contrastive (SimCSE)

Reference:
- SimCSE (Gao et al., EMNLP 2021)
- HiCLR (Wang et al., 2022)
- CLIP (Radford et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Hard Negative Mining
# ============================================================================

class HardNegativeMiner:
    """
    Hard Negative Mining

    Cross-batch negatives와 Semi-hard negative 선택
    Memory Bank를 유지하여 더 많은 negative 후보 확보
    """

    def __init__(
        self,
        memory_size: int = 65536,
        hidden_dim: int = 1536,
        margin: float = 0.2,
        mining_strategy: str = 'semi_hard'  # 'hard', 'semi_hard', 'all'
    ):
        """
        Args:
            memory_size: Memory Bank 크기
            hidden_dim: 임베딩 차원
            margin: Semi-hard negative 마진
            mining_strategy: Mining 전략
        """
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.mining_strategy = mining_strategy

        # Memory Bank (FIFO)
        self.memory_bank = deque(maxlen=memory_size)
        self.memory_labels = deque(maxlen=memory_size)

    def update_memory(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Memory Bank 업데이트

        Args:
            embeddings: [batch_size, hidden_dim]
            labels: [batch_size] (선택)
        """
        # Detach하여 gradient 끊기
        embeddings_np = embeddings.detach().cpu().numpy()

        for i in range(embeddings_np.shape[0]):
            self.memory_bank.append(embeddings_np[i])
            if labels is not None:
                self.memory_labels.append(labels[i].item())
            else:
                self.memory_labels.append(-1)

    def get_memory_embeddings(self, device: torch.device) -> torch.Tensor:
        """Memory Bank 임베딩 반환"""
        if len(self.memory_bank) == 0:
            return None

        memory_array = np.stack(list(self.memory_bank), axis=0)
        return torch.tensor(memory_array, device=device, dtype=torch.float32)

    def mine_hard_negatives(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        num_negatives: int = 8
    ) -> torch.Tensor:
        """
        Hard Negatives 선택

        Args:
            query_embeddings: [batch_size, hidden_dim]
            positive_embeddings: [batch_size, hidden_dim]
            num_negatives: 선택할 negative 수

        Returns:
            hard_negatives: [batch_size, num_negatives, hidden_dim]
        """
        device = query_embeddings.device
        batch_size = query_embeddings.shape[0]

        memory_embeddings = self.get_memory_embeddings(device)
        if memory_embeddings is None or len(self.memory_bank) < num_negatives:
            return None

        # Query와 Memory Bank 간 유사도
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        memory_norm = F.normalize(memory_embeddings, p=2, dim=1)

        similarities = torch.mm(query_norm, memory_norm.T)  # [B, M]

        # Positive와의 유사도
        pos_norm = F.normalize(positive_embeddings, p=2, dim=1)
        pos_similarities = (query_norm * pos_norm).sum(dim=1)  # [B]

        hard_negatives_list = []

        for i in range(batch_size):
            pos_sim = pos_similarities[i]
            neg_sims = similarities[i]

            if self.mining_strategy == 'hard':
                # Hard: 가장 유사한 것 선택 (positive보다 유사한 것 우선)
                _, indices = neg_sims.topk(num_negatives)

            elif self.mining_strategy == 'semi_hard':
                # Semi-hard: positive보다 덜 유사하지만 margin 내
                mask = (neg_sims < pos_sim) & (neg_sims > pos_sim - self.margin)
                valid_indices = mask.nonzero(as_tuple=True)[0]

                if len(valid_indices) >= num_negatives:
                    # 가장 어려운 것 선택
                    valid_sims = neg_sims[valid_indices]
                    _, top_indices = valid_sims.topk(num_negatives)
                    indices = valid_indices[top_indices]
                else:
                    # 부족하면 그냥 hard negative로
                    _, indices = neg_sims.topk(num_negatives)

            else:  # 'all'
                # 랜덤 선택
                indices = torch.randperm(len(neg_sims))[:num_negatives]

            hard_negatives_list.append(memory_embeddings[indices])

        hard_negatives = torch.stack(hard_negatives_list, dim=0)  # [B, N, D]
        return hard_negatives


# ============================================================================
# Token-level Contrastive Loss
# ============================================================================

class TokenContrastiveLoss(nn.Module):
    """
    토큰 레벨 대조 학습

    같은 의미의 토큰끼리 가깝게, 다른 의미의 토큰끼리 멀게
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, hidden_dim]
            positive_embeddings: [batch_size, hidden_dim]
            negative_embeddings: [batch_size, num_negatives, hidden_dim] (선택)

        Returns:
            loss: 토큰 레벨 대조 손실
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        # Normalize
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=1)

        # Positive similarity
        pos_sim = (anchor_norm * positive_norm).sum(dim=1) / self.temperature  # [B]

        if negative_embeddings is not None and self.use_hard_negatives:
            # Hard negative similarity
            neg_norm = F.normalize(negative_embeddings, p=2, dim=2)  # [B, N, D]
            neg_sim = torch.bmm(
                anchor_norm.unsqueeze(1),  # [B, 1, D]
                neg_norm.transpose(1, 2)    # [B, D, N]
            ).squeeze(1) / self.temperature  # [B, N]

            # InfoNCE with hard negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)

        else:
            # In-batch negatives only
            sim_matrix = torch.mm(anchor_norm, positive_norm.T) / self.temperature
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ============================================================================
# Phrase-level Contrastive Loss
# ============================================================================

class PhraseContrastiveLoss(nn.Module):
    """
    구절 레벨 대조 학습

    의미적으로 유사한 구절끼리 가깝게
    Hard negatives로 변별력 강화
    """

    def __init__(
        self,
        temperature: float = 0.05,
        hard_negative_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        phrase_embeddings_1: torch.Tensor,
        phrase_embeddings_2: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            phrase_embeddings_1: [batch_size, hidden_dim]
            phrase_embeddings_2: [batch_size, hidden_dim]
            hard_negatives: [batch_size, num_negatives, hidden_dim]

        Returns:
            loss: 구절 레벨 대조 손실
        """
        batch_size = phrase_embeddings_1.shape[0]
        device = phrase_embeddings_1.device

        # Normalize
        emb1_norm = F.normalize(phrase_embeddings_1, p=2, dim=1)
        emb2_norm = F.normalize(phrase_embeddings_2, p=2, dim=1)

        # Standard SimCSE loss
        sim_matrix = torch.mm(emb1_norm, emb2_norm.T) / self.temperature
        labels = torch.arange(batch_size, device=device)
        simcse_loss = F.cross_entropy(sim_matrix, labels)

        if hard_negatives is not None:
            # Hard negative loss
            neg_norm = F.normalize(hard_negatives, p=2, dim=2)

            # Positive similarity
            pos_sim = (emb1_norm * emb2_norm).sum(dim=1)  # [B]

            # Hard negative similarity
            neg_sim = torch.bmm(
                emb1_norm.unsqueeze(1),
                neg_norm.transpose(1, 2)
            ).squeeze(1)  # [B, N]

            # Margin-based loss
            margin = 0.2
            hard_loss = F.relu(neg_sim - pos_sim.unsqueeze(1) + margin).mean()

            # Combined loss
            loss = simcse_loss + self.hard_negative_weight * hard_loss
        else:
            loss = simcse_loss

        return loss


# ============================================================================
# Sentence-level Contrastive Loss (SimCSE)
# ============================================================================

class SentenceContrastiveLoss(nn.Module):
    """
    문장 레벨 대조 학습 (SimCSE)

    Dropout을 이용한 unsupervised contrastive learning
    """

    def __init__(
        self,
        temperature: float = 0.05,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling

    def mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            model: Embedding model
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            loss: 문장 레벨 대조 손실
            embeddings: 풀링된 임베딩
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Enable dropout
        model.train()

        # First forward pass
        outputs1 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Second forward pass (different dropout)
        outputs2 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pooling
        if self.pooling == 'mean':
            emb1 = self.mean_pooling(outputs1.last_hidden_state, attention_mask)
            emb2 = self.mean_pooling(outputs2.last_hidden_state, attention_mask)
        elif self.pooling == 'cls':
            emb1 = outputs1.last_hidden_state[:, 0, :]
            emb2 = outputs2.last_hidden_state[:, 0, :]
        elif self.pooling == 'last':
            # Last token pooling
            seq_lengths = attention_mask.sum(dim=1) - 1
            emb1 = outputs1.last_hidden_state[torch.arange(batch_size), seq_lengths]
            emb2 = outputs2.last_hidden_state[torch.arange(batch_size), seq_lengths]
        else:
            emb1 = self.mean_pooling(outputs1.last_hidden_state, attention_mask)
            emb2 = self.mean_pooling(outputs2.last_hidden_state, attention_mask)

        # Normalize
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(emb1_norm, emb2_norm.T) / self.temperature

        # Labels (diagonal is positive)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss, emb1


# ============================================================================
# Hierarchical Contrastive Loss (Combined)
# ============================================================================

class HierarchicalContrastiveLoss(nn.Module):
    """
    계층적 대조 학습 (HCL)

    Token + Phrase + Sentence 레벨 통합
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        token_temperature: float = 0.07,
        phrase_temperature: float = 0.05,
        sentence_temperature: float = 0.05,
        token_weight: float = 0.2,
        phrase_weight: float = 0.3,
        sentence_weight: float = 0.5,
        use_hard_negatives: bool = True,
        memory_size: int = 65536,
        pooling: str = 'mean'
    ):
        """
        Args:
            hidden_dim: 임베딩 차원
            token_temperature: 토큰 레벨 temperature
            phrase_temperature: 구절 레벨 temperature
            sentence_temperature: 문장 레벨 temperature
            token_weight: 토큰 레벨 가중치
            phrase_weight: 구절 레벨 가중치
            sentence_weight: 문장 레벨 가중치
            use_hard_negatives: Hard negative mining 사용
            memory_size: Memory Bank 크기
            pooling: Pooling 방식
        """
        super().__init__()

        self.token_weight = token_weight
        self.phrase_weight = phrase_weight
        self.sentence_weight = sentence_weight
        self.use_hard_negatives = use_hard_negatives

        # Sub-losses
        self.token_loss = TokenContrastiveLoss(
            temperature=token_temperature,
            use_hard_negatives=use_hard_negatives
        )
        self.phrase_loss = PhraseContrastiveLoss(
            temperature=phrase_temperature
        )
        self.sentence_loss = SentenceContrastiveLoss(
            temperature=sentence_temperature,
            pooling=pooling
        )

        # Hard negative miner
        if use_hard_negatives:
            self.hard_negative_miner = HardNegativeMiner(
                memory_size=memory_size,
                hidden_dim=hidden_dim
            )
        else:
            self.hard_negative_miner = None

    def forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: Embedding model
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_embeddings: [batch_size, hidden_dim] (토큰 임베딩, 선택)

        Returns:
            total_loss: 총 손실
            loss_dict: 개별 손실 값
        """
        # 1. Sentence-level loss (SimCSE)
        sent_loss, sentence_embeddings = self.sentence_loss(
            model, input_ids, attention_mask
        )

        # 2. Phrase-level loss
        # 두 번째 forward로 다른 임베딩 얻기
        model.train()
        outputs2 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        phrase_embeddings_2 = self.sentence_loss.mean_pooling(
            outputs2.last_hidden_state, attention_mask
        )

        # Hard negatives
        hard_negs = None
        if self.use_hard_negatives and self.hard_negative_miner:
            hard_negs = self.hard_negative_miner.mine_hard_negatives(
                sentence_embeddings,
                phrase_embeddings_2,
                num_negatives=8
            )
            # Memory 업데이트
            self.hard_negative_miner.update_memory(sentence_embeddings)

        phrase_loss = self.phrase_loss(
            sentence_embeddings,
            phrase_embeddings_2,
            hard_negatives=hard_negs
        )

        # 3. Token-level loss
        if token_embeddings is not None:
            # 토큰 임베딩이 제공된 경우
            embed_layer = model.get_input_embeddings()
            # 랜덤 토큰 선택
            batch_size = input_ids.shape[0]
            random_indices = torch.randint(
                0, input_ids.shape[1],
                (batch_size,),
                device=input_ids.device
            )
            anchor_tokens = input_ids[torch.arange(batch_size), random_indices]
            anchor_embeddings = embed_layer(anchor_tokens)

            # Positive: 같은 토큰의 다른 표현
            positive_embeddings = anchor_embeddings + torch.randn_like(anchor_embeddings) * 0.1

            token_loss = self.token_loss(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings=None  # In-batch negatives
            )
        else:
            token_loss = torch.tensor(0.0, device=input_ids.device)

        # Combined loss
        total_loss = (
            self.token_weight * token_loss +
            self.phrase_weight * phrase_loss +
            self.sentence_weight * sent_loss
        )

        loss_dict = {
            'token_loss': token_loss.item() if isinstance(token_loss, torch.Tensor) else token_loss,
            'phrase_loss': phrase_loss.item(),
            'sentence_loss': sent_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# Multiple Negatives Ranking Loss
# ============================================================================

class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss

    Query-Document 쌍 학습에 효과적
    """

    def __init__(
        self,
        temperature: float = 0.05,
        scale: float = 20.0
    ):
        super().__init__()
        self.temperature = temperature
        self.scale = scale

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: [batch_size, hidden_dim]
            document_embeddings: [batch_size, hidden_dim]

        Returns:
            loss: MNRL 손실
        """
        batch_size = query_embeddings.shape[0]
        device = query_embeddings.device

        # Normalize
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        doc_norm = F.normalize(document_embeddings, p=2, dim=1)

        # Similarity matrix
        scores = torch.mm(query_norm, doc_norm.T) * self.scale

        # Labels (diagonal is positive)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss
        loss = F.cross_entropy(scores, labels)

        return loss


# ============================================================================
# Utility Functions
# ============================================================================

def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    """임베딩 통계 계산"""
    with torch.no_grad():
        norms = torch.norm(embeddings, dim=1)
        similarities = torch.mm(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(embeddings, p=2, dim=1).T
        )
        # 대각선 제외
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        off_diag = similarities[mask]

    return {
        'mean_norm': norms.mean().item(),
        'std_norm': norms.std().item(),
        'mean_similarity': off_diag.mean().item(),
        'std_similarity': off_diag.std().item()
    }
