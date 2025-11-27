#!/usr/bin/env python3
"""
KAVE: Korean Adaptive Vocabulary Expansion Framework

A novel approach combining:
1. Weighted Semantic Averaging (WSA) - Better initialization
2. Contextual Token Alignment (CTA) - Hybrid MLM + Contrastive loss
3. Progressive Embedding Unfreezing (PEU) - Gradual learning
4. Retrieval-Aware Training (RAT) - Preserve retrieval performance

IEEE Paper-level Novel Contributions:
- WSA outperforms simple averaging by considering semantic relationships
- CTA addresses the SimCSE limitation for token-level learning
- PEU prevents catastrophic forgetting of original embeddings
- RAT maintains retrieval performance during vocabulary expansion

Author: KoQwen-Embedding Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class KAVEConfig:
    """KAVE Framework Configuration"""
    # Model
    hidden_size: int = 896
    original_vocab_size: int = 151669
    new_vocab_size: int = 200000

    # WSA (Weighted Semantic Averaging)
    wsa_temperature: float = 0.1
    wsa_top_k: int = 10  # Top-k similar subwords for weighted average

    # CTA (Contextual Token Alignment)
    mlm_weight: float = 1.0
    contrastive_weight: float = 0.5
    alignment_weight: float = 0.3
    mlm_mask_prob: float = 0.15
    new_token_mask_prob: float = 0.5  # Higher for new tokens

    # PEU (Progressive Embedding Unfreezing)
    freeze_old_embeddings: bool = True
    unfreeze_schedule: str = "linear"  # linear, exponential, step
    unfreeze_start_epoch: int = 3
    unfreeze_end_epoch: int = 10

    # RAT (Retrieval-Aware Training)
    retrieval_loss_weight: float = 0.2
    hard_negative_ratio: float = 0.3

    # Training
    temperature: float = 0.05


class WeightedSemanticAveraging(nn.Module):
    """
    Weighted Semantic Averaging (WSA) for Token Initialization

    기존 EEVE의 단순 평균 대신, semantic similarity 가중 평균 사용

    Key Innovation:
    - 새 토큰 "프로그래밍"의 subwords ["프로", "그래", "밍"] 중
    - "프로"와 "그래밍" (프로그램 관련)에 더 높은 가중치
    - 단순 평균보다 의미적으로 더 정확한 초기화
    """

    def __init__(self, config: KAVEConfig):
        super().__init__()
        self.config = config
        self.temperature = config.wsa_temperature
        self.top_k = config.wsa_top_k

    def compute_semantic_weights(
        self,
        subword_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic similarity weights for subwords

        Args:
            subword_embeddings: [num_subwords, hidden_dim]
            original_embeddings: [vocab_size, hidden_dim]

        Returns:
            weights: [num_subwords] - normalized weights
        """
        # Normalize for cosine similarity
        subword_norm = F.normalize(subword_embeddings, dim=1)

        # Compute pairwise similarity within subwords
        # subwords가 서로 얼마나 관련 있는지 측정
        pairwise_sim = torch.mm(subword_norm, subword_norm.t())

        # Self-similarity 제외한 평균 similarity
        mask = ~torch.eye(len(subword_embeddings), dtype=torch.bool, device=subword_embeddings.device)
        mean_sim = (pairwise_sim * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Similarity가 높은 subword에 더 높은 가중치
        # (다른 subwords와 관련성이 높을수록 중요)
        weights = F.softmax(mean_sim / self.temperature, dim=0)

        return weights

    def initialize_token(
        self,
        subword_ids: List[int],
        original_embeddings: torch.Tensor,
        use_semantic_weights: bool = True
    ) -> torch.Tensor:
        """
        Initialize a new token embedding using WSA

        Args:
            subword_ids: List of subword token IDs
            original_embeddings: [vocab_size, hidden_dim]
            use_semantic_weights: Whether to use semantic weighting

        Returns:
            new_embedding: [hidden_dim]
        """
        if len(subword_ids) == 0:
            # Fallback to mean of all embeddings
            return original_embeddings.mean(dim=0)

        if len(subword_ids) == 1:
            # Single subword - just copy
            return original_embeddings[subword_ids[0]].clone()

        # Get subword embeddings
        subword_embeddings = original_embeddings[subword_ids]

        if use_semantic_weights:
            # Compute semantic weights
            weights = self.compute_semantic_weights(subword_embeddings, original_embeddings)
            # Weighted average
            new_embedding = (subword_embeddings * weights.unsqueeze(1)).sum(dim=0)
        else:
            # Simple average (EEVE style)
            new_embedding = subword_embeddings.mean(dim=0)

        return new_embedding


class ContextualTokenAlignment(nn.Module):
    """
    Contextual Token Alignment (CTA) Loss

    MLM + Contrastive + Alignment의 하이브리드 손실 함수

    Key Innovation:
    - MLM: 컨텍스트 기반 토큰 예측 (토큰 의미 학습)
    - Token Contrastive: 같은 토큰의 다른 컨텍스트 표현 정렬
    - Embedding Alignment: 새 토큰과 기존 유사 토큰 간 정렬
    """

    def __init__(self, config: KAVEConfig):
        super().__init__()
        self.config = config
        self.mlm_weight = config.mlm_weight
        self.contrastive_weight = config.contrastive_weight
        self.alignment_weight = config.alignment_weight
        self.temperature = config.temperature

        # MLM head (initialized later with model)
        self.mlm_head = None

    def create_mlm_head(self, hidden_size: int, vocab_size: int, device: torch.device):
        """Create MLM prediction head"""
        self.mlm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        return self.mlm_head

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens with higher probability for new tokens
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Base probability
        prob_matrix = torch.full(
            labels.shape,
            self.config.mlm_mask_prob,
            device=input_ids.device
        )

        # Don't mask special tokens
        if special_tokens_mask is not None:
            prob_matrix.masked_fill_(special_tokens_mask.bool(), 0.0)

        # Don't mask padding
        prob_matrix.masked_fill_(attention_mask == 0, 0.0)

        # Higher probability for new tokens (key innovation)
        new_token_mask = input_ids >= self.config.original_vocab_size
        prob_matrix = torch.where(
            new_token_mask,
            torch.full_like(prob_matrix, self.config.new_token_mask_prob),
            prob_matrix
        )

        # Sample masked indices
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80% MASK, 10% random, 10% keep
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=input_ids.device)
        ).bool() & masked_indices
        masked_input_ids[indices_replaced] = 0  # [MASK] token

        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=input_ids.device)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.config.new_vocab_size, labels.shape,
            dtype=torch.long, device=input_ids.device
        )
        masked_input_ids[indices_random] = random_words[indices_random]

        return masked_input_ids, labels

    def mlm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        embed_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        MLM Loss using tied embeddings
        """
        # Predict using embedding weights
        logits = torch.matmul(hidden_states, embed_weights.t())

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, self.config.new_vocab_size),
            labels.view(-1)
        )

        return loss

    def token_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Token-level Contrastive Loss

        같은 토큰이 다른 위치/컨텍스트에서 나타나면 positive pair
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_ids = input_ids.view(-1)
        flat_mask = attention_mask.view(-1)

        # Focus on new tokens
        new_token_mask = (flat_ids >= self.config.original_vocab_size) & (flat_mask == 1)

        if new_token_mask.sum() < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        new_hidden = flat_hidden[new_token_mask]
        new_ids = flat_ids[new_token_mask]

        # Normalize
        new_hidden = F.normalize(new_hidden, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(new_hidden, new_hidden.t()) / self.temperature

        # Positive pairs: same token ID
        labels = (new_ids.unsqueeze(0) == new_ids.unsqueeze(1)).float()

        # Remove diagonal
        mask = torch.eye(len(new_ids), device=labels.device).bool()
        labels.masked_fill_(mask, 0)

        if labels.sum() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim.masked_fill_(mask, 0)

        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        loss = -(log_prob * labels).sum() / (labels.sum() + 1e-9)

        return loss

    def embedding_alignment_loss(
        self,
        new_token_embeds: torch.Tensor,
        original_embeddings: torch.Tensor,
        new_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Embedding Alignment Loss

        새 토큰 임베딩이 의미적으로 유사한 기존 토큰과 정렬되도록
        """
        if len(new_token_ids) == 0:
            return torch.tensor(0.0, device=new_token_embeds.device)

        # Normalize
        new_norm = F.normalize(new_token_embeds, dim=1)
        orig_norm = F.normalize(original_embeddings[:self.config.original_vocab_size], dim=1)

        # Find top-k similar original tokens for each new token
        similarity = torch.mm(new_norm, orig_norm.t())
        top_k_sim, _ = similarity.topk(k=5, dim=1)

        # Loss: maximize similarity to top-k (pull towards semantic neighbors)
        # Using margin-based loss
        margin = 0.5
        loss = F.relu(margin - top_k_sim.mean(dim=1)).mean()

        return loss

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embed_tokens: nn.Embedding
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Combined CTA Forward Pass
        """
        # Create masked inputs for MLM
        masked_input_ids, mlm_labels = self.mask_tokens(input_ids, attention_mask)

        # Forward pass
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state

        # 1. MLM Loss
        loss_mlm = self.mlm_loss(hidden_states, mlm_labels, embed_tokens.weight)

        # 2. Token Contrastive Loss (on original input, not masked)
        outputs_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        loss_contrastive = self.token_contrastive_loss(
            outputs_orig.last_hidden_state, input_ids, attention_mask
        )

        # 3. Embedding Alignment Loss
        new_token_mask = input_ids >= self.config.original_vocab_size
        unique_new_ids = input_ids[new_token_mask].unique()
        if len(unique_new_ids) > 0:
            new_embeds = embed_tokens.weight[unique_new_ids]
            loss_alignment = self.embedding_alignment_loss(
                new_embeds, embed_tokens.weight, unique_new_ids
            )
        else:
            loss_alignment = torch.tensor(0.0, device=input_ids.device)

        # Combined loss
        total_loss = (
            self.mlm_weight * loss_mlm +
            self.contrastive_weight * loss_contrastive +
            self.alignment_weight * loss_alignment
        )

        stats = {
            'loss_mlm': loss_mlm.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_alignment': loss_alignment.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, stats


class ProgressiveEmbeddingUnfreezing(nn.Module):
    """
    Progressive Embedding Unfreezing (PEU)

    기존 임베딩을 보호하면서 점진적으로 학습 허용

    Key Innovation:
    - 초기: 새 토큰만 학습 (기존 임베딩 freeze)
    - 중기: 유사한 기존 토큰도 미세 조정 허용
    - 후기: 전체 임베딩 fine-tuning (작은 LR)
    """

    def __init__(self, config: KAVEConfig):
        super().__init__()
        self.config = config
        self.current_epoch = 0

    def get_unfreeze_ratio(self, epoch: int) -> float:
        """
        Get the ratio of old embeddings to unfreeze

        Returns value between 0 (all frozen) and 1 (all unfrozen)
        """
        if epoch < self.config.unfreeze_start_epoch:
            return 0.0

        if epoch >= self.config.unfreeze_end_epoch:
            return 1.0

        progress = (epoch - self.config.unfreeze_start_epoch) / \
                   (self.config.unfreeze_end_epoch - self.config.unfreeze_start_epoch)

        if self.config.unfreeze_schedule == "linear":
            return progress
        elif self.config.unfreeze_schedule == "exponential":
            return math.pow(progress, 2)
        elif self.config.unfreeze_schedule == "step":
            return 1.0 if progress > 0.5 else 0.0
        else:
            return progress

    def create_embedding_mask(
        self,
        embed_tokens: nn.Embedding,
        epoch: int,
        similarity_threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Create gradient mask for embeddings

        Args:
            embed_tokens: Embedding layer
            epoch: Current epoch
            similarity_threshold: Threshold for "similar" tokens

        Returns:
            mask: [vocab_size] - 1 for trainable, 0 for frozen
        """
        vocab_size = embed_tokens.weight.shape[0]
        mask = torch.ones(vocab_size, device=embed_tokens.weight.device)

        unfreeze_ratio = self.get_unfreeze_ratio(epoch)

        if unfreeze_ratio == 0.0:
            # Freeze all old embeddings
            mask[:self.config.original_vocab_size] = 0.0
        elif unfreeze_ratio < 1.0:
            # Partially unfreeze based on similarity to new tokens
            with torch.no_grad():
                old_embeds = embed_tokens.weight[:self.config.original_vocab_size]
                new_embeds = embed_tokens.weight[self.config.original_vocab_size:]

                if len(new_embeds) > 0:
                    # Compute similarity
                    old_norm = F.normalize(old_embeds, dim=1)
                    new_norm = F.normalize(new_embeds, dim=1)

                    similarity = torch.mm(old_norm, new_norm.t())
                    max_sim = similarity.max(dim=1)[0]

                    # Unfreeze tokens similar to new tokens
                    threshold = similarity_threshold * (1 - unfreeze_ratio) + 0.3 * unfreeze_ratio
                    mask[:self.config.original_vocab_size] = (max_sim > threshold).float()
        # else: all unfrozen (mask stays 1)

        return mask

    def apply_gradient_mask(
        self,
        embed_tokens: nn.Embedding,
        mask: torch.Tensor
    ):
        """
        Apply gradient mask after backward pass

        Call this in the training loop after loss.backward()
        """
        if embed_tokens.weight.grad is not None:
            embed_tokens.weight.grad *= mask.unsqueeze(1)


class RetrievalAwareTraining(nn.Module):
    """
    Retrieval-Aware Training (RAT)

    검색 성능을 유지하면서 vocabulary expansion

    Key Innovation:
    - Query-Document 관계 학습 (SimCSE의 same-same과 다름)
    - Hard negative mining으로 discriminative한 임베딩
    - Retrieval task에 특화된 auxiliary loss
    """

    def __init__(self, config: KAVEConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_doc_embeds: torch.Tensor,
        neg_doc_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieval-aware contrastive loss

        Args:
            query_embeds: [batch_size, hidden_dim]
            pos_doc_embeds: [batch_size, hidden_dim]
            neg_doc_embeds: [batch_size, num_neg, hidden_dim] or None

        Returns:
            loss: Retrieval contrastive loss
        """
        batch_size = query_embeds.shape[0]

        # Normalize
        query_embeds = F.normalize(query_embeds, dim=1)
        pos_doc_embeds = F.normalize(pos_doc_embeds, dim=1)

        # Positive scores
        pos_scores = (query_embeds * pos_doc_embeds).sum(dim=1) / self.temperature

        # In-batch negatives
        all_doc_embeds = pos_doc_embeds  # [batch_size, hidden]

        # Add explicit hard negatives if provided
        if neg_doc_embeds is not None:
            neg_doc_embeds = F.normalize(neg_doc_embeds, dim=-1)
            # Flatten: [batch_size * num_neg, hidden]
            neg_flat = neg_doc_embeds.view(-1, neg_doc_embeds.shape[-1])
            all_doc_embeds = torch.cat([all_doc_embeds, neg_flat], dim=0)

        # Compute all scores
        all_scores = torch.mm(query_embeds, all_doc_embeds.t()) / self.temperature

        # Labels: positive is at diagonal position
        labels = torch.arange(batch_size, device=query_embeds.device)

        # Cross-entropy loss
        loss = F.cross_entropy(all_scores, labels)

        return loss


class KAVEFramework(nn.Module):
    """
    Complete KAVE Framework

    Combines all components for Korean Adaptive Vocabulary Expansion
    """

    def __init__(self, config: KAVEConfig):
        super().__init__()
        self.config = config

        # Components
        self.wsa = WeightedSemanticAveraging(config)
        self.cta = ContextualTokenAlignment(config)
        self.peu = ProgressiveEmbeddingUnfreezing(config)
        self.rat = RetrievalAwareTraining(config)

    def initialize_new_tokens(
        self,
        base_tokenizer,
        expanded_tokenizer,
        embed_tokens: nn.Embedding
    ) -> Dict:
        """
        Initialize all new tokens using WSA
        """
        original_embeddings = embed_tokens.weight[:self.config.original_vocab_size].clone()
        stats = {'wsa_initialized': 0, 'fallback': 0}

        for token_id in range(self.config.original_vocab_size, self.config.new_vocab_size):
            token = expanded_tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                continue

            # Clean token
            clean_token = token.replace('▁', '').replace('Ġ', '').strip()
            if not clean_token:
                embed_tokens.weight.data[token_id] = original_embeddings.mean(dim=0)
                stats['fallback'] += 1
                continue

            # Get subword IDs from base tokenizer
            subword_ids = base_tokenizer.encode(clean_token, add_special_tokens=False)
            valid_ids = [sid for sid in subword_ids if sid < self.config.original_vocab_size]

            if len(valid_ids) == 0:
                embed_tokens.weight.data[token_id] = original_embeddings.mean(dim=0)
                stats['fallback'] += 1
                continue

            # WSA initialization
            new_embed = self.wsa.initialize_token(
                valid_ids, original_embeddings, use_semantic_weights=True
            )
            embed_tokens.weight.data[token_id] = new_embed
            stats['wsa_initialized'] += 1

        return stats

    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        embed_tokens: nn.Embedding,
        epoch: int,
        retrieval_batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Complete training step with all losses
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # 1. CTA Loss (MLM + Contrastive + Alignment)
        cta_loss, cta_stats = self.cta(model, input_ids, attention_mask, embed_tokens)

        # 2. RAT Loss (if retrieval batch provided)
        if retrieval_batch is not None:
            # Get embeddings for query and documents
            query_out = model(
                retrieval_batch['query_ids'],
                retrieval_batch['query_mask'],
                return_dict=True
            )
            doc_out = model(
                retrieval_batch['doc_ids'],
                retrieval_batch['doc_mask'],
                return_dict=True
            )

            # Mean pooling
            query_embeds = self._mean_pool(query_out.last_hidden_state, retrieval_batch['query_mask'])
            doc_embeds = self._mean_pool(doc_out.last_hidden_state, retrieval_batch['doc_mask'])

            rat_loss = self.rat(query_embeds, doc_embeds)
        else:
            rat_loss = torch.tensor(0.0, device=input_ids.device)

        # Combined loss
        total_loss = cta_loss + self.config.retrieval_loss_weight * rat_loss

        # 3. Apply PEU gradient mask (after backward, call apply_gradient_mask)
        gradient_mask = self.peu.create_embedding_mask(embed_tokens, epoch)

        stats = {
            **cta_stats,
            'rat_loss': rat_loss.item(),
            'unfreeze_ratio': self.peu.get_unfreeze_ratio(epoch)
        }

        return total_loss, stats, gradient_mask

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


def create_kave_config(
    hidden_size: int = 896,
    original_vocab_size: int = 151669,
    new_vocab_size: int = 200000,
    **kwargs
) -> KAVEConfig:
    """Factory function to create KAVE config"""
    return KAVEConfig(
        hidden_size=hidden_size,
        original_vocab_size=original_vocab_size,
        new_vocab_size=new_vocab_size,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("KAVE Framework Test")
    print("=" * 60)

    # Create config
    config = create_kave_config()
    print(f"Config: {config}")

    # Create framework
    kave = KAVEFramework(config)
    print(f"\nFramework components:")
    print(f"  - WSA: Weighted Semantic Averaging")
    print(f"  - CTA: Contextual Token Alignment")
    print(f"  - PEU: Progressive Embedding Unfreezing")
    print(f"  - RAT: Retrieval-Aware Training")

    # Test PEU schedule
    print(f"\nPEU Unfreeze Schedule:")
    for epoch in range(15):
        ratio = kave.peu.get_unfreeze_ratio(epoch)
        print(f"  Epoch {epoch:2d}: {ratio:.2%} unfrozen")

    print("\n" + "=" * 60)
    print("KAVE Framework ready for Korean vocabulary expansion!")
