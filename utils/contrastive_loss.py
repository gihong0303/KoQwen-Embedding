"""
Contrastive Learning Loss Functions for Embedding Models
"""

import torch
import torch.nn.functional as F


def mean_pooling(hidden_states, attention_mask):
    """
    Mean pooling with attention mask

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]

    Returns:
        pooled: [batch_size, hidden_dim]
    """
    # Expand mask to match hidden_states dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    # Sum masked hidden states
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)

    # Sum of mask (to get mean)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

    return sum_embeddings / sum_mask


def simcse_loss(embeddings1, embeddings2, temperature=0.05):
    """
    SimCSE Unsupervised Contrastive Loss

    두 번의 forward pass로 얻은 임베딩을 대조 학습
    (같은 문장, 다른 dropout)

    Args:
        embeddings1: [batch_size, hidden_dim] - First pass
        embeddings2: [batch_size, hidden_dim] - Second pass
        temperature: Temperature for scaling

    Returns:
        loss: Scalar contrastive loss
    """
    batch_size = embeddings1.shape[0]

    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Cosine similarity matrix
    # [batch_size, batch_size]
    cos_sim = torch.mm(embeddings1, embeddings2.t()) / temperature

    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=embeddings1.device)

    # Cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(cos_sim, labels)

    return loss


def in_batch_negatives_loss(query_embeddings, key_embeddings, temperature=0.05):
    """
    In-batch Negatives Contrastive Loss

    Query와 Key 임베딩 간의 대조 학습
    배치 내의 다른 샘플들을 negative로 사용

    Args:
        query_embeddings: [batch_size, hidden_dim]
        key_embeddings: [batch_size, hidden_dim]
        temperature: Temperature scaling

    Returns:
        loss: Contrastive loss
    """
    batch_size = query_embeddings.shape[0]

    # Normalize
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    key_embeddings = F.normalize(key_embeddings, p=2, dim=1)

    # Similarity matrix
    sim_matrix = torch.mm(query_embeddings, key_embeddings.t()) / temperature

    # Labels (positive pairs are on diagonal)
    labels = torch.arange(batch_size, device=query_embeddings.device)

    loss = F.cross_entropy(sim_matrix, labels)

    return loss


class ContrastiveLoss(torch.nn.Module):
    """
    Unified Contrastive Loss Module
    """

    def __init__(self, temperature=0.05, pooling="mean"):
        super().__init__()
        self.temperature = temperature
        self.pooling = pooling

    def forward(self, model, input_ids, attention_mask):
        """
        Forward pass with two dropout passes (SimCSE style)

        Args:
            model: Embedding model
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            loss: Contrastive loss
            embeddings: Mean pooled embeddings
        """
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
        if self.pooling == "mean":
            emb1 = mean_pooling(outputs1.last_hidden_state, attention_mask)
            emb2 = mean_pooling(outputs2.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            emb1 = outputs1.last_hidden_state[:, 0, :]
            emb2 = outputs2.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # SimCSE loss
        loss = simcse_loss(emb1, emb2, temperature=self.temperature)

        return loss, emb1


def compute_embedding_stats(model, tokenizer, device="cuda"):
    """
    새 토큰 임베딩의 통계 정보 계산

    Args:
        model: Embedding model
        tokenizer: Tokenizer
        device: Device

    Returns:
        stats: Dictionary with statistics
    """
    embed_tokens = model.get_input_embeddings()
    vocab_size = len(tokenizer)

    with torch.no_grad():
        # 전체 임베딩 norm
        all_norms = torch.norm(embed_tokens.weight, dim=1)

        # 새 토큰 범위 (기존 Qwen vocab: 151,669)
        old_vocab_size = 151669
        new_token_mask = torch.arange(vocab_size, device=device) >= old_vocab_size

        # 기존 토큰 vs 새 토큰 norm 비교
        old_norms = all_norms[~new_token_mask]
        new_norms = all_norms[new_token_mask]

        stats = {
            'old_tokens_mean_norm': old_norms.mean().item(),
            'old_tokens_std_norm': old_norms.std().item(),
            'new_tokens_mean_norm': new_norms.mean().item() if len(new_norms) > 0 else 0.0,
            'new_tokens_std_norm': new_norms.std().item() if len(new_norms) > 0 else 0.0,
            'vocab_size': vocab_size,
            'new_tokens_count': new_token_mask.sum().item()
        }

    return stats
