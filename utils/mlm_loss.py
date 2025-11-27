#!/usr/bin/env python3
"""
MLM-style Loss for Token Embedding Learning

SimCSE는 문장 표현 학습용이라 토큰 임베딩 학습에 부적합.
MLM은 context 기반으로 토큰을 예측하므로 토큰 임베딩 학습에 적합.

핵심 아이디어:
- 새 토큰을 마스킹하고 예측하도록 학습
- 기존 토큰은 freeze하고 새 토큰만 학습
- Context를 활용하므로 의미적 임베딩 학습 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import random


class MLMTokenLoss(nn.Module):
    """
    MLM-style loss for learning new token embeddings

    새 토큰에 대해서만 MLM을 적용하여 토큰 임베딩 학습
    """

    def __init__(
        self,
        vocab_size: int,
        old_vocab_size: int = 151669,
        mask_prob: float = 0.15,
        new_token_focus: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.old_vocab_size = old_vocab_size
        self.mask_prob = mask_prob
        self.new_token_focus = new_token_focus

        # MLM prediction head (will be set from model)
        self.mlm_head = None

    def create_mlm_head(self, hidden_size: int, device: torch.device):
        """Create MLM prediction head"""
        self.mlm_head = nn.Linear(hidden_size, self.vocab_size, bias=False).to(device)

        # Initialize with embedding weights (tied embeddings)
        return self.mlm_head

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for MLM training

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            special_tokens_mask: [batch_size, seq_len] - 1 for special tokens

        Returns:
            masked_input_ids: Input with some tokens replaced by [MASK]
            labels: Original token ids for masked positions, -100 elsewhere
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mask_prob, device=input_ids.device)

        # Don't mask special tokens
        if special_tokens_mask is not None:
            probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

        # Don't mask padding
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        # Focus on new tokens if enabled
        if self.new_token_focus:
            # Increase masking probability for new tokens
            new_token_mask = input_ids >= self.old_vocab_size
            probability_matrix = torch.where(
                new_token_mask,
                torch.full_like(probability_matrix, 0.5),  # 50% for new tokens
                probability_matrix * 0.5  # Reduce for old tokens
            )

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: -100 for non-masked positions (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of time: replace with [MASK] token (use a placeholder ID)
        # 10% of time: replace with random token
        # 10% of time: keep original

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        # Use 0 as mask token (or you can use a specific mask token ID)
        masked_input_ids[indices_replaced] = 0

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
        masked_input_ids[indices_random] = random_words[indices_random]

        return masked_input_ids, labels

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embed_tokens: nn.Embedding
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for MLM loss

        Args:
            model: The transformer model
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            embed_tokens: Embedding layer for predictions

        Returns:
            loss: MLM loss
            stats: Dictionary with statistics
        """
        # Create masked inputs
        masked_input_ids, labels = self.mask_tokens(input_ids, attention_mask)

        # Forward pass
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        # Predict using embedding weights (tied)
        # logits = hidden @ embed_weight.T
        logits = torch.matmul(hidden_states, embed_tokens.weight.t())  # [batch, seq, vocab]

        # Calculate loss only on masked positions
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        # Statistics
        with torch.no_grad():
            mask = labels != -100
            if mask.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions == input_ids) & mask
                accuracy = correct.sum().float() / mask.sum().float()

                # New token specific stats
                new_token_mask = (input_ids >= self.old_vocab_size) & mask
                if new_token_mask.sum() > 0:
                    new_correct = correct & new_token_mask
                    new_accuracy = new_correct.sum().float() / new_token_mask.sum().float()
                else:
                    new_accuracy = torch.tensor(0.0)
            else:
                accuracy = torch.tensor(0.0)
                new_accuracy = torch.tensor(0.0)

        stats = {
            'total_masked': mask.sum().item(),
            'accuracy': accuracy.item(),
            'new_token_accuracy': new_accuracy.item()
        }

        return loss, stats


class TokenContrastiveLoss(nn.Module):
    """
    Token-level Contrastive Loss

    토큰 임베딩끼리 직접 contrastive learning
    - Positive: 같은 토큰의 다른 context에서의 표현
    - Negative: 다른 토큰들
    """

    def __init__(
        self,
        temperature: float = 0.1,
        old_vocab_size: int = 151669
    ):
        super().__init__()
        self.temperature = temperature
        self.old_vocab_size = old_vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Token-level contrastive loss

        같은 토큰이 다른 위치에서 나타나면 positive pair
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden]
        flat_ids = input_ids.view(-1)  # [batch*seq]
        flat_mask = attention_mask.view(-1)  # [batch*seq]

        # Only consider new tokens
        new_token_mask = (flat_ids >= self.old_vocab_size) & (flat_mask == 1)

        if new_token_mask.sum() < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        # Get new token hidden states
        new_hidden = flat_hidden[new_token_mask]  # [N, hidden]
        new_ids = flat_ids[new_token_mask]  # [N]

        # Normalize
        new_hidden = F.normalize(new_hidden, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(new_hidden, new_hidden.t()) / self.temperature

        # Labels: same token = positive
        labels = (new_ids.unsqueeze(0) == new_ids.unsqueeze(1)).float()

        # Remove diagonal (self-similarity)
        mask = torch.eye(len(new_ids), device=labels.device).bool()
        labels.masked_fill_(mask, 0)

        # If no positive pairs, return 0
        if labels.sum() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        # InfoNCE-style loss
        # For each token, maximize similarity to same tokens, minimize to others
        exp_sim = torch.exp(sim_matrix)
        exp_sim.masked_fill_(mask, 0)  # Remove self

        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Average over positive pairs
        loss = -(log_prob * labels).sum() / (labels.sum() + 1e-9)

        return loss
