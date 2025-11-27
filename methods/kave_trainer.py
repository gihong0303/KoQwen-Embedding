#!/usr/bin/env python3
"""
KAVE Trainer: Korean Adaptive Vocabulary Expansion Training Loop

Integrates all KAVE components:
- WSA: Weighted Semantic Averaging (initialization)
- CTA: Contextual Token Alignment (loss)
- PEU: Progressive Embedding Unfreezing (gradient control)
- RAT: Retrieval-Aware Training (auxiliary loss)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods.kave_framework import (
    KAVEConfig,
    KAVEFramework,
    WeightedSemanticAveraging,
    ContextualTokenAlignment,
    ProgressiveEmbeddingUnfreezing,
    RetrievalAwareTraining
)

logger = logging.getLogger(__name__)


@dataclass
class KAVETrainingArgs:
    """Training arguments for KAVE"""
    # Model
    model_path: str = "outputs/kave-initialized"
    tokenizer_path: str = "outputs/koqwen-expanded"
    original_vocab_size: int = 151669

    # Training
    num_epochs: int = 1
    batch_size: int = 48
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Loss weights
    mlm_weight: float = 1.0
    contrastive_weight: float = 0.5
    alignment_weight: float = 0.3
    retrieval_weight: float = 0.2

    # PEU settings
    freeze_old_embeddings: bool = True
    peu_enabled: bool = True
    unfreeze_start_epoch: int = 3
    unfreeze_end_epoch: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints/kave"
    save_steps: int = 1000
    log_steps: int = 100

    # Distributed
    local_rank: int = -1


class KAVETrainer:
    """
    KAVE Training Pipeline

    Handles the complete training loop with all KAVE components
    """

    def __init__(
        self,
        args: KAVETrainingArgs,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        retrieval_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.retrieval_dataloader = retrieval_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = args.local_rank >= 0

        if self.is_distributed:
            self.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(self.device)

        # Initialize KAVE framework
        self.kave_config = KAVEConfig(
            hidden_size=model.config.hidden_size,
            original_vocab_size=args.original_vocab_size,
            new_vocab_size=len(tokenizer),
            mlm_weight=args.mlm_weight,
            contrastive_weight=args.contrastive_weight,
            alignment_weight=args.alignment_weight,
            retrieval_loss_weight=args.retrieval_weight,
            freeze_old_embeddings=args.freeze_old_embeddings,
            unfreeze_start_epoch=args.unfreeze_start_epoch,
            unfreeze_end_epoch=args.unfreeze_end_epoch
        )

        self.kave = KAVEFramework(self.kave_config)

        # Get embedding layer
        self.embed_tokens = model.get_input_embeddings()

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different LR for embeddings"""
        # Separate parameters
        embed_params = list(self.embed_tokens.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                       if 'embed' not in n and p.requires_grad]

        param_groups = [
            {
                'params': embed_params,
                'lr': self.args.learning_rate * 2,  # Higher LR for new embeddings
                'weight_decay': 0.0  # No weight decay for embeddings
            },
            {
                'params': other_params,
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay
            }
        ]

        return torch.optim.AdamW(param_groups)

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling with attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def compute_mlm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute MLM loss with focus on new tokens"""
        # Create masked inputs
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Probability matrix
        prob_matrix = torch.full(
            labels.shape,
            self.kave_config.mlm_mask_prob,
            device=input_ids.device
        )

        # Don't mask padding
        prob_matrix.masked_fill_(attention_mask == 0, 0.0)

        # Higher probability for new tokens
        new_token_mask = input_ids >= self.kave_config.original_vocab_size
        prob_matrix = torch.where(
            new_token_mask,
            torch.full_like(prob_matrix, self.kave_config.new_token_mask_prob),
            prob_matrix
        )

        # Sample masked indices
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80% MASK, 10% random, 10% keep
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=input_ids.device)
        ).bool() & masked_indices
        masked_input_ids[indices_replaced] = 0

        # Forward pass
        outputs = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state

        # Predict using tied embeddings
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())

        # MLM loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, self.kave_config.new_vocab_size),
            labels.view(-1)
        )

        # Calculate accuracy
        with torch.no_grad():
            mask = labels != -100
            if mask.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions == input_ids) & mask
                accuracy = correct.sum().float() / mask.sum().float()

                # New token accuracy
                new_mask = new_token_mask & mask
                if new_mask.sum() > 0:
                    new_correct = correct & new_mask
                    new_accuracy = new_correct.sum().float() / new_mask.sum().float()
                else:
                    new_accuracy = torch.tensor(0.0)
            else:
                accuracy = torch.tensor(0.0)
                new_accuracy = torch.tensor(0.0)

        stats = {
            'mlm_loss': loss.item(),
            'mlm_accuracy': accuracy.item(),
            'new_token_accuracy': new_accuracy.item(),
            'total_masked': mask.sum().item()
        }

        return loss, stats

    def compute_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Token-level contrastive loss for new tokens"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_ids = input_ids.view(-1)
        flat_mask = attention_mask.view(-1)

        # Focus on new tokens
        new_token_mask = (flat_ids >= self.kave_config.original_vocab_size) & (flat_mask == 1)

        if new_token_mask.sum() < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        new_hidden = flat_hidden[new_token_mask]
        new_ids = flat_ids[new_token_mask]

        # Normalize
        new_hidden = F.normalize(new_hidden, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(new_hidden, new_hidden.t()) / self.kave_config.temperature

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

    def compute_alignment_loss(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Embedding alignment loss for new tokens"""
        # Get unique new token IDs in batch
        new_token_mask = input_ids >= self.kave_config.original_vocab_size
        unique_new_ids = input_ids[new_token_mask].unique()

        if len(unique_new_ids) == 0:
            return torch.tensor(0.0, device=input_ids.device)

        # Get embeddings
        new_embeds = self.embed_tokens.weight[unique_new_ids]
        old_embeds = self.embed_tokens.weight[:self.kave_config.original_vocab_size]

        # Normalize
        new_norm = F.normalize(new_embeds, dim=1)
        old_norm = F.normalize(old_embeds, dim=1)

        # Find top-k similar old tokens
        similarity = torch.mm(new_norm, old_norm.t())
        top_k_sim, _ = similarity.topk(k=5, dim=1)

        # Margin-based loss: pull toward semantic neighbors
        margin = 0.5
        loss = F.relu(margin - top_k_sim.mean(dim=1)).mean()

        return loss

    def compute_retrieval_loss(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_mask: torch.Tensor
    ) -> torch.Tensor:
        """Retrieval-aware contrastive loss"""
        # Get embeddings
        query_out = self.model(query_ids, query_mask, return_dict=True)
        doc_out = self.model(doc_ids, doc_mask, return_dict=True)

        # Mean pooling
        query_embeds = self._mean_pooling(query_out.last_hidden_state, query_mask)
        doc_embeds = self._mean_pooling(doc_out.last_hidden_state, doc_mask)

        # Normalize
        query_embeds = F.normalize(query_embeds, dim=1)
        doc_embeds = F.normalize(doc_embeds, dim=1)

        # In-batch negatives
        sim_matrix = torch.mm(query_embeds, doc_embeds.t()) / self.kave_config.temperature

        # Labels: positive on diagonal
        labels = torch.arange(len(query_embeds), device=query_embeds.device)

        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        retrieval_batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Complete KAVE training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # 1. MLM Loss
        mlm_loss, mlm_stats = self.compute_mlm_loss(input_ids, attention_mask)

        # 2. Contrastive Loss (need forward without masking)
        outputs = self.model(input_ids, attention_mask, return_dict=True)
        contrastive_loss = self.compute_contrastive_loss(
            outputs.last_hidden_state, input_ids, attention_mask
        )

        # 3. Alignment Loss
        alignment_loss = self.compute_alignment_loss(input_ids)

        # 4. Retrieval Loss (if retrieval batch provided)
        if retrieval_batch is not None:
            retrieval_loss = self.compute_retrieval_loss(
                retrieval_batch['query_ids'].to(self.device),
                retrieval_batch['query_mask'].to(self.device),
                retrieval_batch['doc_ids'].to(self.device),
                retrieval_batch['doc_mask'].to(self.device)
            )
        else:
            retrieval_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = (
            self.args.mlm_weight * mlm_loss +
            self.args.contrastive_weight * contrastive_loss +
            self.args.alignment_weight * alignment_loss +
            self.args.retrieval_weight * retrieval_loss
        )

        stats = {
            **mlm_stats,
            'contrastive_loss': contrastive_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'retrieval_loss': retrieval_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, stats

    def apply_peu_gradient_mask(self):
        """Apply PEU gradient mask after backward"""
        if not self.args.peu_enabled:
            return

        grad_mask = self.kave.peu.create_embedding_mask(
            self.embed_tokens,
            self.current_epoch
        )

        if self.embed_tokens.weight.grad is not None:
            self.embed_tokens.weight.grad *= grad_mask.unsqueeze(1)

    def train(self):
        """Main training loop"""
        num_training_steps = (
            len(self.train_dataloader) *
            self.args.num_epochs //
            self.args.gradient_accumulation_steps
        )

        scheduler = self._create_scheduler(num_training_steps)

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # DDP wrapper
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=False
            )

        # Retrieval data iterator
        retrieval_iter = None
        if self.retrieval_dataloader:
            retrieval_iter = iter(self.retrieval_dataloader)

        # Training loop
        logger.info(f"Starting KAVE training for {self.args.num_epochs} epochs")
        logger.info(f"Total steps: {num_training_steps}")

        checkpoint_dir = Path(self.args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_stats = {}

            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.args.num_epochs}",
                disable=self.args.local_rank > 0
            )

            for step, batch in enumerate(progress_bar):
                # Get retrieval batch if available
                retrieval_batch = None
                if retrieval_iter:
                    try:
                        retrieval_batch = next(retrieval_iter)
                    except StopIteration:
                        retrieval_iter = iter(self.retrieval_dataloader)
                        retrieval_batch = next(retrieval_iter)

                # Training step
                loss, stats = self.training_step(batch, retrieval_batch)

                # Scale loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                # Apply PEU gradient mask
                self.apply_peu_gradient_mask()

                # Update weights
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update stats
                epoch_loss += stats['total_loss']
                for k, v in stats.items():
                    epoch_stats[k] = epoch_stats.get(k, 0) + v

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{stats['total_loss']:.4f}",
                    'mlm_acc': f"{stats['mlm_accuracy']:.3f}",
                    'new_acc': f"{stats['new_token_accuracy']:.3f}"
                })

                # Logging
                if self.global_step % self.args.log_steps == 0:
                    unfreeze_ratio = self.kave.peu.get_unfreeze_ratio(epoch)
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={stats['total_loss']:.4f}, "
                        f"mlm={stats['mlm_loss']:.4f}, "
                        f"contrast={stats['contrastive_loss']:.4f}, "
                        f"unfreeze={unfreeze_ratio:.1%}"
                    )

                # Checkpointing
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            # Epoch summary
            avg_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

        # Save final model
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        if self.args.local_rank > 0:
            return  # Only save on rank 0

        checkpoint_path = Path(self.args.checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Get model without DDP wrapper
        model_to_save = self.model.module if self.is_distributed else self.model

        # Save model and tokenizer
        model_to_save.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'args': vars(self.args)
        }
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval dataset"""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask, return_dict=True)

                # Simple perplexity calculation
                logits = torch.matmul(
                    outputs.last_hidden_state,
                    self.embed_tokens.weight.t()
                )

                # Shift for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss.view(shift_labels.size()) * shift_mask

                total_loss += loss.sum().item()
                total_tokens += shift_mask.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity
        }


def main():
    """Example usage"""
    print("KAVE Trainer - Example Usage")
    print("=" * 60)
    print("""
    # Initialize model and tokenizer
    model = AutoModel.from_pretrained("outputs/kave-initialized")
    tokenizer = AutoTokenizer.from_pretrained("outputs/koqwen-expanded")

    # Create training args
    args = KAVETrainingArgs(
        model_path="outputs/kave-initialized",
        num_epochs=3,
        batch_size=48,
        learning_rate=2e-5
    )

    # Create dataloaders
    train_dataloader = ...
    retrieval_dataloader = ...

    # Initialize trainer
    trainer = KAVETrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        retrieval_dataloader=retrieval_dataloader
    )

    # Train!
    trainer.train()
    """)


if __name__ == "__main__":
    main()
