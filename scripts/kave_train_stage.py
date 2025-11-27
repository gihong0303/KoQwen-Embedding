#!/usr/bin/env python3
"""
KAVE Training Script for All Stages

Unified training script that handles:
- Stage 2: Easy token learning
- Stage 3: Medium token learning
- Stage 4: Hard token learning
- Stage 5: Unified training with PEU
- Stage 6: Retrieval fine-tuning

Usage:
    # Single GPU
    python scripts/kave_train_stage.py --stage 2 --config configs/kave_pipeline_config.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/kave_train_stage.py --stage 2 --config configs/kave_pipeline_config.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize process group with timeout
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )

        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class KAVETextDataset(Dataset):
    """Dataset for KAVE training"""

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        max_length: int = 512,
        token_filter: str = "all",
        token_categories_path: str = "outputs/token_difficulty/token_categories.json",
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load token categories for filtering
        self.token_ids_filter = set()
        if token_filter != "all" and Path(token_categories_path).exists():
            with open(token_categories_path, 'r') as f:
                categories = json.load(f)

            filter_types = token_filter.split(',')
            for ft in filter_types:
                ft = ft.strip()
                if ft in categories:
                    for token in categories[ft]:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id is not None:
                            self.token_ids_filter.add(token_id)

            logger.info(f"Token filter: {filter_types}, {len(self.token_ids_filter)} token IDs")

        # Load data
        for data_path in data_paths:
            if not Path(data_path).exists():
                logger.warning(f"Data file not found: {data_path}")
                continue

            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and len(self.samples) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())
                        text = item.get('text', item.get('content', ''))
                        if text:
                            # Filter by token presence if needed
                            if self.token_ids_filter:
                                tokens = tokenizer.encode(text, add_special_tokens=False)
                                if any(t in self.token_ids_filter for t in tokens):
                                    self.samples.append(text)
                            else:
                                self.samples.append(text)
                    except json.JSONDecodeError:
                        continue

            if max_samples and len(self.samples) >= max_samples:
                break

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class RetrievalDataset(Dataset):
    """Dataset for retrieval-aware training"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and len(self.samples) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())
                        query = item.get('query', '')
                        doc = item.get('document', item.get('positive', ''))
                        if query and doc:
                            self.samples.append({'query': query, 'document': doc})
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.samples)} retrieval pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        query_enc = self.tokenizer(
            sample['query'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        doc_enc = self.tokenizer(
            sample['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_ids': query_enc['input_ids'].squeeze(0),
            'query_mask': query_enc['attention_mask'].squeeze(0),
            'doc_ids': doc_enc['input_ids'].squeeze(0),
            'doc_mask': doc_enc['attention_mask'].squeeze(0)
        }


class KAVETrainer:
    """KAVE Training Loop"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: dict,
        stage_config: dict,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.stage_config = stage_config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Get embedding layer
        self.embed_tokens = model.get_input_embeddings()

        # KAVE config
        self.kave_config = config['kave']
        self.original_vocab_size = config['model']['original_vocab_size']
        self.new_vocab_size = len(tokenizer)

        # Loss weights
        self.loss_weights = stage_config.get('loss_weights', {
            'mlm': 1.0, 'contrastive': 0.5, 'alignment': 0.3, 'retrieval': 0.2
        })

        # Training params
        self.training_config = stage_config['training']

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # State
        self.global_step = 0
        self.current_epoch = 0

    def _create_optimizer(self):
        """Create optimizer with separate LR for embeddings"""
        embed_params = list(self.embed_tokens.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                       if 'embed' not in n and p.requires_grad]

        param_groups = [
            {'params': embed_params, 'lr': self.training_config['learning_rate'] * 2, 'weight_decay': 0.0},
            {'params': other_params, 'lr': self.training_config['learning_rate'],
             'weight_decay': self.training_config.get('weight_decay', 0.01)}
        ]

        return torch.optim.AdamW(param_groups)

    def _mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling with attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def compute_mlm_loss(self, input_ids, attention_mask):
        """Compute MLM loss focusing on new tokens"""
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Masking probability
        mlm_prob = self.kave_config['cta']['mlm_mask_prob']
        new_token_prob = self.kave_config['cta']['new_token_mask_prob']

        prob_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
        prob_matrix.masked_fill_(attention_mask == 0, 0.0)

        # Higher prob for new tokens
        new_token_mask = input_ids >= self.original_vocab_size
        prob_matrix = torch.where(
            new_token_mask,
            torch.full_like(prob_matrix, new_token_prob),
            prob_matrix
        )

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80% MASK
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=input_ids.device)
        ).bool() & masked_indices
        masked_input_ids[indices_replaced] = 0

        # Forward
        outputs = self.model(masked_input_ids, attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state

        # Predict
        logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.new_vocab_size), labels.view(-1))

        # Stats
        with torch.no_grad():
            mask = labels != -100
            if mask.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions == input_ids) & mask
                accuracy = correct.sum().float() / mask.sum().float()

                new_mask = new_token_mask & mask
                if new_mask.sum() > 0:
                    new_correct = correct & new_mask
                    new_accuracy = new_correct.sum().float() / new_mask.sum().float()
                else:
                    new_accuracy = torch.tensor(0.0, device=input_ids.device)
            else:
                accuracy = torch.tensor(0.0, device=input_ids.device)
                new_accuracy = torch.tensor(0.0, device=input_ids.device)

        return loss, accuracy.item(), new_accuracy.item()

    def compute_contrastive_loss(self, hidden_states, input_ids, attention_mask):
        """Token-level contrastive loss"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        temperature = self.kave_config['rat']['temperature']

        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_ids = input_ids.view(-1)
        flat_mask = attention_mask.view(-1)

        new_token_mask = (flat_ids >= self.original_vocab_size) & (flat_mask == 1)

        if new_token_mask.sum() < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        new_hidden = flat_hidden[new_token_mask]
        new_ids = flat_ids[new_token_mask]

        new_hidden = F.normalize(new_hidden, dim=1)
        sim_matrix = torch.mm(new_hidden, new_hidden.t()) / temperature

        labels = (new_ids.unsqueeze(0) == new_ids.unsqueeze(1)).float()
        mask = torch.eye(len(new_ids), device=labels.device).bool()
        labels.masked_fill_(mask, 0)

        if labels.sum() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        exp_sim = torch.exp(sim_matrix)
        exp_sim.masked_fill_(mask, 0)

        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        loss = -(log_prob * labels).sum() / (labels.sum() + 1e-9)

        return loss

    def compute_alignment_loss(self, input_ids):
        """Embedding alignment loss"""
        new_token_mask = input_ids >= self.original_vocab_size
        unique_new_ids = input_ids[new_token_mask].unique()

        if len(unique_new_ids) == 0:
            return torch.tensor(0.0, device=input_ids.device)

        new_embeds = self.embed_tokens.weight[unique_new_ids]
        old_embeds = self.embed_tokens.weight[:self.original_vocab_size]

        new_norm = F.normalize(new_embeds, dim=1)
        old_norm = F.normalize(old_embeds, dim=1)

        similarity = torch.mm(new_norm, old_norm.t())
        top_k_sim, _ = similarity.topk(k=5, dim=1)

        margin = 0.5
        loss = F.relu(margin - top_k_sim.mean(dim=1)).mean()

        return loss

    def compute_retrieval_loss(self, query_ids, query_mask, doc_ids, doc_mask):
        """Retrieval-aware loss"""
        temperature = self.kave_config['rat']['temperature']

        query_out = self.model(query_ids, query_mask, return_dict=True)
        doc_out = self.model(doc_ids, doc_mask, return_dict=True)

        query_embeds = self._mean_pooling(query_out.last_hidden_state, query_mask)
        doc_embeds = self._mean_pooling(doc_out.last_hidden_state, doc_mask)

        query_embeds = F.normalize(query_embeds, dim=1)
        doc_embeds = F.normalize(doc_embeds, dim=1)

        sim_matrix = torch.mm(query_embeds, doc_embeds.t()) / temperature
        labels = torch.arange(len(query_embeds), device=query_embeds.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def get_peu_gradient_mask(self, epoch: int):
        """Get PEU gradient mask"""
        if not self.stage_config.get('embedding', {}).get('peu_enabled', False):
            return None

        peu_config = self.kave_config['peu']
        start_epoch = peu_config['unfreeze_start_epoch']
        end_epoch = peu_config['unfreeze_end_epoch']

        if epoch < start_epoch:
            ratio = 0.0
        elif epoch >= end_epoch:
            ratio = 1.0
        else:
            ratio = (epoch - start_epoch) / (end_epoch - start_epoch)

        mask = torch.ones(self.new_vocab_size, device=self.device)

        if ratio == 0.0:
            mask[:self.original_vocab_size] = 0.0
        elif ratio < 1.0:
            # Partially unfreeze based on similarity
            with torch.no_grad():
                old_embeds = self.embed_tokens.weight[:self.original_vocab_size]
                new_embeds = self.embed_tokens.weight[self.original_vocab_size:]

                if len(new_embeds) > 0:
                    old_norm = F.normalize(old_embeds, dim=1)
                    new_norm = F.normalize(new_embeds, dim=1)

                    similarity = torch.mm(old_norm, new_norm.t())
                    max_sim = similarity.max(dim=1)[0]

                    threshold = peu_config['similarity_threshold'] * (1 - ratio) + 0.3 * ratio
                    mask[:self.original_vocab_size] = (max_sim > threshold).float()

        return mask

    def train_epoch(self, train_dataloader, retrieval_dataloader=None, epoch=0):
        """Train one epoch"""
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        total_mlm_acc = 0.0
        total_new_acc = 0.0
        num_steps = 0

        # PEU mask
        peu_mask = self.get_peu_gradient_mask(epoch)

        # Retrieval iterator
        retrieval_iter = iter(retrieval_dataloader) if retrieval_dataloader else None

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}",
            disable=self.rank != 0
        )

        grad_accum_steps = self.training_config.get('gradient_accumulation_steps', 1)

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # 1. MLM Loss
            mlm_loss, mlm_acc, new_acc = self.compute_mlm_loss(input_ids, attention_mask)

            # 2. Contrastive Loss
            outputs = self.model(input_ids, attention_mask, return_dict=True)
            contrastive_loss = self.compute_contrastive_loss(
                outputs.last_hidden_state, input_ids, attention_mask
            )

            # 3. Alignment Loss
            alignment_loss = self.compute_alignment_loss(input_ids)

            # 4. Retrieval Loss
            if retrieval_iter:
                try:
                    ret_batch = next(retrieval_iter)
                except StopIteration:
                    retrieval_iter = iter(retrieval_dataloader)
                    ret_batch = next(retrieval_iter)

                retrieval_loss = self.compute_retrieval_loss(
                    ret_batch['query_ids'].to(self.device),
                    ret_batch['query_mask'].to(self.device),
                    ret_batch['doc_ids'].to(self.device),
                    ret_batch['doc_mask'].to(self.device)
                )
            else:
                retrieval_loss = torch.tensor(0.0, device=self.device)

            # Combined loss
            loss = (
                self.loss_weights['mlm'] * mlm_loss +
                self.loss_weights['contrastive'] * contrastive_loss +
                self.loss_weights['alignment'] * alignment_loss +
                self.loss_weights['retrieval'] * retrieval_loss
            )

            # Scale for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            # Apply PEU mask
            if peu_mask is not None and self.embed_tokens.weight.grad is not None:
                self.embed_tokens.weight.grad *= peu_mask.unsqueeze(1)

            # Update
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.get('max_grad_norm', 1.0)
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Stats
            total_loss += loss.item() * grad_accum_steps
            total_mlm_acc += mlm_acc
            total_new_acc += new_acc
            num_steps += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item() * grad_accum_steps:.4f}",
                'mlm_acc': f"{mlm_acc:.3f}",
                'new_acc': f"{new_acc:.3f}"
            })

        return {
            'loss': total_loss / num_steps,
            'mlm_accuracy': total_mlm_acc / num_steps,
            'new_token_accuracy': total_new_acc / num_steps
        }

    def save(self, output_dir: str):
        """Save model and tokenizer"""
        if self.rank != 0:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get model without DDP
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch
        }
        with open(output_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KAVE Training")
    parser.add_argument("--stage", type=int, required=True, help="Training stage (2-6)")
    parser.add_argument("--config", type=str, default="configs/kave_pipeline_config.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"KAVE Stage {args.stage} Training")
        logger.info(f"World size: {world_size}")

    # Load config
    config = load_config(args.config)

    # Get stage config
    stage_key = f"stage{args.stage}_{'easy' if args.stage == 2 else 'medium' if args.stage == 3 else 'hard' if args.stage == 4 else 'unify' if args.stage == 5 else 'retrieval'}"
    stage_config = config.get(stage_key, config.get(f"stage{args.stage}"))

    if stage_config is None:
        logger.error(f"Stage {args.stage} config not found")
        return

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    elif args.stage == 2:
        model_path = config['model']['initialized_model_path']
    else:
        prev_stage = args.stage - 1
        prev_key = f"stage{prev_stage}_{'easy' if prev_stage == 2 else 'medium' if prev_stage == 3 else 'hard' if prev_stage == 4 else 'unify' if prev_stage == 5 else 'retrieval'}"
        prev_config = config.get(prev_key, config.get(f"stage{prev_stage}"))
        model_path = prev_config['checkpoint']['save_dir'] + "/final"

    if rank == 0:
        logger.info(f"Loading model from: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['tokenizer_path'],
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config['hardware'].get('mixed_precision') == 'bf16' else torch.float32
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Create datasets
    data_config = stage_config['data']
    data_paths = [s['path'] for s in data_config['sources']]

    train_dataset = KAVETextDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        token_filter=data_config.get('token_filter', 'all'),
        max_samples=data_config.get('max_samples')
    )

    sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=stage_config['training']['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )

    # Retrieval dataloader (for stages 5-6)
    retrieval_dataloader = None
    if args.stage >= 5:
        retrieval_paths = [s['path'] for s in data_config['sources'] if 'retrieval' in s['path']]
        if retrieval_paths:
            retrieval_dataset = RetrievalDataset(
                data_path=retrieval_paths[0],
                tokenizer=tokenizer,
                max_samples=data_config.get('max_samples', 100000) // 2
            )
            retrieval_dataloader = DataLoader(
                retrieval_dataset,
                batch_size=stage_config['training']['batch_size'] // 2,
                shuffle=True,
                num_workers=2
            )

    # Create trainer
    trainer = KAVETrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage_config=stage_config,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size
    )

    # Training loop
    num_epochs = stage_config['training']['num_epochs']

    for epoch in range(num_epochs):
        if sampler:
            sampler.set_epoch(epoch)

        stats = trainer.train_epoch(train_dataloader, retrieval_dataloader, epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: {stats}")

        # Save checkpoint
        if rank == 0:
            trainer.save(f"{stage_config['checkpoint']['save_dir']}/epoch_{epoch+1}")

    # Save final
    if rank == 0:
        trainer.save(f"{stage_config['checkpoint']['save_dir']}/final")

    cleanup_distributed()

    if rank == 0:
        logger.info(f"Stage {args.stage} training complete!")


if __name__ == "__main__":
    main()
