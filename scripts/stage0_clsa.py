#!/usr/bin/env python3
"""
Stage 0: Cross-lingual Semantic Anchoring (CLSA)
새 한국어 토큰을 다국어 의미 공간에 정렬
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModel, get_scheduler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_distributed():
    """DDP 초기화"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=10)
        )

        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"[DDP] world_size={world_size}, local_rank={local_rank}", flush=True)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class CLSADataset(Dataset):
    """
    Cross-lingual Semantic Anchoring Dataset

    Returns:
        {
            'korean_token_id': int,
            'anchor_token_ids': List[int],  # English + Chinese anchors
            'anchor_weights': List[float],  # Similarity weights
        }
    """

    def __init__(
        self,
        bilingual_dict_path: str,
        base_tokenizer,
        korean_tokenizer,
        min_anchors: int = 2
    ):
        # Load bilingual dictionary
        with open(bilingual_dict_path, 'r', encoding='utf-8') as f:
            self.bilingual_dict = json.load(f)

        self.base_tokenizer = base_tokenizer
        self.korean_tokenizer = korean_tokenizer

        # Filter tokens with sufficient anchors
        self.valid_tokens = [
            token for token, anchors in self.bilingual_dict.items()
            if len(anchors) >= min_anchors
        ]

        logging.info(f"CLSA Dataset: {len(self.valid_tokens):,} tokens with ≥{min_anchors} anchors")

    def __len__(self):
        return len(self.valid_tokens)

    def __getitem__(self, idx):
        ko_token = self.valid_tokens[idx]
        anchors = self.bilingual_dict[ko_token]

        # Get Korean token ID (from expanded tokenizer)
        ko_token_id = self.korean_tokenizer.convert_tokens_to_ids(ko_token)

        # Get anchor token IDs (from base tokenizer)
        anchor_token_ids = []
        anchor_weights = []

        for anchor_info in anchors:
            anchor = anchor_info['anchor']
            similarity = anchor_info['similarity']

            # Convert to token ID in base tokenizer
            anchor_id = self.base_tokenizer.convert_tokens_to_ids(anchor)

            anchor_token_ids.append(anchor_id)
            anchor_weights.append(similarity)

        return {
            'korean_token_id': ko_token_id,
            'anchor_token_ids': anchor_token_ids,
            'anchor_weights': anchor_weights
        }


def collate_clsa(batch):
    """Collate function for CLSA dataset"""
    korean_ids = torch.tensor([item['korean_token_id'] for item in batch])

    # Variable-length anchors - pad to max length in batch
    max_anchors = max(len(item['anchor_token_ids']) for item in batch)

    anchor_ids = []
    anchor_weights = []
    anchor_masks = []

    for item in batch:
        ids = item['anchor_token_ids']
        weights = item['anchor_weights']

        # Pad
        pad_len = max_anchors - len(ids)
        ids_padded = ids + [0] * pad_len
        weights_padded = weights + [0.0] * pad_len
        mask = [1] * len(ids) + [0] * pad_len

        anchor_ids.append(ids_padded)
        anchor_weights.append(weights_padded)
        anchor_masks.append(mask)

    return {
        'korean_token_ids': korean_ids,
        'anchor_token_ids': torch.tensor(anchor_ids),
        'anchor_weights': torch.tensor(anchor_weights),
        'anchor_masks': torch.tensor(anchor_masks)
    }


class CLSALoss(nn.Module):
    """
    Cross-lingual Semantic Anchoring Loss

    Loss = weighted_anchor_distance + diversity_regularization

    Aligns Korean token embeddings to weighted center of anchor embeddings
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        distance_type: str = 'cosine'  # 'cosine' or 'euclidean'
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.distance_type = distance_type

    def forward(
        self,
        korean_embeddings: torch.Tensor,  # [batch_size, hidden_dim]
        anchor_embeddings: torch.Tensor,  # [batch_size, num_anchors, hidden_dim]
        anchor_weights: torch.Tensor,     # [batch_size, num_anchors]
        anchor_masks: torch.Tensor        # [batch_size, num_anchors]
    ):
        """
        Compute CLSA loss

        Args:
            korean_embeddings: Korean token embeddings
            anchor_embeddings: Anchor token embeddings
            anchor_weights: Similarity weights (from bilingual dictionary)
            anchor_masks: Valid anchor mask

        Returns:
            loss: CLSA loss
        """
        batch_size = korean_embeddings.size(0)

        # Normalize embeddings
        korean_emb_norm = F.normalize(korean_embeddings, p=2, dim=1)  # [B, D]
        anchor_emb_norm = F.normalize(anchor_embeddings, p=2, dim=2)  # [B, N, D]

        # Compute weighted anchor center
        # Weight by similarity scores
        weights_normalized = anchor_weights * anchor_masks  # [B, N]
        weights_sum = weights_normalized.sum(dim=1, keepdim=True) + 1e-8  # [B, 1]
        weights_normalized = weights_normalized / weights_sum  # [B, N]

        # Weighted average of anchors
        # Ensure dtype consistency for bmm operation
        weights_for_bmm = weights_normalized.unsqueeze(1).to(anchor_emb_norm.dtype)  # [B, 1, N]
        anchor_center = torch.bmm(
            weights_for_bmm,                   # [B, 1, N]
            anchor_emb_norm                    # [B, N, D]
        ).squeeze(1)  # [B, D]

        anchor_center = F.normalize(anchor_center, p=2, dim=1)

        # Alignment loss: pull Korean embedding toward anchor center
        if self.distance_type == 'cosine':
            # Cosine distance (1 - cosine similarity)
            alignment_loss = 1.0 - (korean_emb_norm * anchor_center).sum(dim=1)
        else:
            # Euclidean distance
            alignment_loss = ((korean_emb_norm - anchor_center) ** 2).sum(dim=1)

        alignment_loss = alignment_loss.mean()

        # Diversity regularization: prevent token collapse
        # Maximize pairwise distances between Korean embeddings
        if batch_size > 1:
            similarity_matrix = torch.mm(korean_emb_norm, korean_emb_norm.t())

            # Remove diagonal (self-similarity)
            mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
            similarity_matrix = similarity_matrix.masked_fill(mask, 0.0)

            # Diversity loss: minimize similarity (maximize diversity)
            diversity_loss = similarity_matrix.abs().mean()
        else:
            diversity_loss = 0.0

        # Total loss
        total_loss = alignment_loss + self.diversity_weight * diversity_loss

        return total_loss, {
            'alignment': alignment_loss.item(),
            'diversity': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        }


class CLSATrainer:
    """Stage 0: CLSA Trainer"""

    def __init__(self, config_path: str, stage_name: str = 'stage0'):
        self.stage_name = stage_name
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.project_config = config['project']
        self.stage_config = config[stage_name]

        self.setup_logging()
        self.prepare_model()
        self.prepare_data()
        self.prepare_optimizer()

    def setup_logging(self):
        if is_main_process():
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_dir / f'{self.stage_name}.log'),
                    logging.StreamHandler()
                ]
            )
        self.logger = logging.getLogger(__name__)

    def log(self, message):
        if is_main_process():
            self.logger.info(message)

    def prepare_model(self):
        self.log("=" * 80)
        self.log(f"{self.stage_config['name']}")
        self.log("=" * 80)

        local_files_only = (dist.is_initialized() and dist.get_rank() != 0)

        # Load base model (for extracting anchor embeddings)
        base_model_path = self.project_config['base_model']
        self.log(f"\n🔧 Loading base model: {base_model_path}")

        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only
        )

        # Load Korean-expanded tokenizer
        tokenizer_path = PROJECT_ROOT / self.project_config['tokenizer_path']
        self.log(f"🔧 Loading Korean tokenizer: {tokenizer_path}")

        self.korean_tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True
        )

        # Load model with expanded vocabulary
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only
        )

        # Resize embeddings
        old_size = self.model.get_input_embeddings().weight.shape[0]
        new_size = len(self.korean_tokenizer)
        self.log(f"📚 Resizing embeddings: {old_size:,} → {new_size:,}")
        self.model.resize_token_embeddings(new_size)

        if dist.is_initialized():
            dist.barrier()

        self.model = self.model.to(self.device)

        # Freeze all parameters except new token embeddings
        for param in self.model.parameters():
            param.requires_grad = False

        embed_tokens = self.model.get_input_embeddings()
        embed_tokens.weight.requires_grad = True

        # Gradient masking: only train NEW tokens
        old_vocab_size = self.stage_config['old_vocab_size']
        new_token_mask = torch.zeros(new_size, dtype=torch.bool, device=self.device)
        new_token_mask[old_vocab_size:] = True

        def gradient_mask_hook(grad):
            if grad is None:
                return None
            masked_grad = grad.clone()
            masked_grad[~new_token_mask] = 0.0
            return masked_grad

        embed_tokens.weight.register_hook(gradient_mask_hook)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"\n📊 Trainable parameters: {trainable:,}")
        self.log(f"   (New tokens only: {new_size - old_vocab_size:,})")

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            self.log(f"\n✓ DDP (world_size={self.world_size})")

        # CLSA Loss
        self.criterion = CLSALoss(
            diversity_weight=self.stage_config.get('diversity_weight', 0.1),
            distance_type=self.stage_config.get('distance_type', 'cosine')
        ).to(self.device)

    def prepare_data(self):
        self.log("\n" + "=" * 80)
        self.log("Data Preparation")
        self.log("=" * 80)

        dataset_cfg = self.stage_config['dataset']

        # Load CLSA dataset
        bilingual_dict_path = PROJECT_ROOT / dataset_cfg['bilingual_dict_path']
        self.log(f"\n🗂️  Loading bilingual dictionary: {bilingual_dict_path}")

        dataset = CLSADataset(
            bilingual_dict_path=str(bilingual_dict_path),
            base_tokenizer=self.base_tokenizer,
            korean_tokenizer=self.korean_tokenizer,
            min_anchors=dataset_cfg.get('min_anchors', 2)
        )

        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.stage_config['training']['batch_size'],
            collate_fn=collate_clsa,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        self.log(f"\n✓ DataLoader ready")
        self.log(f"  Tokens: {len(dataset):,}")
        self.log(f"  Batches: {len(self.train_dataloader)}")

    def prepare_optimizer(self):
        self.log("\n" + "=" * 80)
        self.log("Optimizer & Scheduler")
        self.log("=" * 80)

        train_cfg = self.stage_config['training']

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(train_cfg['learning_rate']),
            weight_decay=float(train_cfg['weight_decay'])
        )

        num_epochs = train_cfg['num_epochs']
        grad_accum = train_cfg['gradient_accumulation_steps']
        self.total_steps = (len(self.train_dataloader) * num_epochs) // grad_accum

        self.scheduler = get_scheduler(
            train_cfg.get('lr_scheduler_type', 'cosine'),
            optimizer=self.optimizer,
            num_warmup_steps=int(self.total_steps * train_cfg['warmup_ratio']),
            num_training_steps=self.total_steps
        )

        self.log(f"\n✓ AdamW (LR={train_cfg['learning_rate']})")
        self.log(f"  Total steps: {self.total_steps:,}")

    def train_epoch(self, epoch: int):
        self.model.train()
        train_cfg = self.stage_config['training']
        grad_accum = train_cfg['gradient_accumulation_steps']

        if self.world_size > 1 and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        total_loss = 0
        total_alignment = 0
        total_diversity = 0
        step = 0

        if is_main_process():
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_dataloader

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # Get embeddings
            embed_layer = self.model.module.get_input_embeddings() if isinstance(self.model, DDP) else self.model.get_input_embeddings()

            # Korean token embeddings
            korean_embeddings = embed_layer(batch['korean_token_ids'])  # [B, D]

            # Anchor token embeddings
            anchor_token_ids = batch['anchor_token_ids']  # [B, N]
            batch_size, num_anchors = anchor_token_ids.shape

            # Flatten to get all anchor embeddings
            anchor_ids_flat = anchor_token_ids.view(-1)  # [B*N]
            anchor_embeddings_flat = embed_layer(anchor_ids_flat)  # [B*N, D]
            anchor_embeddings = anchor_embeddings_flat.view(batch_size, num_anchors, -1)  # [B, N, D]

            # Compute loss
            loss, loss_dict = self.criterion(
                korean_embeddings,
                anchor_embeddings,
                batch['anchor_weights'],
                batch['anchor_masks']
            )

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

            total_loss += loss.item() * grad_accum
            total_alignment += loss_dict['alignment']
            total_diversity += loss_dict['diversity']

            if is_main_process() and (batch_idx + 1) % self.stage_config['logging']['log_steps'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_align = total_alignment / (batch_idx + 1)
                avg_div = total_diversity / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'align': f'{avg_align:.4f}',
                    'div': f'{avg_div:.4f}',
                    'lr': f'{lr:.2e}'
                })

        return total_loss / len(self.train_dataloader)

    def save_checkpoint(self, epoch: int):
        if not is_main_process():
            return

        final_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        # Save model without triggering DeepSpeed import
        # First save config
        model_to_save.config.save_pretrained(final_dir)

        # Save model weights directly with safetensors
        try:
            from safetensors.torch import save_file
            state_dict = model_to_save.state_dict()
            # Convert to CPU and proper format
            cpu_state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
            save_file(cpu_state_dict, final_dir / "model.safetensors")
        except ImportError:
            # Fallback to torch.save if safetensors not available
            torch.save(model_to_save.state_dict(), final_dir / "pytorch_model.bin")

        self.korean_tokenizer.save_pretrained(final_dir)

        self.log(f"💾 Checkpoint saved: {final_dir}")

    def train(self):
        self.log("\n" + "=" * 80)
        self.log("🚀 CLSA Training Start")
        self.log("=" * 80)

        num_epochs = self.stage_config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch {epoch}/{num_epochs}")
            self.log('='*80)

            avg_loss = self.train_epoch(epoch)
            self.log(f"\n✓ Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        self.save_checkpoint(num_epochs)

        self.log("\n" + "=" * 80)
        self.log(f"✅ {self.stage_config['name']} 완료!")
        self.log("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 0: CLSA Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Pipeline config path"
    )

    args = parser.parse_args()

    trainer = CLSATrainer(config_path=args.config, stage_name='stage0')
    trainer.train()

    cleanup_distributed()


if __name__ == "__main__":
    main()
