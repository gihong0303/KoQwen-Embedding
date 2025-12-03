#!/usr/bin/env python3
"""
Stage 6: Supervised Retrieval Contrastive Learning

이 단계는 MIRACL + KorNLI 데이터를 활용하여 검색 능력을 직접 최적화합니다.
기존 SimCSE (unsupervised)와 달리 실제 query-document 쌍을 사용합니다.

Key Features:
- Query-Positive-Negative triplet 학습
- In-batch negatives + Hard negatives
- 낮은 learning rate로 기존 능력 보존
- LoRA로 parameter-efficient fine-tuning
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import yaml
from transformers import AutoTokenizer, AutoModel, get_scheduler, set_seed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import utilities
from utils.retrieval_loss import SupervisedRetrievalLoss, MultipleNegativesRankingLoss
from utils.retrieval_dataset import (
    MIRACLDataset,
    CombinedRetrievalDataset,
    RetrievalCollator,
    create_retrieval_dataloader
)


def setup_distributed():
    """DDP initialization with robust error handling"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(minutes=30),
                world_size=world_size,
                rank=rank
            )

        torch.cuda.set_device(local_rank)
        dist.barrier()

        if rank == 0:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            print(f"[DDP] Initialized: world_size={world_size}, local_rank={local_rank}, "
                  f"CUDA_VISIBLE_DEVICES='{visible}'", flush=True)

        return rank, world_size, local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class Stage6RetrievalTrainer:
    """
    Stage 6 Trainer for Supervised Retrieval Learning

    Uses MIRACL Korean + KorNLI for query-document matching.
    """

    def __init__(self, config_path: str, model_path: str):
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.project_config = config['project']
        self.stage_config = config['stage6']
        self.model_path = model_path

        self.setup_logging()
        self.prepare_model()
        self.prepare_data()
        self.prepare_optimizer()

    def setup_logging(self):
        if is_main_process():
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / 'stage6_retrieval.log'

            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='a'),
                    logging.StreamHandler()
                ],
                force=True
            )
            print(f"📝 Logging to: {log_file}")
        self.logger = logging.getLogger(__name__)

    def log(self, message):
        if is_main_process():
            self.logger.info(message)

    def prepare_model(self):
        self.log("=" * 80)
        self.log("Stage 6: Supervised Retrieval Contrastive Learning")
        self.log("=" * 80)

        local_files_only = (dist.is_initialized() and dist.get_rank() != 0)

        self.log(f"\n🔧 Loading from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only
        )

        vocab_size = len(self.tokenizer)
        self.log(f"\n📚 Vocabulary: {vocab_size:,}")

        if dist.is_initialized():
            dist.barrier()

        self.model = self.model.to(self.device)

        # Enable gradient checkpointing if configured
        if self.stage_config.get('optimization', {}).get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.log(f"\n✓ Gradient checkpointing enabled")

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA
        if self.stage_config.get('use_lora'):
            from peft import get_peft_model, LoraConfig, TaskType

            lora_cfg = self.stage_config['lora_config']
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg['r'],
                lora_alpha=lora_cfg['lora_alpha'],
                lora_dropout=lora_cfg['lora_dropout'],
                target_modules=lora_cfg['target_modules'],
                bias=lora_cfg['bias']
            )
            self.model = get_peft_model(self.model, peft_config)
            self.log(f"\n✓ LoRA enabled (r={lora_cfg['r']})")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        self.log(f"\n📊 Parameters:")
        self.log(f"  Total: {total:,}")
        self.log(f"  Trainable: {trainable:,}")
        self.log(f"  Percentage: {100 * trainable / total:.2f}%")

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            self.log(f"\n✓ DDP (world_size={self.world_size})")

        # Loss function
        retrieval_cfg = self.stage_config.get('retrieval', {})
        loss_type = retrieval_cfg.get('loss_type', 'mnrl')

        if loss_type == 'supervised':
            self.criterion = SupervisedRetrievalLoss(
                temperature=retrieval_cfg.get('temperature', 0.05),
                pooling=retrieval_cfg.get('pooling', 'mean'),
                use_hard_negatives=retrieval_cfg.get('use_hard_negatives', True)
            )
        else:  # mnrl (Multiple Negatives Ranking Loss)
            self.criterion = MultipleNegativesRankingLoss(
                temperature=retrieval_cfg.get('temperature', 0.05),
                pooling=retrieval_cfg.get('pooling', 'mean')
            )

        self.log(f"\n✓ Loss: {loss_type} (temperature={retrieval_cfg.get('temperature', 0.05)})")

    def prepare_data(self):
        self.log("\n" + "=" * 80)
        self.log("Data Preparation: MIRACL + KorNLI")
        self.log("=" * 80)

        dataset_cfg = self.stage_config.get('dataset', {})

        # Load combined dataset
        miracl_samples = dataset_cfg.get('miracl_samples', 50000)
        kornli_samples = dataset_cfg.get('kornli_samples', 100000)

        self.log(f"\n🗂️  Loading datasets:")
        self.log(f"   • MIRACL Korean: {miracl_samples:,} samples")
        self.log(f"   • KorNLI: {kornli_samples:,} samples")

        try:
            dataset = CombinedRetrievalDataset(
                miracl_samples=miracl_samples,
                kornli_samples=kornli_samples,
                include_hard_negatives=dataset_cfg.get('include_hard_negatives', True)
            )
        except Exception as e:
            self.log(f"Warning: Could not load CombinedRetrievalDataset: {e}")
            self.log("Falling back to MIRACL only...")
            dataset = MIRACLDataset(
                split="train",
                max_samples=miracl_samples,
                include_hard_negatives=True
            )

        # Create DataLoader
        train_cfg = self.stage_config['training']

        collator = RetrievalCollator(
            tokenizer=self.tokenizer,
            max_query_length=train_cfg.get('max_query_length', 128),
            max_passage_length=train_cfg.get('max_passage_length', 384)
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
            batch_size=train_cfg['batch_size'],
            collate_fn=collator,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.log(f"\n✓ DataLoader ready")
        self.log(f"  Batch/GPU: {train_cfg['batch_size']}")
        self.log(f"  Total batches: {len(self.train_dataloader)}")
        self.log(f"  Total samples: {len(dataset):,}")

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
        self.log(f"✓ Scheduler: {train_cfg.get('lr_scheduler_type', 'cosine')}")
        self.log(f"  Total steps: {self.total_steps:,}")

    def train_epoch(self, epoch: int):
        self.model.train()
        train_cfg = self.stage_config['training']
        grad_accum = train_cfg['gradient_accumulation_steps']

        if self.world_size > 1 and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        total_loss = 0
        total_accuracy = 0
        step = 0

        if is_main_process():
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_dataloader

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            model_unwrapped = self.model.module if isinstance(self.model, DDP) else self.model

            # Forward pass with retrieval loss
            if "neg_input_ids" in batch:
                loss, metrics = self.criterion(
                    model_unwrapped,
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    pos_input_ids=batch['pos_input_ids'],
                    pos_attention_mask=batch['pos_attention_mask'],
                    neg_input_ids=batch['neg_input_ids'],
                    neg_attention_mask=batch['neg_attention_mask']
                )
            else:
                loss, metrics = self.criterion(
                    model_unwrapped,
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    pos_input_ids=batch['pos_input_ids'],
                    pos_attention_mask=batch['pos_attention_mask']
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
            total_accuracy += metrics.get('accuracy', 0)

            if is_main_process() and (batch_idx + 1) % self.stage_config['logging']['log_steps'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = total_accuracy / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.4f}',
                    'lr': f'{lr:.2e}'
                })

            # Save checkpoint
            if step > 0 and step % self.stage_config['checkpoint']['save_steps'] == 0:
                self.save_checkpoint(epoch, step, total_loss / (batch_idx + 1))

        return total_loss / len(self.train_dataloader), total_accuracy / len(self.train_dataloader)

    def _safe_save_model(self, model_to_save, save_dir):
        """Save model - handles both regular and LoRA models"""
        save_dir = Path(save_dir)

        is_peft_model = hasattr(model_to_save, 'merge_and_unload')

        if is_peft_model:
            self.log("  Merging LoRA adapters into base model...")
            merged_model = model_to_save.merge_and_unload()
            model_to_save = merged_model

        model_to_save.config.save_pretrained(save_dir)

        try:
            from safetensors.torch import save_file
            state_dict = model_to_save.state_dict()
            cpu_state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
            save_file(cpu_state_dict, save_dir / "model.safetensors")
        except ImportError:
            torch.save(model_to_save.state_dict(), save_dir / "pytorch_model.bin")

        if is_peft_model:
            self.log("  LoRA adapters merged and saved successfully")

    def save_checkpoint(self, epoch: int, step: int, loss: float):
        if not is_main_process():
            return

        ckpt_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        self._safe_save_model(model_to_save, ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        with open(ckpt_dir / "metadata.json", 'w') as f:
            json.dump({'epoch': epoch, 'step': step, 'loss': loss}, f, indent=2)

        self.log(f"💾 Checkpoint: {ckpt_dir}")

    def train(self):
        self.log("\n" + "=" * 80)
        self.log("🚀 Stage 6 Training Start - Supervised Retrieval")
        self.log("=" * 80)

        num_epochs = self.stage_config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch {epoch}/{num_epochs}")
            self.log('='*80)

            avg_loss, avg_acc = self.train_epoch(epoch)
            self.log(f"\n✓ Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        # Save final model
        if is_main_process():
            final_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            self._safe_save_model(model_to_save, final_dir)
            self.tokenizer.save_pretrained(final_dir)

            self.log("\n" + "=" * 80)
            self.log("✅ Stage 6 - Supervised Retrieval Training Complete!")
            self.log(f"   Output: {final_dir}")
            self.log("=" * 80)

        # Cleanup
        if dist.is_initialized():
            dist.barrier()
        cleanup_distributed()

        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        trainer = Stage6RetrievalTrainer(args.config, args.model_path)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
