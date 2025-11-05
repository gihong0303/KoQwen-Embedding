"""
Base Trainer for Embedding Model Training
공통 트레이너 클래스
"""

import os
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModel, get_scheduler
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
import importlib.util

# Import contrastive_loss
spec = importlib.util.spec_from_file_location("contrastive_loss", PROJECT_ROOT / "utils" / "contrastive_loss.py")
contrastive_loss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contrastive_loss_module)
ContrastiveLoss = contrastive_loss_module.ContrastiveLoss
compute_embedding_stats = contrastive_loss_module.compute_embedding_stats

# Import local_dataset_loader
spec = importlib.util.spec_from_file_location("local_dataset_loader", PROJECT_ROOT / "utils" / "local_dataset_loader.py")
local_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_dataset_module)
LocalDatasetLoader = local_dataset_module.LocalDatasetLoader


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
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            print(f"[DDP] world_size={world_size}, local_rank={local_rank}, "
                  f"CUDA_VISIBLE_DEVICES='{visible}'", flush=True)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class BaseEmbeddingTrainer:
    """Base Trainer for all stages"""

    def __init__(self, stage_name: str, config_path: str, model_path: str = None):
        self.stage_name = stage_name
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.project_config = config['project']
        self.stage_config = config[stage_name]
        self.model_path = model_path

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

        # Load model
        if self.model_path:
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
        else:
            # Stage 1: load base + resize
            self.log(f"\n🔧 Loading base: {self.project_config['base_model']}")
            tokenizer_path = PROJECT_ROOT / self.project_config['tokenizer_path']
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.project_config['base_model'],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only
            )

            # Resize
            old_size = self.model.get_input_embeddings().weight.shape[0]
            new_size = len(self.tokenizer)
            self.log(f"   Resizing: {old_size:,} → {new_size:,}")
            self.model.resize_token_embeddings(new_size)

        vocab_size = len(self.tokenizer)
        self.log(f"\n📚 Vocabulary: {vocab_size:,}")

        if dist.is_initialized():
            dist.barrier()

        self.model = self.model.to(self.device)

        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False

        # Trainable params
        if self.stage_config.get('use_lora'):
            # LoRA
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
        else:
            # Embed tokens
            embed_tokens = self.model.get_input_embeddings()
            embed_tokens.weight.requires_grad = True

            # Gradient masking for new tokens only
            if self.stage_config.get('train_new_tokens_only'):
                old_vocab = self.stage_config['old_vocab_size']
                new_token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
                new_token_mask[old_vocab:] = True

                def gradient_mask_hook(grad):
                    if grad is None:
                        return None
                    masked_grad = grad.clone()
                    masked_grad[~new_token_mask] = 0.0
                    return masked_grad

                embed_tokens.weight.register_hook(gradient_mask_hook)
                self.log(f"\n⚠️  Training embed_tokens (new tokens only: {vocab_size - old_vocab:,})")
            else:
                self.log(f"\n⚠️  Training embed_tokens (ALL tokens: {vocab_size:,})")

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

        # Contrastive Loss
        contrastive_cfg = self.stage_config['contrastive']
        self.criterion = ContrastiveLoss(
            temperature=contrastive_cfg['temperature'],
            pooling=contrastive_cfg['pooling']
        )

    def prepare_data(self):
        self.log("\n" + "=" * 80)
        self.log("Data Preparation")
        self.log("=" * 80)

        dataset_cfg = self.stage_config['dataset']

        # Check if using local dataset
        if dataset_cfg.get('local', False):
            # Log message depends on dataset type
            if dataset_cfg.get('mixed'):
                self.log(f"\n🗂️  Loading mixed local datasets:")
                for ds in dataset_cfg['mixed']:
                    self.log(f"     • {ds['name']}: {ds.get('max_samples', 'all'):,} samples")
            else:
                self.log(f"\n🗂️  Loading local dataset: {dataset_cfg['name']}")

            loader = LocalDatasetLoader(base_path=dataset_cfg.get('local_path', '~/haerae_dataset'))

            # Handle different dataset types
            if dataset_cfg.get('mixed'):
                # Mixed dataset (must check FIRST before accessing 'name')
                dataset = loader.load_mixed_dataset(dataset_cfg['mixed'])
            elif dataset_cfg['name'] == "KoSimpleEval":
                dataset = loader.load_kosimpleeval(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "HAE-RAE-COT":
                dataset = loader.load_cot_dataset(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "HR-Instruct-Math":
                dataset = loader.load_math_dataset(max_samples=dataset_cfg.get('max_samples'))
            elif dataset_cfg['name'] == "K2-Feedback":
                dataset = loader.load_feedback_dataset(
                    min_score=dataset_cfg.get('min_score', 5),
                    max_samples=dataset_cfg.get('max_samples')
                )
            else:
                # Generic local dataset (KOREAN-WEBTEXT, KOREAN-SyntheticText, etc.)
                dataset = loader.load_dataset(
                    dataset_cfg['name'],
                    max_samples=dataset_cfg.get('max_samples')
                )
        else:
            # Use HuggingFace Hub
            self.log(f"\n🤗 Loading HuggingFace dataset: {dataset_cfg['name']}")
            dataset = load_dataset(
                dataset_cfg['name'],
                split='train',
                streaming=dataset_cfg.get('streaming', False)
            )

            if dataset_cfg.get('max_samples'):
                dataset = dataset.select(range(min(
                    dataset_cfg['max_samples'],
                    len(dataset)
                )))

        def collate_fn(examples):
            texts = [ex['text'] for ex in examples]
            encodings = self.tokenizer(
                texts,
                max_length=self.stage_config['training']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }

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
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        self.log(f"\n✓ DataLoader ready")
        self.log(f"  Batch/GPU: {self.stage_config['training']['batch_size']}")
        self.log(f"  Total batches: {len(self.train_dataloader)}")

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
        self.log(f"✓ Scheduler: cosine")
        self.log(f"  Total steps: {self.total_steps:,}")

    def train_epoch(self, epoch: int):
        self.model.train()
        train_cfg = self.stage_config['training']
        grad_accum = train_cfg['gradient_accumulation_steps']

        if self.world_size > 1 and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        total_loss = 0
        step = 0

        if is_main_process():
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_dataloader

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            model_unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
            loss, _ = self.criterion(model_unwrapped, batch['input_ids'], batch['attention_mask'])

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

            total_loss += loss.item() * grad_accum

            if is_main_process() and (batch_idx + 1) % self.stage_config['logging']['log_steps'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

            if step > 0 and step % self.stage_config['checkpoint']['save_steps'] == 0:
                self.save_checkpoint(epoch, step, total_loss / (batch_idx + 1))

        return total_loss / len(self.train_dataloader)

    def save_checkpoint(self, epoch: int, step: int, loss: float):
        if not is_main_process():
            return

        ckpt_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(ckpt_dir)
        else:
            # LoRA model
            model_to_save.save_pretrained(ckpt_dir)

        self.tokenizer.save_pretrained(ckpt_dir)

        import json
        with open(ckpt_dir / "metadata.json", 'w') as f:
            json.dump({'epoch': epoch, 'step': step, 'loss': loss}, f, indent=2)

        self.log(f"💾 Checkpoint: {ckpt_dir}")

    def train(self):
        self.log("\n" + "=" * 80)
        self.log("🚀 Training Start")
        self.log("=" * 80)

        num_epochs = self.stage_config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch {epoch}/{num_epochs}")
            self.log('='*80)

            avg_loss = self.train_epoch(epoch)
            self.log(f"\n✓ Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

            # Embedding stats
            if is_main_process() and not self.stage_config.get('use_lora'):
                model_unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
                stats = compute_embedding_stats(model_unwrapped, self.tokenizer, self.device)
                self.log(f"\n📊 Embedding Stats:")
                self.log(f"  Old: {stats['old_tokens_mean_norm']:.4f} ± {stats['old_tokens_std_norm']:.4f}")
                self.log(f"  New: {stats['new_tokens_mean_norm']:.4f} ± {stats['new_tokens_std_norm']:.4f}")

        if is_main_process():
            final_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

            if hasattr(model_to_save, 'save_pretrained'):
                model_to_save.save_pretrained(final_dir)
            else:
                model_to_save.save_pretrained(final_dir)

            self.tokenizer.save_pretrained(final_dir)

            self.log("\n" + "=" * 80)
            self.log(f"✅ {self.stage_config['name']} 완료!")
            self.log(f"   Output: {final_dir}")
            self.log("=" * 80)
