#!/usr/bin/env python3
"""
KURE Trainer: Korean Universal Representation Enhancement

통합 KURE 트레이너
- PJC: Phonological Jamo Composition
- MGC: Morpheme-guided Curriculum
- HCL: Hierarchical Contrastive Learning
- MGR: Multi-granularity Representation
- ALB: Adaptive Loss Balancing
- VGT: Validation-guided Training
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModel, get_scheduler, set_seed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import KURE components
from utils.phonological_jamo import PhonologicalJamoComposer, PJCLoss
from utils.morpheme_curriculum import (
    MorphemeAnalyzer, AdaptiveCurriculumScheduler,
    MorphemeCurriculumDataset, MGCLossWeighting
)
from utils.hierarchical_contrastive import (
    HierarchicalContrastiveLoss, HardNegativeMiner,
    SentenceContrastiveLoss
)
from utils.kure_components import (
    MatryoshkaLoss, GradNormBalancer, SimpleLossBalancer,
    EarlyStopping, ValidationCallback, KURELoss
)
from utils.local_dataset_loader import LocalDatasetLoader

logger = logging.getLogger(__name__)


# ============================================================================
# Distributed Setup
# ============================================================================

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


# ============================================================================
# KURE Trainer
# ============================================================================

class KURETrainer:
    """
    KURE 통합 트레이너

    모든 KURE 구성 요소를 통합하여 학습 수행
    """

    def __init__(
        self,
        config_path: str,
        stage_name: str,
        model_path: Optional[str] = None
    ):
        self.stage_name = stage_name
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.project_config = self.config['project']
        self.stage_config = self.config[stage_name]
        self.kure_config = self.config.get('kure', {})
        self.model_path = model_path

        self.setup_logging()
        self.prepare_components()
        self.prepare_model()
        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_validation()

    def setup_logging(self):
        if is_main_process():
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_dir / f'kure_{self.stage_name}.log'),
                    logging.StreamHandler()
                ]
            )
        self.logger = logging.getLogger(__name__)

    def log(self, message):
        if is_main_process():
            self.logger.info(message)

    def prepare_components(self):
        """KURE 구성 요소 초기화"""
        self.log("=" * 80)
        self.log("Initializing KURE Components")
        self.log("=" * 80)

        stage_cfg = self.stage_config
        hidden_dim = 1536  # Qwen3-Embedding hidden size

        # 1. PJC (Phonological Jamo Composition)
        self.use_pjc = stage_cfg.get('use_pjc', False)
        if self.use_pjc:
            pjc_cfg = stage_cfg.get('pjc', {})
            self.pjc_composer = PhonologicalJamoComposer(
                hidden_dim=hidden_dim,
                use_phonological_rules=True,
                inter_syllable_type=pjc_cfg.get('inter_syllable_type', 'transformer')
            ).to(self.device)
            self.pjc_blend_ratio = pjc_cfg.get('blend_ratio', 0.4)
            self.pjc_loss = PJCLoss()
            self.log(f"  ✓ PJC enabled (blend_ratio={self.pjc_blend_ratio})")

        # 2. MGC (Morpheme-guided Curriculum)
        self.use_mgc = stage_cfg.get('use_mgc', False)
        if self.use_mgc:
            mgc_cfg = self.kure_config.get('mgc', {})
            self.morpheme_analyzer = MorphemeAnalyzer(
                use_mecab=mgc_cfg.get('use_mecab', True),
                cache_path=mgc_cfg.get('cache_path')
            )
            self.curriculum_scheduler = AdaptiveCurriculumScheduler(
                adaptation_rate=mgc_cfg.get('adaptation_rate', 0.1)
            )
            curriculum_cfg = stage_cfg.get('curriculum', {})
            self.mgc_mode = curriculum_cfg.get('mode', 'easy')
            self.mgc_loss_weighting = MGCLossWeighting(
                stage=int(self.stage_name[-1]) if self.stage_name[-1].isdigit() else 1
            )
            self.log(f"  ✓ MGC enabled (mode={self.mgc_mode})")

        # 3. HCL (Hierarchical Contrastive Learning)
        self.use_hcl = stage_cfg.get('use_hcl', False)
        if self.use_hcl:
            hcl_cfg = stage_cfg.get('hcl', self.kure_config.get('hcl', {}))
            self.hcl_loss = HierarchicalContrastiveLoss(
                hidden_dim=hidden_dim,
                token_weight=hcl_cfg.get('token_weight', 0.2),
                phrase_weight=hcl_cfg.get('phrase_weight', 0.3),
                sentence_weight=hcl_cfg.get('sentence_weight', 0.5),
                use_hard_negatives=hcl_cfg.get('use_hard_negatives', True),
                memory_size=hcl_cfg.get('memory_size', 65536)
            )
            self.log(f"  ✓ HCL enabled")

        # 4. MGR (Multi-granularity Representation / Matryoshka)
        self.use_mgr = stage_cfg.get('use_mgr', False)
        if self.use_mgr:
            mgr_cfg = stage_cfg.get('mgr', self.kure_config.get('mgr', {}))
            self.matryoshka_loss = MatryoshkaLoss(
                full_dim=hidden_dim,
                dimensions=mgr_cfg.get('dimensions', [1536, 768, 384, 192, 96]),
                temperature=stage_cfg.get('contrastive', {}).get('temperature', 0.05)
            )
            self.log(f"  ✓ MGR enabled (Matryoshka)")

        # 5. ALB (Adaptive Loss Balancing)
        self.use_alb = stage_cfg.get('use_alb', False)
        if self.use_alb:
            alb_cfg = stage_cfg.get('alb', self.kure_config.get('alb', {}))
            loss_names = self._get_active_loss_names()
            if alb_cfg.get('use_gradnorm', True):
                self.loss_balancer = GradNormBalancer(
                    num_losses=len(loss_names),
                    loss_names=loss_names,
                    alpha=alb_cfg.get('alpha', 1.5)
                ).to(self.device)
            else:
                self.loss_balancer = SimpleLossBalancer(loss_names)
            self.log(f"  ✓ ALB enabled (losses: {loss_names})")

        # 기본 Contrastive Loss (HCL 미사용 시)
        if not self.use_hcl:
            contrastive_cfg = stage_cfg.get('contrastive', {})
            self.base_contrastive = SentenceContrastiveLoss(
                temperature=contrastive_cfg.get('temperature', 0.05),
                pooling=contrastive_cfg.get('pooling', 'mean')
            )

    def _get_active_loss_names(self) -> List[str]:
        """활성화된 손실 함수 이름 반환"""
        names = ['contrastive']
        if self.use_pjc:
            names.append('pjc')
        if self.use_mgc:
            names.append('curriculum')
        if self.use_mgr:
            names.append('matryoshka')
        return names

    def prepare_model(self):
        """모델 준비"""
        self.log("\n" + "=" * 80)
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
            # Stage 0 또는 첫 스테이지
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
            if old_size != new_size:
                self.log(f"   Resizing: {old_size:,} → {new_size:,}")
                self.model.resize_token_embeddings(new_size)

        vocab_size = len(self.tokenizer)
        self.log(f"\n📚 Vocabulary: {vocab_size:,}")

        if dist.is_initialized():
            dist.barrier()

        self.model = self.model.to(self.device)

        # Freeze and set trainable
        self._setup_trainable_params()

        # DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.stage_config.get('use_lora', False)
            )
            self.log(f"\n✓ DDP (world_size={self.world_size})")

    def _setup_trainable_params(self):
        """학습 가능 파라미터 설정"""
        stage_cfg = self.stage_config

        # 모든 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False

        if stage_cfg.get('use_lora', False):
            # LoRA 적용
            from peft import get_peft_model, LoraConfig, TaskType

            lora_cfg = stage_cfg['lora_config']
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
            # Embedding만 학습
            embed_tokens = self.model.get_input_embeddings()
            embed_tokens.weight.requires_grad = True

            if stage_cfg.get('train_new_tokens_only', False):
                old_vocab = stage_cfg['old_vocab_size']
                vocab_size = len(self.tokenizer)
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
                self.log(f"\n⚠️  Training embed_tokens (ALL tokens)")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        self.log(f"\n📊 Parameters:")
        self.log(f"  Total: {total:,}")
        self.log(f"  Trainable: {trainable:,}")
        self.log(f"  Percentage: {100 * trainable / total:.2f}%")

    def prepare_data(self):
        """데이터 준비"""
        self.log("\n" + "=" * 80)
        self.log("Data Preparation")
        self.log("=" * 80)

        dataset_cfg = self.stage_config.get('dataset', {})
        loader = LocalDatasetLoader(base_path=dataset_cfg.get('local_path', '~/haerae_dataset'))

        # 데이터셋 로드
        if dataset_cfg.get('mixed'):
            self.log(f"\n🗂️  Loading mixed datasets:")
            for ds in dataset_cfg['mixed']:
                self.log(f"     • {ds['name']}: {ds.get('max_samples', 'all'):,} samples")
            dataset = loader.load_mixed_dataset(dataset_cfg['mixed'])
        else:
            dataset_name = dataset_cfg.get('name', 'KOREAN-WEBTEXT')
            self.log(f"\n🗂️  Loading: {dataset_name}")
            dataset = loader.load_dataset(
                dataset_name,
                max_samples=dataset_cfg.get('max_samples')
            )

        # MGC 커리큘럼 적용
        if self.use_mgc:
            stage_num = int(self.stage_name[-1]) if self.stage_name[-1].isdigit() else 1
            dataset = MorphemeCurriculumDataset(
                base_dataset=dataset,
                analyzer=self.morpheme_analyzer,
                tokenizer=self.tokenizer,
                stage=stage_num,
                curriculum_weights=self.curriculum_scheduler.get_stage_weights(stage_num)
            )
            self.log(f"\n✓ MGC curriculum applied (stage={stage_num})")

        # Collate function
        def collate_fn(examples):
            texts = [ex['text'] for ex in examples]
            encodings = self.tokenizer(
                texts,
                max_length=self.stage_config['training']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            batch = {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }
            # MGC 카테고리 추가
            if self.use_mgc and 'category' in examples[0]:
                batch['categories'] = [ex['category'] for ex in examples]
            return batch

        # Sampler
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
        """Optimizer 준비"""
        self.log("\n" + "=" * 80)
        self.log("Optimizer & Scheduler")
        self.log("=" * 80)

        train_cfg = self.stage_config['training']

        # Collect all trainable params
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        # PJC params
        if self.use_pjc:
            trainable_params.extend(self.pjc_composer.parameters())

        # ALB params (if GradNorm)
        if self.use_alb and hasattr(self.loss_balancer, 'parameters'):
            trainable_params.extend(self.loss_balancer.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(train_cfg['learning_rate']),
            weight_decay=float(train_cfg['weight_decay'])
        )

        num_epochs = train_cfg['num_epochs']
        grad_accum = train_cfg['gradient_accumulation_steps']
        self.total_steps = (len(self.train_dataloader) * num_epochs) // grad_accum

        scheduler_type = train_cfg.get('lr_scheduler_type', 'cosine')
        self.scheduler = get_scheduler(
            scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.total_steps * train_cfg['warmup_ratio']),
            num_training_steps=self.total_steps
        )

        self.log(f"\n✓ AdamW (LR={train_cfg['learning_rate']})")
        self.log(f"✓ Scheduler: {scheduler_type}")
        self.log(f"  Total steps: {self.total_steps:,}")

    def prepare_validation(self):
        """VGT 검증 준비"""
        val_cfg = self.stage_config.get('validation', {})
        self.use_vgt = val_cfg.get('enabled', False)

        if self.use_vgt and is_main_process():
            # Early stopping
            es_cfg = val_cfg.get('early_stopping', {})
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get('patience', 5),
                min_delta=es_cfg.get('min_delta', 0.001),
                mode='min'
            )
            self.eval_steps = val_cfg.get('eval_steps', 500)
            self.log(f"\n✓ VGT enabled (eval_steps={self.eval_steps})")

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        손실 계산

        Returns:
            total_loss: 총 손실
            loss_dict: 개별 손실
        """
        model = self.model.module if isinstance(self.model, DDP) else self.model
        losses = {}
        loss_dict = {}

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # 1. HCL or Base Contrastive
        if self.use_hcl:
            hcl_loss, hcl_info = self.hcl_loss(
                model, input_ids, attention_mask
            )
            losses['contrastive'] = hcl_loss
            loss_dict.update(hcl_info)
        else:
            cont_loss, embeddings = self.base_contrastive(
                model, input_ids, attention_mask
            )
            losses['contrastive'] = cont_loss
            loss_dict['contrastive_loss'] = cont_loss.item()

        # 2. PJC Loss
        if self.use_pjc:
            # 토큰 문자열 추출
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[:, 0]]
            pjc_embeddings = self.pjc_composer(tokens, device=self.device)

            # 타겟 임베딩
            embed_layer = model.get_input_embeddings()
            target_embeddings = embed_layer(input_ids[:, 0])

            pjc_loss = self.pjc_loss(pjc_embeddings, target_embeddings)
            losses['pjc'] = pjc_loss
            loss_dict['pjc_loss'] = pjc_loss.item()

        # 3. MGC Loss weighting
        if self.use_mgc and 'categories' in batch:
            # 카테고리별 가중치 적용은 ALB에서 처리
            pass

        # 4. MGR (Matryoshka) Loss
        if self.use_mgr:
            # 두 번의 forward로 임베딩 얻기
            model.train()
            out1 = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            out2 = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            # Mean pooling
            mask = attention_mask.unsqueeze(-1)
            emb1 = (out1.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb2 = (out2.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            matryoshka_loss, matryoshka_info = self.matryoshka_loss(emb1, emb2)
            losses['matryoshka'] = matryoshka_loss
            loss_dict.update(matryoshka_info)

        # 5. ALB로 가중 합산
        if self.use_alb:
            loss_list = [losses.get(name, torch.tensor(0.0, device=self.device))
                        for name in self._get_active_loss_names()]

            shared_layer = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else None

            if hasattr(self.loss_balancer, 'forward'):
                total_loss, balance_info = self.loss_balancer(loss_list, shared_layer)
            else:
                total_loss = self.loss_balancer.get_weighted_loss(losses)
                balance_info = {}

            loss_dict.update(balance_info)
        else:
            # 단순 합산
            total_loss = sum(losses.values())

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, epoch: int) -> float:
        """에폭 학습"""
        self.model.train()
        if self.use_pjc:
            self.pjc_composer.train()

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
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            loss, loss_dict = self.compute_losses(batch)

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    train_cfg['max_grad_norm']
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

            total_loss += loss.item() * grad_accum

            if is_main_process() and (batch_idx + 1) % self.stage_config['logging']['log_steps'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}'
                })

            # VGT: 주기적 검증
            if self.use_vgt and is_main_process() and step > 0 and step % self.eval_steps == 0:
                val_loss = self._validate()
                should_stop, is_best = self.early_stopping(val_loss, epoch, self.model)
                self.log(f"\n  Validation Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
                if should_stop:
                    self.log(f"\n  Early stopping triggered!")
                    break

            # 체크포인트 저장
            if step > 0 and step % self.stage_config['checkpoint']['save_steps'] == 0:
                self.save_checkpoint(epoch, step, total_loss / (batch_idx + 1))

        return total_loss / len(self.train_dataloader)

    def _validate(self) -> float:
        """간단한 검증"""
        self.model.eval()
        val_loss = 0
        count = 0

        with torch.no_grad():
            for batch in list(self.train_dataloader)[:10]:  # 10 배치만
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                loss, _ = self.compute_losses(batch)
                val_loss += loss.item()
                count += 1

        self.model.train()
        return val_loss / count if count > 0 else 0

    def save_checkpoint(self, epoch: int, step: int, loss: float):
        if not is_main_process():
            return

        ckpt_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        # PJC 저장
        if self.use_pjc:
            torch.save(self.pjc_composer.state_dict(), ckpt_dir / "pjc_composer.pt")

        # 메타데이터
        with open(ckpt_dir / "metadata.json", 'w') as f:
            json.dump({'epoch': epoch, 'step': step, 'loss': loss}, f, indent=2)

        self.log(f"💾 Checkpoint: {ckpt_dir}")

    def train(self):
        """전체 학습"""
        self.log("\n" + "=" * 80)
        self.log("🚀 KURE Training Start")
        self.log("=" * 80)

        num_epochs = self.stage_config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch {epoch}/{num_epochs}")
            self.log('='*80)

            avg_loss = self.train_epoch(epoch)
            self.log(f"\n✓ Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        # 최종 저장
        if is_main_process():
            final_dir = PROJECT_ROOT / self.stage_config['checkpoint']['output_dir'] / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_save.save_pretrained(final_dir)
            self.tokenizer.save_pretrained(final_dir)

            if self.use_pjc:
                torch.save(self.pjc_composer.state_dict(), final_dir / "pjc_composer.pt")

            self.log("\n" + "=" * 80)
            self.log(f"✅ {self.stage_config['name']} 완료!")
            self.log(f"   Output: {final_dir}")
            self.log("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KURE Trainer")
    parser.add_argument("--config", type=str, default="configs/kure_config.yaml")
    parser.add_argument("--stage", type=str, required=True, help="Stage name (e.g., stage1)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        trainer = KURETrainer(
            config_path=args.config,
            stage_name=args.stage,
            model_path=args.model_path
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
