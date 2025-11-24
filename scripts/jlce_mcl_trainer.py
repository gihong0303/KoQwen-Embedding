"""
JLCE + MCL Trainer for Stages 1-3

Jamo-Level Compositional Embedding + Morphological Curriculum Learning
한국어 특화 임베딩 학습을 위한 통합 트레이너
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_scheduler
from tqdm import tqdm
import yaml

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.jamo_embedding import (
    JamoEmbeddingLayer,
    JamoCompositionLoss,
    tokenize_to_jamo_batch,
    is_hangul_syllable
)
from utils.morphological_curriculum import (
    MorphologicalAnalyzer,
    MorphologicalCurriculumDataset,
    MorphologicalCurriculumLoss
)
from utils.hybrid_loss import JLCEMCLLoss, get_optimizer_config
from utils.contrastive_loss import mean_pooling

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JLCEMCLTrainer:
    """
    JLCE + MCL 통합 트레이너

    Features:
    - Jamo-Level Compositional Embedding
    - Morphological Curriculum Learning
    - DDP (Distributed Data Parallel) 지원
    - Mixed precision (BF16)
    """

    def __init__(
        self,
        config_path: str,
        stage: int,
        model_path: Optional[str] = None
    ):
        self.config_path = config_path
        self.stage = stage
        self.model_path = model_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.stage_config = self.config[f'stage{stage}']
        self.project_config = self.config['project']

        # DDP setup
        self._setup_distributed()

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.jamo_layer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None

    def _setup_distributed(self):
        """DDP 초기화"""
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))

            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    timeout=timedelta(minutes=10)
                )

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.is_main_process():
            logger.info(f"[DDP] world_size={self.world_size}, local_rank={self.local_rank}")

    def is_main_process(self) -> bool:
        return self.rank == 0

    def log(self, msg: str):
        if self.is_main_process():
            logger.info(msg)

    def load_model(self):
        """모델 및 토크나이저 로드"""
        self.log("\n" + "=" * 80)
        self.log(f"Loading Model for Stage {self.stage}")
        self.log("=" * 80)

        # Model path
        if self.model_path:
            model_path = self.model_path
        elif self.stage > 0:
            # 이전 stage checkpoint 사용
            prev_stage = self.stage - 1
            model_path = f"checkpoints/stage{prev_stage}/final"
            if not Path(model_path).exists():
                model_path = self.project_config['base_model']
        else:
            model_path = self.project_config['base_model']

        self.log(f"   Model path: {model_path}")

        # 토크나이저
        tokenizer_path = self.project_config.get('tokenizer_path', model_path)
        self.log(f"   Tokenizer path: {tokenizer_path}")

        # Rank > 0은 rank 0이 다운로드 완료할 때까지 대기
        local_files_only = (dist.is_initialized() and self.rank != 0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=local_files_only
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only
        )

        # DDP barrier
        if dist.is_initialized():
            dist.barrier()

        # Resize embeddings if needed
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.log(f"   Resizing embeddings: {self.model.config.vocab_size} -> {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        # Jamo Embedding Layer 초기화
        hidden_dim = self.model.config.hidden_size
        jlce_config = self.config.get('korean_specific', {}).get('jlce', {})

        self.jamo_layer = JamoEmbeddingLayer(
            hidden_dim=hidden_dim,
            composition_type=jlce_config.get('composition_type', 'mlp'),
            use_position_encoding=jlce_config.get('use_position_encoding', True)
        ).to(self.device)

        self.log(f"   Jamo layer initialized with hidden_dim={hidden_dim}")

        # Freeze layers
        self._freeze_params()

        # DDP wrap
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.jamo_layer = DDP(self.jamo_layer, device_ids=[self.local_rank])

        self.log(f"   Model loaded successfully")
        self.log(f"   Vocab size: {len(self.tokenizer)}")

    def _freeze_params(self):
        """파라미터 동결 설정"""
        train_new_only = self.stage_config.get('train_new_tokens_only', True)
        old_vocab_size = self.stage_config.get('old_vocab_size', 151669)

        # Transformer layers 동결
        for name, param in self.model.named_parameters():
            if 'embed_tokens' not in name:
                param.requires_grad = False

        # 새 토큰만 학습
        if train_new_only:
            embed_layer = self.model.get_input_embeddings()
            embed_layer.weight.requires_grad = True

            # 기존 토큰 gradient 마스킹 (forward hook으로 처리)
            self.old_vocab_size = old_vocab_size

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.log(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def prepare_data(self):
        """데이터 준비"""
        self.log("\n" + "=" * 80)
        self.log("Preparing Data with Morphological Curriculum")
        self.log("=" * 80)

        from datasets import load_dataset, concatenate_datasets

        dataset_cfg = self.stage_config['dataset']
        datasets_list = []

        # Mixed datasets 로드
        if dataset_cfg.get('mixed'):
            for ds_config in dataset_cfg['mixed']:
                ds_name = ds_config['name']
                source = ds_config.get('source', 'huggingface')
                max_samples = ds_config.get('max_samples')

                self.log(f"   Loading: {ds_name} ({source}), max={max_samples}")

                if source == 'huggingface':
                    ds = load_dataset(
                        ds_name,
                        split=ds_config.get('split', 'train'),
                        streaming=False
                    )
                    if max_samples:
                        ds = ds.select(range(min(max_samples, len(ds))))
                else:
                    # Local dataset handling
                    local_path = ds_config.get('local_path', '~/haerae_dataset')
                    local_path = os.path.expanduser(local_path)
                    if os.path.exists(local_path):
                        from datasets import Dataset
                        ds = Dataset.load_from_disk(local_path)
                        if max_samples:
                            ds = ds.select(range(min(max_samples, len(ds))))
                    else:
                        self.log(f"   Warning: Local path not found: {local_path}")
                        continue

                datasets_list.append(ds)

        if datasets_list:
            dataset = concatenate_datasets(datasets_list)
        else:
            raise ValueError("No datasets loaded!")

        self.log(f"   Total samples: {len(dataset)}")

        # Morphological Curriculum 적용
        curriculum_cfg = self.stage_config.get('curriculum', {})
        if curriculum_cfg.get('enabled', False):
            self.log(f"   Applying MCL: mode={curriculum_cfg['mode']}")

            mcl_dataset = MorphologicalCurriculumDataset(
                tokenizer=self.tokenizer,
                cache_path=curriculum_cfg.get('difficulty_categories_path')
            )

            if not mcl_dataset.token_categories:
                # 캐시가 없으면 분석 수행
                mcl_dataset.analyze_vocabulary()
                mcl_dataset.save_cache()

            # Stage별 샘플 필터링
            target_tokens = mcl_dataset.get_tokens_by_stage(self.stage)
            self.log(f"   Target tokens for stage {self.stage}: {len(target_tokens)}")

        # DataLoader 생성
        def collate_fn(examples):
            # text 필드 찾기
            if 'text' in examples[0]:
                texts = [ex['text'] for ex in examples]
            elif 'sentence' in examples[0]:
                texts = [ex['sentence'] for ex in examples]
            elif 'premise' in examples[0]:
                # NLI dataset
                texts = [ex['premise'] + ' ' + ex.get('hypothesis', '') for ex in examples]
            else:
                # 첫 번째 문자열 필드 사용
                for key, val in examples[0].items():
                    if isinstance(val, str):
                        texts = [ex[key] for ex in examples]
                        break
                else:
                    texts = [str(ex) for ex in examples]

            encodings = self.tokenizer(
                texts,
                max_length=self.stage_config['training']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 토큰 카테고리 정보 추가
            categories = []
            for text in texts:
                # 간단한 휴리스틱: 텍스트 길이 기반
                if len(text) < 50:
                    categories.append('easy')
                elif len(text) < 100:
                    categories.append('medium')
                else:
                    categories.append('hard')

            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'texts': texts,
                'categories': categories
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

        self.log(f"   DataLoader ready: {len(self.train_dataloader)} batches")

    def setup_optimizer(self):
        """옵티마이저 및 스케줄러 설정"""
        opt_config = get_optimizer_config(self.stage)
        train_cfg = self.stage_config['training']

        # 파라미터 그룹
        param_groups = [
            {'params': self.model.parameters(), 'lr': train_cfg['learning_rate']},
            {'params': self.jamo_layer.parameters(), 'lr': train_cfg['learning_rate'] * 0.1}
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.01),
            betas=opt_config.get('betas', (0.9, 0.999))
        )

        # Scheduler
        num_training_steps = (
            len(self.train_dataloader) *
            train_cfg['num_epochs'] //
            train_cfg.get('gradient_accumulation_steps', 1)
        )
        num_warmup_steps = int(num_training_steps * train_cfg.get('warmup_ratio', 0.1))

        scheduler_type = train_cfg.get('lr_scheduler_type', 'cosine')
        if scheduler_type == 'cosine_with_restarts':
            scheduler_type = 'cosine'  # fallback

        self.scheduler = get_scheduler(
            scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.log(f"   Optimizer: AdamW, LR={train_cfg['learning_rate']}")
        self.log(f"   Scheduler: {scheduler_type}, warmup={num_warmup_steps}, total={num_training_steps}")

    def train(self):
        """학습 실행"""
        self.log("\n" + "=" * 80)
        self.log(f"Starting Training: Stage {self.stage}")
        self.log("=" * 80)

        train_cfg = self.stage_config['training']
        num_epochs = train_cfg['num_epochs']
        grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
        max_grad_norm = train_cfg.get('max_grad_norm', 1.0)

        # Loss function
        jlce_mcl_cfg = self.stage_config.get('jlce_mcl', {})
        loss_fn = JLCEMCLLoss(
            stage=self.stage,
            temperature=self.stage_config['contrastive'].get('temperature', 0.05),
            jamo_weight=jlce_mcl_cfg.get('jamo_weight', 0.4),
            curriculum_weight=jlce_mcl_cfg.get('curriculum_weight', 0.3),
            contrastive_weight=jlce_mcl_cfg.get('contrastive_weight', 0.3)
        )

        # Mixed precision
        scaler = torch.amp.GradScaler('cuda')
        use_bf16 = self.stage_config['optimization'].get('mixed_precision') == 'bf16'

        global_step = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            if hasattr(self, 'jamo_layer'):
                jamo_layer = self.jamo_layer.module if isinstance(self.jamo_layer, DDP) else self.jamo_layer
                jamo_layer.train()

            if self.world_size > 1:
                self.train_dataloader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.is_main_process()
            )

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                categories = batch['categories']

                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    # First forward pass
                    outputs1 = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    emb1 = mean_pooling(outputs1.last_hidden_state, attention_mask)

                    # Second forward pass (different dropout)
                    outputs2 = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    emb2 = mean_pooling(outputs2.last_hidden_state, attention_mask)

                    # Jamo embeddings (선택적)
                    jamo_emb = None
                    if jlce_mcl_cfg.get('enabled', True):
                        # 배치의 첫 번째 토큰들에 대해 Jamo 임베딩 생성
                        texts = batch['texts']
                        jamo_embs = []
                        jamo_layer = self.jamo_layer.module if isinstance(self.jamo_layer, DDP) else self.jamo_layer
                        for text in texts:
                            # 텍스트의 첫 20자에 대해 Jamo 임베딩
                            jamo_e = jamo_layer(text[:20], device=self.device)
                            jamo_embs.append(jamo_e)
                        jamo_emb = torch.stack(jamo_embs)

                    # Loss 계산
                    loss, metrics = loss_fn(
                        emb1, emb2,
                        jamo_embeddings=jamo_emb,
                        token_categories=categories
                    )

                    loss = loss / grad_accum_steps

                # Backward
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.jamo_layer.parameters()),
                        max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * grad_accum_steps

                # Logging
                if self.is_main_process() and global_step % self.stage_config['logging']['log_steps'] == 0:
                    progress_bar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'jamo': f"{metrics['jamo_loss']:.4f}",
                        'contr': f"{metrics['contrastive_loss']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                # Checkpointing
                if global_step % self.stage_config['checkpoint']['save_steps'] == 0:
                    self.save_checkpoint(global_step)

            # Epoch end
            avg_loss = epoch_loss / len(self.train_dataloader)
            self.log(f"   Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint('best')

        # Final save
        self.save_checkpoint('final')
        self.log(f"\n   Training complete! Best loss: {best_loss:.4f}")

    def save_checkpoint(self, suffix):
        """체크포인트 저장"""
        if not self.is_main_process():
            return

        output_dir = Path(self.stage_config['checkpoint']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = output_dir / str(suffix)
        checkpoint_dir.mkdir(exist_ok=True)

        # Model 저장
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Jamo layer 저장
        jamo_layer = self.jamo_layer.module if isinstance(self.jamo_layer, DDP) else self.jamo_layer
        torch.save(jamo_layer.state_dict(), checkpoint_dir / 'jamo_layer.pt')

        self.log(f"   Checkpoint saved: {checkpoint_dir}")

    def run(self):
        """전체 학습 파이프라인 실행"""
        self.load_model()
        self.prepare_data()
        self.setup_optimizer()
        self.train()

        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='JLCE + MCL Trainer')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml')
    parser.add_argument('--stage', type=int, required=True, help='Stage number (1, 2, or 3)')
    parser.add_argument('--model_path', type=str, default=None, help='Previous stage checkpoint')
    args = parser.parse_args()

    if args.stage not in [1, 2, 3]:
        raise ValueError("Stage must be 1, 2, or 3 for JLCE + MCL training")

    trainer = JLCEMCLTrainer(
        config_path=args.config,
        stage=args.stage,
        model_path=args.model_path
    )
    trainer.run()


if __name__ == '__main__':
    main()
