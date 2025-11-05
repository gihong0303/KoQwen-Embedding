"""
Data utilities for loading and processing datasets
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    텍스트 데이터셋 (SimCSE 무지도 학습용)
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # 토크나이징
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


class PairDataset(Dataset):
    """
    Pair 데이터셋 (감독 학습용)
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        max_length: int = 256
    ):
        """
        Args:
            pairs: [{"text1": "...", "text2": "...", "label": 0/1}, ...]
            tokenizer: 토크나이저
            max_length: 최대 길이
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Text 1
        enc1 = self.tokenizer(
            pair['text1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Text 2
        enc2 = self.tokenizer(
            pair['text2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids_1': enc1['input_ids'].squeeze(0),
            'attention_mask_1': enc1['attention_mask'].squeeze(0),
            'input_ids_2': enc2['input_ids'].squeeze(0),
            'attention_mask_2': enc2['attention_mask'].squeeze(0),
            'label': torch.tensor(pair.get('label', 1), dtype=torch.long)
        }


def load_dataset(
    dataset_name: str,
    text_field: str = "text",
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    streaming: bool = False,
    cache_dir: Optional[str] = None
) -> List[str]:
    """
    HuggingFace 데이터셋 로드

    Args:
        dataset_name: 데이터셋 이름
        text_field: 텍스트 필드 이름
        split: 스플릿 ("train", "validation", "test")
        max_samples: 최대 샘플 수
        shuffle: 셔플 여부
        streaming: 스트리밍 모드
        cache_dir: 캐시 디렉토리

    Returns:
        텍스트 리스트
    """
    logger.info("=" * 80)
    logger.info(f"데이터셋 로딩: {dataset_name}")
    logger.info("=" * 80)
    logger.info(f"Split: {split}")
    logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
    logger.info(f"Streaming: {streaming}")

    try:
        # 데이터셋 로드
        dataset = hf_load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )

        # 텍스트 추출
        texts = []

        if streaming:
            # 스트리밍 모드
            for i, example in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                if text_field in example:
                    texts.append(example[text_field])
        else:
            # 일반 모드
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            if shuffle:
                dataset = dataset.shuffle(seed=42)

            texts = [ex[text_field] for ex in tqdm(dataset, desc="텍스트 추출")]

        # 필터링 (빈 텍스트 제거)
        texts = [t for t in texts if t and len(t.strip()) > 0]

        logger.info(f"✓ 로드 완료: {len(texts):,}개 샘플")
        logger.info("")

        return texts

    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    DataLoader 생성

    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        num_workers: Worker 수
        shuffle: 셔플 여부
        pin_memory: Pin memory 사용 여부
        distributed: 분산 학습 여부
        world_size: 전체 프로세스 수
        rank: 현재 프로세스 rank

    Returns:
        DataLoader
    """
    sampler = None

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # sampler가 있으면 shuffle은 False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 마지막 불완전한 배치 제거
    )

    return dataloader


def prepare_batch(batch: Dict, device: torch.device) -> Dict:
    """
    배치를 디바이스로 이동

    Args:
        batch: 배치 딕셔너리
        device: 타겟 디바이스

    Returns:
        디바이스로 이동된 배치
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}
