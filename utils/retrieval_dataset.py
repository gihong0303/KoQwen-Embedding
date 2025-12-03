"""
Retrieval Dataset Loaders (Production Version)

MIRACL, mMARCO, Ko-Triplets 등 다양한 검색 태스크용 데이터셋 로더.
Query-Positive-Negative 형태로 데이터를 제공합니다.

주요 데이터 소스:
1. MIRACL Korean: 네이티브 스피커가 annotate한 hard negatives
2. mMARCO Korean: MS MARCO의 한국어 버전 (대규모)
3. KorNLI: Entailment 쌍을 pseudo-retrieval로 활용
4. Ko-Triplets: 한국어 특화 triplet 데이터

Production 개선사항:
- DDP 환경에서 rank 0만 데이터 다운로드
- Reproducible shuffling with seed
- 메모리 효율적인 lazy loading
- 에러 핸들링 강화
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class RetrievalExample:
    """Single retrieval example"""
    query: str
    positive: str
    negative: Optional[str] = None
    query_id: Optional[str] = None


def is_main_process() -> bool:
    """Check if current process is main (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def wait_for_data_ready():
    """Synchronize all processes after data loading"""
    if dist.is_initialized():
        dist.barrier()


class MIRACLDataset(Dataset):
    """
    MIRACL Korean Retrieval Dataset (Production)

    Structure:
        - query_id: str
        - query: str
        - positive_passages: List[Dict] with docid, title, text
        - negative_passages: List[Dict] with docid, title, text (native speaker annotated)

    Reference: https://huggingface.co/datasets/miracl/miracl
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_hard_negatives: bool = True,
        num_negatives: int = 1,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        self.split = split
        self.max_samples = max_samples
        self.include_hard_negatives = include_hard_negatives
        self.num_negatives = num_negatives
        self.seed = seed
        self.examples = []

        # Set seed for reproducibility
        random.seed(seed)

        # Only rank 0 loads data
        if is_main_process():
            self._load_data(cache_dir)

        # Sync across processes
        wait_for_data_ready()

        # Broadcast data to other ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._broadcast_examples()

    def _load_data(self, cache_dir: Optional[str]):
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info(f"Loading MIRACL Korean ({self.split})...")

        try:
            dataset = load_dataset(
                "miracl/miracl",
                "ko",
                split=self.split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Error loading MIRACL: {e}")
            return

        if self.max_samples and len(dataset) > self.max_samples:
            indices = random.sample(range(len(dataset)), self.max_samples)
            dataset = dataset.select(indices)

        logger.info(f"Processing {len(dataset)} MIRACL samples...")

        for item in tqdm(dataset, desc="MIRACL", disable=not is_main_process()):
            if not item.get("positive_passages"):
                continue

            query = item["query"]
            query_id = item.get("query_id", "")
            positives = item["positive_passages"]
            negatives = item.get("negative_passages", []) if self.include_hard_negatives else []

            for pos in positives:
                pos_text = self._format_passage(pos)

                neg_text = ""
                if negatives:
                    sampled_neg = random.choice(negatives)
                    neg_text = self._format_passage(sampled_neg)

                self.examples.append({
                    "query": query,
                    "query_id": query_id,
                    "positive": pos_text,
                    "negative": neg_text
                })

        logger.info(f"MIRACL: {len(self.examples)} examples loaded")

    def _broadcast_examples(self):
        """Broadcast examples from rank 0 to all other ranks"""
        import pickle

        if dist.get_rank() == 0:
            data = pickle.dumps(self.examples)
            size = torch.tensor([len(data)], dtype=torch.long, device='cuda')
        else:
            size = torch.tensor([0], dtype=torch.long, device='cuda')

        dist.broadcast(size, src=0)

        if dist.get_rank() == 0:
            data_tensor = torch.ByteTensor(list(data)).cuda()
        else:
            data_tensor = torch.ByteTensor(size.item()).cuda()

        dist.broadcast(data_tensor, src=0)

        if dist.get_rank() != 0:
            data = bytes(data_tensor.cpu().numpy())
            self.examples = pickle.loads(data)

    def _format_passage(self, passage: Dict) -> str:
        title = passage.get("title", "")
        text = passage.get("text", "")
        if title:
            return f"{title}\n{text}"
        return text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class MMarcoKoreanDataset(Dataset):
    """
    mMARCO Korean Dataset (Production)

    MS MARCO의 한국어 번역 버전. 대규모 검색 데이터.

    Reference: https://huggingface.co/datasets/unicamp-dl/mmarco
    """

    def __init__(
        self,
        max_samples: int = 100000,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        self.max_samples = max_samples
        self.seed = seed
        self.examples = []

        random.seed(seed)

        if is_main_process():
            self._load_data(cache_dir)

        wait_for_data_ready()

        if dist.is_initialized() and dist.get_world_size() > 1:
            self._broadcast_examples()

    def _load_data(self, cache_dir: Optional[str]):
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("Loading mMARCO Korean...")

        try:
            # mMARCO Korean triples (query, positive, negative)
            dataset = load_dataset(
                "unicamp-dl/mmarco",
                "korean",
                split="train",
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            logger.info(f"mMARCO total: {len(dataset)}, sampling {self.max_samples}")

            if len(dataset) > self.max_samples:
                indices = random.sample(range(len(dataset)), self.max_samples)
                dataset = dataset.select(indices)

            for item in tqdm(dataset, desc="mMARCO", disable=not is_main_process()):
                self.examples.append({
                    "query": item.get("query", ""),
                    "positive": item.get("positive", ""),
                    "negative": item.get("negative", "")
                })

            logger.info(f"mMARCO: {len(self.examples)} examples loaded")

        except Exception as e:
            logger.warning(f"Could not load mMARCO: {e}")
            logger.info("mMARCO may require authentication. Skipping...")

    def _broadcast_examples(self):
        import pickle

        if dist.get_rank() == 0:
            data = pickle.dumps(self.examples)
            size = torch.tensor([len(data)], dtype=torch.long, device='cuda')
        else:
            size = torch.tensor([0], dtype=torch.long, device='cuda')

        dist.broadcast(size, src=0)

        if dist.get_rank() == 0:
            data_tensor = torch.ByteTensor(list(data)).cuda()
        else:
            data_tensor = torch.ByteTensor(size.item()).cuda()

        dist.broadcast(data_tensor, src=0)

        if dist.get_rank() != 0:
            data = bytes(data_tensor.cpu().numpy())
            self.examples = pickle.loads(data)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class KorNLIRetrievalDataset(Dataset):
    """
    KorNLI as Retrieval Dataset (Production)

    NLI 데이터를 pseudo-retrieval로 활용:
    - Entailment: premise → hypothesis (positive pair)
    - Contradiction: premise의 contradiction을 hard negative로 사용
    """

    def __init__(
        self,
        max_samples: int = 100000,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        self.max_samples = max_samples
        self.seed = seed
        self.examples = []

        random.seed(seed)

        if is_main_process():
            self._load_data(cache_dir)

        wait_for_data_ready()

        if dist.is_initialized() and dist.get_world_size() > 1:
            self._broadcast_examples()

    def _load_data(self, cache_dir: Optional[str]):
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("Loading KorNLI...")

        try:
            kornli = load_dataset(
                "kakaobrain/kor_nli",
                "snli",
                split="train",
                cache_dir=cache_dir
            )

            # Group by label
            entailment_pairs = []
            contradiction_by_premise = {}

            for item in tqdm(kornli, desc="KorNLI Grouping", disable=not is_main_process()):
                premise = item["premise"]
                hypothesis = item["hypothesis"]
                label = item["label"]

                if label == 0:  # Entailment
                    entailment_pairs.append({
                        "premise": premise,
                        "hypothesis": hypothesis
                    })
                elif label == 2:  # Contradiction
                    if premise not in contradiction_by_premise:
                        contradiction_by_premise[premise] = []
                    contradiction_by_premise[premise].append(hypothesis)

            # Create retrieval examples
            sampled = random.sample(entailment_pairs, min(self.max_samples, len(entailment_pairs)))

            for ex in sampled:
                neg = ""
                if ex["premise"] in contradiction_by_premise:
                    negs = contradiction_by_premise[ex["premise"]]
                    neg = random.choice(negs) if negs else ""

                self.examples.append({
                    "query": ex["premise"],
                    "positive": ex["hypothesis"],
                    "negative": neg
                })

            logger.info(f"KorNLI: {len(self.examples)} examples loaded")

        except Exception as e:
            logger.warning(f"Could not load KorNLI: {e}")

    def _broadcast_examples(self):
        import pickle

        if dist.get_rank() == 0:
            data = pickle.dumps(self.examples)
            size = torch.tensor([len(data)], dtype=torch.long, device='cuda')
        else:
            size = torch.tensor([0], dtype=torch.long, device='cuda')

        dist.broadcast(size, src=0)

        if dist.get_rank() == 0:
            data_tensor = torch.ByteTensor(list(data)).cuda()
        else:
            data_tensor = torch.ByteTensor(size.item()).cuda()

        dist.broadcast(data_tensor, src=0)

        if dist.get_rank() != 0:
            data = bytes(data_tensor.cpu().numpy())
            self.examples = pickle.loads(data)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class CombinedRetrievalDataset(Dataset):
    """
    Combined Retrieval Dataset (Production)

    여러 소스의 검색 데이터를 결합:
    - MIRACL Korean: Hard negatives (high quality)
    - mMARCO Korean: Large scale (quantity)
    - KorNLI: Semantic similarity (diversity)

    Args:
        miracl_samples: MIRACL 샘플 수 (기본 50K, MIRACL train은 작음)
        mmarco_samples: mMARCO 샘플 수 (기본 100K)
        kornli_samples: KorNLI 샘플 수 (기본 100K)
        include_hard_negatives: Hard negative 포함 여부
        seed: Random seed for reproducibility
        cache_dir: HuggingFace cache directory
    """

    def __init__(
        self,
        miracl_samples: int = 50000,
        mmarco_samples: int = 100000,
        kornli_samples: int = 100000,
        include_hard_negatives: bool = True,
        seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        self.seed = seed
        random.seed(seed)

        self.examples = []

        # 1. Load MIRACL (high quality hard negatives)
        try:
            miracl = MIRACLDataset(
                split="train",
                max_samples=miracl_samples,
                include_hard_negatives=include_hard_negatives,
                seed=seed,
                cache_dir=cache_dir
            )
            self.examples.extend([miracl[i] for i in range(len(miracl))])
            if is_main_process():
                logger.info(f"Added {len(miracl)} MIRACL examples")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Could not load MIRACL: {e}")

        # 2. Load mMARCO (large scale)
        try:
            mmarco = MMarcoKoreanDataset(
                max_samples=mmarco_samples,
                seed=seed,
                cache_dir=cache_dir
            )
            self.examples.extend([mmarco[i] for i in range(len(mmarco))])
            if is_main_process():
                logger.info(f"Added {len(mmarco)} mMARCO examples")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Could not load mMARCO: {e}")

        # 3. Load KorNLI (semantic diversity)
        try:
            kornli = KorNLIRetrievalDataset(
                max_samples=kornli_samples,
                seed=seed,
                cache_dir=cache_dir
            )
            self.examples.extend([kornli[i] for i in range(len(kornli))])
            if is_main_process():
                logger.info(f"Added {len(kornli)} KorNLI examples")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Could not load KorNLI: {e}")

        # Shuffle with seed for reproducibility
        random.seed(seed)
        random.shuffle(self.examples)

        if is_main_process():
            logger.info(f"Total combined examples: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class RetrievalCollator:
    """
    Collator for retrieval datasets (Production)

    Tokenizes query, positive, and negative separately.
    Handles variable-length sequences efficiently.
    """

    def __init__(
        self,
        tokenizer,
        max_query_length: int = 128,
        max_passage_length: int = 384
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        queries = [ex["query"] for ex in examples]
        positives = [ex["positive"] for ex in examples]
        negatives = [ex.get("negative", "") for ex in examples]

        # Tokenize queries
        query_encodings = self.tokenizer(
            queries,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize positives
        pos_encodings = self.tokenizer(
            positives,
            max_length=self.max_passage_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        result = {
            "query_input_ids": query_encodings["input_ids"],
            "query_attention_mask": query_encodings["attention_mask"],
            "pos_input_ids": pos_encodings["input_ids"],
            "pos_attention_mask": pos_encodings["attention_mask"],
        }

        # Tokenize negatives if available
        has_negatives = any(neg for neg in negatives)
        if has_negatives:
            # Replace empty negatives with a placeholder (will use in-batch negatives)
            negatives_filled = [neg if neg else pos for neg, pos in zip(negatives, positives)]

            neg_encodings = self.tokenizer(
                negatives_filled,
                max_length=self.max_passage_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            result["neg_input_ids"] = neg_encodings["input_ids"]
            result["neg_attention_mask"] = neg_encodings["attention_mask"]

            # Mask for valid negatives (not placeholders)
            result["neg_valid_mask"] = torch.tensor(
                [1 if neg else 0 for neg in negatives],
                dtype=torch.bool
            )

        return result


def create_retrieval_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 32,
    max_query_length: int = 128,
    max_passage_length: int = 384,
    num_workers: int = 4,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42
) -> DataLoader:
    """
    Create DataLoader for retrieval dataset (Production)

    Handles DDP with proper DistributedSampler.
    """
    collator = RetrievalCollator(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_passage_length=max_passage_length
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
