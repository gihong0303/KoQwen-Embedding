"""
Retrieval Dataset Loaders

MIRACL, MrTyDi 등 검색 태스크용 데이터셋 로더.
Query-Positive-Negative 형태로 데이터를 제공합니다.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


@dataclass
class RetrievalExample:
    """Single retrieval example"""
    query: str
    positive: str
    negative: Optional[str] = None
    query_id: Optional[str] = None


class MIRACLDataset(Dataset):
    """
    MIRACL Korean Retrieval Dataset

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
        max_query_length: int = 128,
        max_passage_length: int = 384,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            split: Dataset split ("train" or "dev")
            max_samples: Maximum number of samples to load
            include_hard_negatives: Whether to include hard negatives
            num_negatives: Number of negative samples per query
            max_query_length: Maximum query length (for info)
            max_passage_length: Maximum passage length (for info)
            cache_dir: HuggingFace cache directory
        """
        self.split = split
        self.max_samples = max_samples
        self.include_hard_negatives = include_hard_negatives
        self.num_negatives = num_negatives
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

        print(f"Loading MIRACL Korean ({split})...")

        # Load MIRACL Korean
        try:
            self.dataset = load_dataset(
                "miracl/miracl",
                "ko",  # Korean
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading MIRACL: {e}")
            print("Trying alternative loading method...")
            self.dataset = load_dataset(
                "miracl/miracl",
                "ko",
                split=split,
                cache_dir=cache_dir
            )

        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        print(f"Loaded {len(self.dataset)} samples")

        # Preprocess: filter samples with at least one positive
        self.examples = self._preprocess()
        print(f"After filtering: {len(self.examples)} valid examples")

    def _preprocess(self) -> List[Dict]:
        """Preprocess and filter dataset"""
        examples = []

        for item in tqdm(self.dataset, desc="Preprocessing MIRACL"):
            # Skip if no positive passages
            if not item.get("positive_passages"):
                continue

            query = item["query"]
            query_id = item.get("query_id", "")

            # Get positive passages
            positives = item["positive_passages"]

            # Get negative passages (if available and requested)
            negatives = []
            if self.include_hard_negatives and item.get("negative_passages"):
                negatives = item["negative_passages"]

            # Create examples
            for pos in positives:
                pos_text = self._format_passage(pos)

                # Sample negatives
                neg_texts = []
                if negatives and self.include_hard_negatives:
                    sampled_negs = random.sample(
                        negatives,
                        min(self.num_negatives, len(negatives))
                    )
                    neg_texts = [self._format_passage(neg) for neg in sampled_negs]

                examples.append({
                    "query": query,
                    "query_id": query_id,
                    "positive": pos_text,
                    "negatives": neg_texts
                })

        return examples

    def _format_passage(self, passage: Dict) -> str:
        """Format passage as text"""
        title = passage.get("title", "")
        text = passage.get("text", "")

        if title:
            return f"{title}\n{text}"
        return text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        result = {
            "query": example["query"],
            "positive": example["positive"],
        }

        if example["negatives"]:
            result["negative"] = example["negatives"][0]  # First negative
        else:
            result["negative"] = ""

        return result


class MrTyDiDataset(Dataset):
    """
    Mr. TyDi Korean Retrieval Dataset

    Similar structure to MIRACL but different source.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        print(f"Loading Mr. TyDi Korean ({split})...")

        try:
            # Mr. TyDi는 MIRACL과 구조가 유사함
            self.dataset = load_dataset(
                "castorini/mr-tydi",
                "korean",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load Mr. TyDi: {e}")
            print("Using MIRACL as fallback...")
            self.dataset = load_dataset(
                "miracl/miracl",
                "ko",
                split="train",
                cache_dir=cache_dir
            )

        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        self.examples = self._preprocess()
        print(f"Loaded {len(self.examples)} Mr. TyDi examples")

    def _preprocess(self) -> List[Dict]:
        examples = []
        for item in self.dataset:
            if "query" in item and "positive_passages" in item:
                for pos in item["positive_passages"]:
                    examples.append({
                        "query": item["query"],
                        "positive": pos.get("text", ""),
                        "negative": ""
                    })
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class CombinedRetrievalDataset(Dataset):
    """
    Combined Retrieval Dataset from multiple sources

    MIRACL + KorNLI (NLI as retrieval proxy) + optional others
    """

    def __init__(
        self,
        miracl_samples: int = 50000,
        kornli_samples: int = 100000,
        include_hard_negatives: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            miracl_samples: Number of samples from MIRACL
            kornli_samples: Number of samples from KorNLI
            include_hard_negatives: Use MIRACL hard negatives
            cache_dir: Cache directory
        """
        self.examples = []

        # Load MIRACL
        print("Loading MIRACL...")
        try:
            miracl = MIRACLDataset(
                split="train",
                max_samples=miracl_samples,
                include_hard_negatives=include_hard_negatives,
                cache_dir=cache_dir
            )
            self.examples.extend([miracl[i] for i in range(len(miracl))])
            print(f"Added {len(miracl)} MIRACL examples")
        except Exception as e:
            print(f"Warning: Could not load MIRACL: {e}")

        # Load KorNLI (entailment pairs as pseudo retrieval)
        print("Loading KorNLI...")
        try:
            kornli = load_dataset(
                "kakaobrain/kor_nli",
                "snli",
                split="train",
                cache_dir=cache_dir
            )

            # Filter entailment pairs (label == 0) and contradiction as hard negative
            entail_examples = []
            contra_by_premise = {}  # Group contradictions by premise

            for item in kornli:
                if item["label"] == 0:  # Entailment
                    entail_examples.append({
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"]
                    })
                elif item["label"] == 2:  # Contradiction
                    premise = item["premise"]
                    if premise not in contra_by_premise:
                        contra_by_premise[premise] = []
                    contra_by_premise[premise].append(item["hypothesis"])

            # Create retrieval examples from NLI
            for ex in entail_examples[:kornli_samples]:
                neg = ""
                if ex["premise"] in contra_by_premise:
                    negs = contra_by_premise[ex["premise"]]
                    neg = random.choice(negs) if negs else ""

                self.examples.append({
                    "query": ex["premise"],
                    "positive": ex["hypothesis"],
                    "negative": neg
                })

            print(f"Added {min(len(entail_examples), kornli_samples)} KorNLI examples")

        except Exception as e:
            print(f"Warning: Could not load KorNLI: {e}")

        # Shuffle
        random.shuffle(self.examples)
        print(f"Total combined examples: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class RetrievalCollator:
    """
    Collator for retrieval datasets

    Tokenizes query, positive, and negative separately.
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
            # Replace empty negatives with positives (will be masked)
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

        return result


def create_retrieval_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 32,
    max_query_length: int = 128,
    max_passage_length: int = 384,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> DataLoader:
    """
    Create DataLoader for retrieval dataset

    Args:
        dataset: Retrieval dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        max_query_length: Max query length
        max_passage_length: Max passage length
        num_workers: Number of data workers
        shuffle: Whether to shuffle
        distributed: Whether to use DistributedSampler
        rank: Process rank (for DDP)
        world_size: World size (for DDP)

    Returns:
        DataLoader
    """
    from torch.utils.data import DistributedSampler

    collator = RetrievalCollator(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_passage_length=max_passage_length
    )

    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning
    )
