#!/usr/bin/env python3
"""
Evaluation Script: CLSA Model vs Baseline
Compare performance on MTEB Korean retrieval tasks
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModelEvaluator:
    """
    Evaluate embedding models on Korean retrieval tasks
    """

    def __init__(self, model_path: str, model_type: str = "qwen"):
        """
        Args:
            model_path: Path to model checkpoint
            model_type: 'qwen' or 'sentence-transformers'
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from: {model_path}")

        if model_type == "qwen":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        else:
            self.model = SentenceTransformer(model_path, device=self.device)

        logger.info(f"Model loaded on: {self.device}")

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            embeddings: numpy array [num_texts, hidden_dim]
        """
        if self.model_type == "sentence-transformers":
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

        # Qwen model
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Forward pass
            outputs = self.model(**encodings)

            # Mean pooling
            last_hidden = outputs.last_hidden_state
            attention_mask = encodings['attention_mask']

            # Expand mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

            # Mean pooling
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask

            # Normalize
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def compute_similarity(self, queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between queries and documents

        Args:
            queries: [num_queries, dim]
            documents: [num_docs, dim]

        Returns:
            similarities: [num_queries, num_docs]
        """
        # Normalize
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
        documents = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-9)

        # Compute similarity
        similarities = np.dot(queries, documents.T)

        return similarities


def evaluate_simple_test_set():
    """
    Simple evaluation on predefined Korean test pairs
    """
    # Simple Korean test pairs
    test_pairs = [
        {
            "query": "한국의 수도는 어디인가요?",
            "positive": "대한민국의 수도는 서울특별시입니다.",
            "negative": "일본의 수도는 도쿄입니다."
        },
        {
            "query": "머신러닝이란 무엇인가요?",
            "positive": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하는 기술입니다.",
            "negative": "딥러닝은 신경망을 사용하여 복잡한 패턴을 학습합니다."
        },
        {
            "query": "김치는 어떤 음식인가요?",
            "positive": "김치는 배추나 무 등의 채소를 소금에 절여 고춧가루 등의 양념으로 버무려 발효시킨 한국의 전통 음식입니다.",
            "negative": "피자는 이탈리아에서 유래한 음식입니다."
        },
        {
            "query": "딥러닝 모델을 학습시키는 방법",
            "positive": "딥러닝 모델은 대량의 데이터를 사용하여 역전파 알고리즘으로 가중치를 업데이트하며 학습합니다.",
            "negative": "머신러닝은 통계적 방법을 사용합니다."
        },
        {
            "query": "임베딩이란?",
            "positive": "임베딩은 텍스트나 이미지를 벡터로 변환하여 수치적으로 표현하는 기술입니다.",
            "negative": "토큰화는 텍스트를 작은 단위로 나누는 과정입니다."
        }
    ]

    return test_pairs


def compare_models(
    baseline_path: str,
    clsa_path: str,
    test_pairs: List[Dict] = None
):
    """
    Compare baseline and CLSA models

    Args:
        baseline_path: Path to baseline model
        clsa_path: Path to CLSA model
        test_pairs: Test query-document pairs (optional)
    """
    logger.info("=" * 80)
    logger.info("Model Comparison: Baseline vs CLSA")
    logger.info("=" * 80)

    # Load models
    logger.info("\n📥 Loading models...")
    baseline_model = EmbeddingModelEvaluator(baseline_path, model_type="qwen")
    clsa_model = EmbeddingModelEvaluator(clsa_path, model_type="qwen")

    # Use default test set if not provided
    if test_pairs is None:
        test_pairs = evaluate_simple_test_set()

    logger.info(f"\n📊 Evaluating on {len(test_pairs)} test pairs...")

    # Extract queries and documents
    queries = [pair['query'] for pair in test_pairs]
    positives = [pair['positive'] for pair in test_pairs]
    negatives = [pair['negative'] for pair in test_pairs]

    # Combine all documents
    all_docs = positives + negatives

    # Encode with baseline
    logger.info("\n🔵 Encoding with Baseline model...")
    baseline_query_embs = baseline_model.encode(queries)
    baseline_doc_embs = baseline_model.encode(all_docs)

    # Encode with CLSA
    logger.info("\n🟢 Encoding with CLSA model...")
    clsa_query_embs = clsa_model.encode(queries)
    clsa_doc_embs = clsa_model.encode(all_docs)

    # Compute similarities
    baseline_sims = baseline_model.compute_similarity(baseline_query_embs, baseline_doc_embs)
    clsa_sims = clsa_model.compute_similarity(clsa_query_embs, clsa_doc_embs)

    # Evaluate accuracy
    num_queries = len(queries)
    baseline_correct = 0
    clsa_correct = 0

    results = []

    for i in range(num_queries):
        pos_idx = i
        neg_idx = i + num_queries

        baseline_pos_sim = baseline_sims[i, pos_idx]
        baseline_neg_sim = baseline_sims[i, neg_idx]
        baseline_is_correct = baseline_pos_sim > baseline_neg_sim

        clsa_pos_sim = clsa_sims[i, pos_idx]
        clsa_neg_sim = clsa_sims[i, neg_idx]
        clsa_is_correct = clsa_pos_sim > clsa_neg_sim

        baseline_correct += int(baseline_is_correct)
        clsa_correct += int(clsa_is_correct)

        results.append({
            'query': queries[i],
            'positive': positives[i],
            'negative': negatives[i],
            'baseline_pos_sim': float(baseline_pos_sim),
            'baseline_neg_sim': float(baseline_neg_sim),
            'baseline_correct': baseline_is_correct,
            'clsa_pos_sim': float(clsa_pos_sim),
            'clsa_neg_sim': float(clsa_neg_sim),
            'clsa_correct': clsa_is_correct,
            'improvement': clsa_is_correct and not baseline_is_correct
        })

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"\n📊 Baseline Accuracy: {baseline_correct}/{num_queries} ({100*baseline_correct/num_queries:.1f}%)")
    logger.info(f"📊 CLSA Accuracy: {clsa_correct}/{num_queries} ({100*clsa_correct/num_queries:.1f}%)")
    logger.info(f"\n🎯 Improvement: +{clsa_correct - baseline_correct} correct ({100*(clsa_correct-baseline_correct)/num_queries:+.1f}%)")

    # Detailed results
    logger.info("\n" + "=" * 80)
    logger.info("Detailed Results")
    logger.info("=" * 80)

    for i, result in enumerate(results):
        logger.info(f"\nQuery {i+1}: {result['query'][:60]}...")
        logger.info(f"  Baseline: Pos={result['baseline_pos_sim']:.3f}, Neg={result['baseline_neg_sim']:.3f} {'✅' if result['baseline_correct'] else '❌'}")
        logger.info(f"  CLSA:     Pos={result['clsa_pos_sim']:.3f}, Neg={result['clsa_neg_sim']:.3f} {'✅' if result['clsa_correct'] else '❌'}")
        if result['improvement']:
            logger.info(f"  🎉 Improvement!")

    # Save results
    output_file = "outputs/clsa_comparison_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_accuracy': baseline_correct / num_queries,
            'clsa_accuracy': clsa_correct / num_queries,
            'improvement': (clsa_correct - baseline_correct) / num_queries,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline and CLSA models")
    parser.add_argument(
        "--baseline",
        type=str,
        default="checkpoints/stage6/final",
        help="Path to baseline model (original pipeline)"
    )
    parser.add_argument(
        "--clsa",
        type=str,
        default="checkpoints/stage6/final",  # Will be from CLSA pipeline
        help="Path to CLSA model (CLSA + Curriculum pipeline)"
    )

    args = parser.parse_args()

    compare_models(
        baseline_path=args.baseline,
        clsa_path=args.clsa
    )


if __name__ == "__main__":
    main()
