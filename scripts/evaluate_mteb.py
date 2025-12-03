#!/usr/bin/env python3
"""
MTEB Korean Retrieval Evaluation Script

Evaluates the model on 6 Korean retrieval tasks:
1. Ko-StrategyQA (default)
2. AutoRAGRetrieval (default)
3. BelebeleRetrieval (kor→kor)
4. BelebeleRetrieval (kor→eng)
5. BelebeleRetrieval (eng→kor)
6. PublicHealthQA (korean)

Usage:
    python scripts/evaluate_mteb.py --model_path checkpoints/stage5/final
    python scripts/evaluate_mteb.py --model_path Qwen/Qwen3-Embedding-0.6B --baseline
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np

# MTEB imports
try:
    import mteb
    from mteb import MTEB
except ImportError:
    print("Error: mteb not installed. Run: pip install mteb")
    sys.exit(1)

# Sentence Transformers for model loading
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)


# Korean Retrieval Tasks Configuration
# 6 Main Tasks for KAVE Evaluation
KOREAN_RETRIEVAL_TASKS = [
    {"task": "Ko-StrategyQA", "subset": "default"},
    {"task": "AutoRAGRetrieval", "subset": "default"},
    {"task": "MIRACLRetrieval", "subset": "ko"},
    {"task": "PublicHealthQA", "subset": "korean"},
    {"task": "BelebeleRetrieval", "subset": "kor_Hang-kor_Hang"},
    {"task": "MrTidyRetrieval", "subset": "korean"},
]


try:
    from mteb import Encoder
    MTEB_ENCODER_AVAILABLE = True
except ImportError:
    MTEB_ENCODER_AVAILABLE = False


class QwenEmbeddingModel(Encoder if MTEB_ENCODER_AVAILABLE else object):
    """
    Wrapper for Qwen3-Embedding model to work with MTEB
    Inherits from mteb.Encoder for compatibility with new MTEB API
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel

        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences to embeddings"""
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                hidden_states = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode queries (required by MTEB Encoder interface)"""
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[dict], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode corpus (required by MTEB Encoder interface)"""
        if isinstance(corpus[0], dict):
            sentences = [doc.get("text", doc.get("title", "")) for doc in corpus]
        else:
            sentences = corpus
        return self.encode(sentences, batch_size=batch_size, **kwargs)


def run_evaluation(
    model_path: str,
    output_dir: str = "evaluation_results",
    batch_size: int = 64,
    device: str = "cuda:0"
) -> Dict:
    """
    Run MTEB evaluation on Korean retrieval tasks
    """
    print("=" * 80)
    print("MTEB Korean Retrieval Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    if "sentence-transformers" in model_path or Path(model_path).exists():
        # Try loading as custom model
        try:
            model = QwenEmbeddingModel(model_path, device=device)
        except Exception as e:
            print(f"Failed to load as Qwen model: {e}")
            print("Trying SentenceTransformer...")
            model = SentenceTransformer(model_path, device=device)
    else:
        # Try HuggingFace model
        try:
            model = QwenEmbeddingModel(model_path, device=device)
        except Exception as e:
            print(f"Failed to load: {e}")
            raise

    print("Model loaded successfully!")
    print()

    # Run evaluation for each task
    results = {}

    for task_config in KOREAN_RETRIEVAL_TASKS:
        task_name = task_config["task"]
        subset = task_config["subset"]

        print("-" * 60)
        print(f"Task: {task_name} (subset: {subset})")
        print("-" * 60)

        try:
            # Get the task
            if subset != "default":
                tasks = mteb.get_tasks(tasks=[task_name], languages=["kor", "eng"])
            else:
                tasks = mteb.get_tasks(tasks=[task_name])

            if not tasks:
                print(f"  Warning: Task {task_name} not found, skipping...")
                continue

            # Run evaluation
            evaluation = MTEB(tasks=tasks)

            # Run with subset if specified
            if subset != "default":
                eval_results = evaluation.run(
                    model,
                    output_folder=str(output_path / task_name),
                    eval_splits=["test"],
                    batch_size=batch_size
                )
            else:
                eval_results = evaluation.run(
                    model,
                    output_folder=str(output_path / task_name),
                    eval_splits=["test"],
                    batch_size=batch_size
                )

            # Extract NDCG@10
            if eval_results:
                for task_result in eval_results:
                    task_key = f"{task_name}_{subset}"

                    # Try to get NDCG@10
                    if hasattr(task_result, 'scores'):
                        scores = task_result.scores
                        if 'test' in scores:
                            test_scores = scores['test']
                            if isinstance(test_scores, dict):
                                ndcg_10 = test_scores.get('ndcg_at_10',
                                          test_scores.get('main_score', 0))
                            else:
                                ndcg_10 = test_scores
                        else:
                            ndcg_10 = 0
                    else:
                        ndcg_10 = 0

                    results[task_key] = {
                        "ndcg_10": float(ndcg_10),
                        "task": task_name,
                        "subset": subset
                    }
                    print(f"  NDCG@10: {ndcg_10:.4f}")

        except Exception as e:
            print(f"  Error evaluating {task_name}: {e}")
            results[f"{task_name}_{subset}"] = {
                "ndcg_10": 0,
                "task": task_name,
                "subset": subset,
                "error": str(e)
            }

    # Summary
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Task':<30} {'Subset':<20} {'NDCG@10':<10}")
    print("-" * 60)

    total_score = 0
    count = 0

    for key, value in results.items():
        task = value.get('task', key)
        subset = value.get('subset', 'default')
        ndcg = value.get('ndcg_10', 0)

        print(f"{task:<30} {subset:<20} {ndcg:.4f}")
        total_score += ndcg
        count += 1

    avg_score = total_score / count if count > 0 else 0
    print("-" * 60)
    print(f"{'Average':<30} {'':<20} {avg_score:.4f}")
    print()

    # Save results
    results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "average_ndcg_10": avg_score
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {results_file}")

    return results


def compare_models(
    model_paths: List[str],
    output_dir: str = "evaluation_results",
    batch_size: int = 64,
    device: str = "cuda:0"
) -> None:
    """
    Compare multiple models on Korean retrieval tasks
    """
    all_results = {}

    for model_path in model_paths:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_path}")
        print('='*80)

        results = run_evaluation(
            model_path=model_path,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device
        )
        all_results[model_path] = results

    # Comparison table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Task':<35}"
    for path in model_paths:
        name = Path(path).name[:15]
        header += f" {name:<15}"
    print(header)
    print("-" * 100)

    # Get all tasks
    all_tasks = set()
    for results in all_results.values():
        all_tasks.update(results.keys())

    for task in sorted(all_tasks):
        row = f"{task:<35}"
        for model_path in model_paths:
            results = all_results.get(model_path, {})
            ndcg = results.get(task, {}).get('ndcg_10', 0)
            row += f" {ndcg:.4f}        "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="MTEB Korean Retrieval Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Mark this as baseline evaluation"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare multiple models"
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(
            model_paths=[args.model_path] + args.compare,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        run_evaluation(
            model_path=args.model_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device
        )


if __name__ == "__main__":
    main()
