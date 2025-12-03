#!/usr/bin/env python3
"""
MTEB Korean Retrieval Evaluation Script (Parallel Version)

GPU별로 병렬 평가를 수행하여 MIRACL, MrTidy 같은 시간이 오래 걸리는 태스크를
효율적으로 처리합니다.

Usage:
    python scripts/evaluate_mteb_parallel.py --model_path checkpoints/stage6/final
    python scripts/evaluate_mteb_parallel.py --model_path Qwen/Qwen3-Embedding-0.6B --baseline
    python scripts/evaluate_mteb_parallel.py --model_path checkpoints/stage6/final --compare Qwen/Qwen3-Embedding-0.6B

Reference: Based on user-provided multiprocessing evaluation code
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, current_process
from typing import Dict, List, Optional

import torch
import numpy as np

# MTEB imports
try:
    import mteb
    from mteb import MTEB, get_tasks
    from mteb.encoder_interface import PromptType
except ImportError:
    print("Error: mteb not installed. Run: pip install mteb>=1.12")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import StaticEmbedding
except ImportError:
    print("Error: sentence-transformers not installed. Run: pip install sentence-transformers>=2.7")
    sys.exit(1)

try:
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
except ImportError:
    SentenceTransformerWrapper = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mteb_eval")


# ============================================================================
# Task Configuration
# ============================================================================

# 6개 핵심 한국어 검색 태스크 (main/main-v2 비교용)
TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval",      # 시간이 오래 걸림
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval",      # 시간이 오래 걸림
]

# GPU별 태스크 분배 (MIRACL, MrTidy는 별도 GPU에서 실행)
TASK_GPU_MAPPING = {
    0: [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
    ],
    1: ["MIRACLRetrieval"],
    2: ["MrTidyRetrieval"],
}


# ============================================================================
# Model Wrapper for Qwen3-Embedding
# ============================================================================

class QwenEmbeddingModel:
    """
    Wrapper for Qwen3-Embedding model compatible with MTEB

    Supports both HuggingFace models and local checkpoints.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel

        self.device = device
        self.model_path = model_path

        print(f"Loading model from: {model_path}")

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

        print(f"Model loaded successfully on {device}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences to embeddings with mean pooling"""
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


# ============================================================================
# Evaluation Functions
# ============================================================================

def get_batch_size_for_model(model_name: str) -> int:
    """모델별 최적 배치 크기 반환"""
    model_lower = model_name.lower()

    if "multilingual-e5" in model_lower or "koe5" in model_lower:
        return 512
    elif "jina" in model_lower:
        return 8
    elif "bge-m3" in model_lower or "snowflake" in model_lower:
        return 32
    elif "gemma" in model_lower:
        return 256
    elif "salesforce" in model_lower:
        return 128
    else:
        return 64


def load_model(model_path: str, device: str = "cuda"):
    """
    Load model with appropriate wrapper

    Supports:
    - Local checkpoints (Qwen3-Embedding based)
    - HuggingFace models
    - Sentence-transformers models
    """
    # Check if local path exists
    if os.path.exists(model_path):
        model_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(model_file):
            return QwenEmbeddingModel(model_path, device=device)

    # Try loading as Qwen model from HF
    try:
        return QwenEmbeddingModel(model_path, device=device)
    except Exception as e:
        logger.warning(f"Could not load as Qwen model: {e}")

    # Fallback to sentence-transformers
    try:
        return SentenceTransformer(model_path, device=device)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        raise


def evaluate_model_on_tasks(
    model_path: str,
    gpu_id: int,
    tasks: List[str],
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    특정 GPU에서 주어진 태스크들을 평가

    Args:
        model_path: 모델 경로
        gpu_id: GPU ID
        tasks: 평가할 태스크 리스트
        output_dir: 결과 저장 디렉토리

    Returns:
        results: 태스크별 결과 딕셔너리
    """
    try:
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"

        process_name = current_process().name
        print(f"[{process_name}] GPU {gpu_id}: Evaluating {tasks}")

        # Load model
        model = load_model(model_path, device=device)
        batch_size = get_batch_size_for_model(model_path)

        results = {}

        for task_name in tasks:
            print(f"[{process_name}] Starting task: {task_name}")

            try:
                # Get task with Korean language filter
                mteb_tasks = get_tasks(
                    tasks=[task_name],
                    languages=["kor-Kore", "kor-Hang", "kor_Hang", "ko"]
                )

                if not mteb_tasks:
                    # Try without language filter
                    mteb_tasks = get_tasks(tasks=[task_name])

                if not mteb_tasks:
                    print(f"[{process_name}] Warning: Task {task_name} not found, skipping...")
                    continue

                # Run evaluation
                evaluation = MTEB(tasks=mteb_tasks)
                eval_results = evaluation.run(
                    model,
                    output_folder=os.path.join(output_dir, model_path.replace("/", "_"), task_name),
                    eval_splits=["test"],
                    encode_kwargs={"batch_size": batch_size}
                )

                # Extract NDCG@10
                if eval_results:
                    for task_result in eval_results:
                        ndcg_10 = 0.0

                        if hasattr(task_result, 'scores'):
                            scores = task_result.scores
                            if 'test' in scores:
                                test_scores = scores['test']
                                if isinstance(test_scores, dict):
                                    # Try different score keys
                                    for key in ['ndcg_at_10', 'main_score', 'ndcg@10']:
                                        if key in test_scores:
                                            if isinstance(test_scores[key], dict):
                                                # Handle subset scores
                                                ndcg_10 = list(test_scores[key].values())[0]
                                            else:
                                                ndcg_10 = test_scores[key]
                                            break

                        results[task_name] = {
                            "ndcg_10": float(ndcg_10),
                            "gpu_id": gpu_id
                        }
                        print(f"[{process_name}] {task_name}: NDCG@10 = {ndcg_10:.4f}")

            except Exception as e:
                print(f"[{process_name}] Error evaluating {task_name}: {e}")
                traceback.print_exc()
                results[task_name] = {
                    "ndcg_10": 0.0,
                    "error": str(e),
                    "gpu_id": gpu_id
                }

        return results

    except Exception as e:
        print(f"Error in process: {e}")
        traceback.print_exc()
        return {}


def run_parallel_evaluation(
    model_path: str,
    output_dir: str = "evaluation_results",
    task_gpu_mapping: Dict[int, List[str]] = None
) -> Dict:
    """
    병렬 평가 실행

    각 GPU에서 할당된 태스크를 동시에 평가합니다.
    """
    if task_gpu_mapping is None:
        task_gpu_mapping = TASK_GPU_MAPPING

    print("=" * 80)
    print("MTEB Korean Retrieval Evaluation (Parallel)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    print("GPU Task Mapping:")
    for gpu_id, tasks in task_gpu_mapping.items():
        print(f"  GPU {gpu_id}: {tasks}")
    print()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Start processes
    processes = []
    for gpu_id, tasks in task_gpu_mapping.items():
        p = Process(
            target=evaluate_model_on_tasks,
            args=(model_path, gpu_id, tasks, output_dir),
            name=f"GPU-{gpu_id}"
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    print("\n" + "=" * 80)
    print("Parallel evaluation complete!")
    print("=" * 80)

    # Aggregate results (results are saved by each process)
    return {}


def run_sequential_evaluation(
    model_path: str,
    output_dir: str = "evaluation_results",
    device: str = "cuda:0"
) -> Dict:
    """
    순차 평가 실행 (단일 GPU)
    """
    print("=" * 80)
    print("MTEB Korean Retrieval Evaluation (Sequential)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(model_path, device=device)
    batch_size = get_batch_size_for_model(model_path)
    print(f"Model loaded. Batch size: {batch_size}")

    results = {}

    for task_name in TASK_LIST_RETRIEVAL:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print('='*60)

        try:
            mteb_tasks = get_tasks(
                tasks=[task_name],
                languages=["kor-Kore", "kor-Hang", "kor_Hang", "ko"]
            )

            if not mteb_tasks:
                mteb_tasks = get_tasks(tasks=[task_name])

            if not mteb_tasks:
                print(f"  Warning: Task not found, skipping...")
                continue

            evaluation = MTEB(tasks=mteb_tasks)
            eval_results = evaluation.run(
                model,
                output_folder=os.path.join(output_dir, task_name),
                eval_splits=["test"],
                encode_kwargs={"batch_size": batch_size}
            )

            if eval_results:
                for task_result in eval_results:
                    ndcg_10 = 0.0
                    if hasattr(task_result, 'scores'):
                        scores = task_result.scores
                        if 'test' in scores:
                            test_scores = scores['test']
                            if isinstance(test_scores, dict):
                                for key in ['ndcg_at_10', 'main_score', 'ndcg@10']:
                                    if key in test_scores:
                                        if isinstance(test_scores[key], dict):
                                            ndcg_10 = list(test_scores[key].values())[0]
                                        else:
                                            ndcg_10 = test_scores[key]
                                        break

                    results[task_name] = {"ndcg_10": float(ndcg_10)}
                    print(f"  NDCG@10: {ndcg_10:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            results[task_name] = {"ndcg_10": 0.0, "error": str(e)}

    # Print summary
    print_summary(results, model_path)

    # Save results
    save_results(results, model_path, output_dir)

    return results


def print_summary(results: Dict, model_path: str):
    """결과 요약 출력"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"\n{'Task':<30} {'NDCG@10':<10}")
    print("-" * 40)

    total = 0
    count = 0

    for task, data in results.items():
        ndcg = data.get('ndcg_10', 0)
        print(f"{task:<30} {ndcg:.4f}")
        total += ndcg
        count += 1

    avg = total / count if count > 0 else 0
    print("-" * 40)
    print(f"{'Average':<30} {avg:.4f}")


def save_results(results: Dict, model_path: str, output_dir: str):
    """결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name if os.path.exists(model_path) else model_path.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_path / f"results_{model_name}_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "average_ndcg_10": sum(r.get('ndcg_10', 0) for r in results.values()) / len(results) if results else 0
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


def compare_models(model_paths: List[str], output_dir: str = "evaluation_results"):
    """여러 모델 비교"""
    all_results = {}

    for model_path in model_paths:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_path}")
        print('='*80)

        results = run_sequential_evaluation(
            model_path=model_path,
            output_dir=output_dir
        )
        all_results[model_path] = results

    # Comparison table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Task':<25}"
    for path in model_paths:
        name = Path(path).name[:20] if os.path.exists(path) else path.split("/")[-1][:20]
        header += f" {name:<20}"
    print(header)
    print("-" * 100)

    # Get all tasks
    all_tasks = set()
    for results in all_results.values():
        all_tasks.update(results.keys())

    for task in sorted(all_tasks):
        row = f"{task:<25}"
        for model_path in model_paths:
            results = all_results.get(model_path, {})
            ndcg = results.get(task, {}).get('ndcg_10', 0)
            row += f" {ndcg:.4f}              "
        print(row)

    # Averages
    print("-" * 100)
    row = f"{'Average':<25}"
    for model_path in model_paths:
        results = all_results.get(model_path, {})
        avg = sum(r.get('ndcg_10', 0) for r in results.values()) / len(results) if results else 0
        row += f" {avg:.4f}              "
    print(row)


def main():
    parser = argparse.ArgumentParser(description="MTEB Korean Retrieval Evaluation (Parallel)")
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
        "--parallel",
        action="store_true",
        help="Run evaluation in parallel across GPUs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for sequential evaluation"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare with other models"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Mark as baseline evaluation"
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(
            model_paths=[args.model_path] + args.compare,
            output_dir=args.output_dir
        )
    elif args.parallel:
        run_parallel_evaluation(
            model_path=args.model_path,
            output_dir=args.output_dir
        )
    else:
        run_sequential_evaluation(
            model_path=args.model_path,
            output_dir=args.output_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()
