#!/usr/bin/env python3
"""
06_evaluate_models_correct.py

CORRECT evaluation: Load previous stage's base + current stage's adapters
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Dict
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModel, AutoTokenizer

# Import adapter utils
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eeve_adapter",
    PROJECT_ROOT / "scripts/utils/eeve_adapter.py"
)
eeve_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eeve_adapter)
inject_adapters = eeve_adapter.inject_adapters


# Test dataset
SIMILAR_PAIRS = [
    ("오늘 날씨가 정말 좋네요.", "날씨가 매우 화창합니다."),
    ("인공지능 기술이 빠르게 발전하고 있다.", "AI 기술의 발전 속도가 빠르다."),
    ("저는 커피를 매우 좋아합니다.", "커피를 마시는 것을 정말 즐깁니다."),
    ("삼성전자가 새로운 스마트폰을 출시했다.", "삼성이 신형 휴대폰을 공개했다."),
    ("이 영화는 정말 재미있었어요.", "이 영화 진짜 재밌었어요."),
    ("주식 시장이 오늘 크게 상승했다.", "오늘 증시가 급등했다."),
    ("건강을 위해 매일 운동하고 있어요.", "건강 관리를 위해 날마다 운동 중이에요."),
    ("이 음식은 너무 맵습니다.", "이 요리는 매우 맵네요."),
    ("서울의 교통 체증이 심각하다.", "서울 시내 교통 정체가 매우 심하다."),
    ("파이썬은 배우기 쉬운 프로그래밍 언어다.", "파이썬은 초보자가 학습하기 좋은 언어다."),
]

DIFFERENT_PAIRS = [
    ("오늘 날씨가 정말 좋네요.", "블록체인 기술의 미래는 밝다."),
    ("인공지능 기술이 빠르게 발전하고 있다.", "저는 피자를 좋아합니다."),
    ("저는 커피를 매우 좋아합니다.", "한국의 경제 성장률이 둔화되고 있다."),
    ("삼성전자가 새로운 스마트폰을 출시했다.", "고양이는 귀여운 동물이다."),
    ("이 영화는 정말 재미있었어요.", "수학 공식을 외우는 것이 어렵다."),
    ("주식 시장이 오늘 크게 상승했다.", "바다에서 서핑을 즐기고 있어요."),
    ("건강을 위해 매일 운동하고 있어요.", "자동차 가격이 계속 오르고 있다."),
    ("이 음식은 너무 맵습니다.", "인터넷 속도가 매우 빠르다."),
    ("서울의 교통 체증이 심각하다.", "식물에 물을 주는 것을 잊었다."),
    ("파이썬은 배우기 쉬운 프로그래밍 언어다.", "여름 휴가를 제주도에서 보냈다."),
]


def load_model_and_tokenizer(
    model_path: str,
    base_model_path: str = None,
    load_adapters: bool = False,
    adapter_type: str = "bottleneck"
):
    """
    CORRECT model loading:
    - For adapter stages: Load BASE from previous stage, inject adapters, load adapter weights from current stage
    """
    print(f"Loading model from: {model_path}")

    if not load_adapters:
        # Non-adapter stages (Stage 0, Stage 1)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    else:
        # Adapter stages (Stage 2, Stage 3)
        print(f"  Base model: {base_model_path}")
        print(f"  Adapter type: {adapter_type}")

        # Load BASE model from previous stage
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model_path,  # Load from PREVIOUS stage!
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Inject adapter structure
        print(f"  Injecting {adapter_type} adapters...")
        model = inject_adapters(
            model,
            adapter_type=adapter_type,
            adapter_size=256,
            dropout=0.1,
            layer_indices=None
        )

        # Load adapter weights from CURRENT stage
        print(f"  Loading adapter weights from {model_path}...")
        from safetensors import safe_open
        state_dict = {}
        safetensors_path = Path(model_path) / "model.safetensors"

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if 'adapter' in key:
                    state_dict[key] = f.get_tensor(key)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded {len(state_dict)} adapter parameters")

    model.eval()
    model = model.cuda()
    return model, tokenizer


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(texts: List[str], model, tokenizer, max_length: int = 512) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = mean_pooling(hidden_states, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return F.cosine_similarity(emb1, emb2, dim=1).item()


def evaluate_model(model, tokenizer, model_name: str) -> Dict[str, float]:
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print('='*80)

    similar_scores = []
    different_scores = []

    print("\n[유사한 문장 쌍]")
    for sent1, sent2 in tqdm(SIMILAR_PAIRS, desc="Similar pairs"):
        emb1 = get_embeddings([sent1], model, tokenizer)
        emb2 = get_embeddings([sent2], model, tokenizer)
        score = compute_similarity(emb1, emb2)
        similar_scores.append(score)
        print(f"  Score: {score:.4f} | {sent1[:30]}... <-> {sent2[:30]}...")

    print("\n[다른 문장 쌍]")
    for sent1, sent2 in tqdm(DIFFERENT_PAIRS, desc="Different pairs"):
        emb1 = get_embeddings([sent1], model, tokenizer)
        emb2 = get_embeddings([sent2], model, tokenizer)
        score = compute_similarity(emb1, emb2)
        different_scores.append(score)
        print(f"  Score: {score:.4f} | {sent1[:30]}... <-> {sent2[:30]}...")

    avg_similar = np.mean(similar_scores)
    avg_different = np.mean(different_scores)
    separation = avg_similar - avg_different

    print(f"\n{'='*80}")
    print(f"Results for {model_name}:")
    print(f"  유사 문장 평균 점수: {avg_similar:.4f} (높을수록 좋음)")
    print(f"  다른 문장 평균 점수: {avg_different:.4f} (낮을수록 좋음)")
    print(f"  구분도 (Separation): {separation:.4f} (높을수록 좋음)")
    print('='*80)

    return {
        'avg_similar': avg_similar,
        'avg_different': avg_different,
        'separation': separation,
        'similar_scores': similar_scores,
        'different_scores': different_scores
    }


def main():
    print("="*80)
    print("한국어 임베딩 모델 성능 비교 평가 (CORRECT)")
    print("="*80)

    # Model configurations with base_model_path for adapter stages
    models = [
        {
            "name": "Stage 0 (Vocab Expanded)",
            "path": str(PROJECT_ROOT / "outputs/koqwen-expanded"),
            "load_adapters": False,
        },
        {
            "name": "Stage 1 v2 (SimCSE 10K)",
            "path": str(PROJECT_ROOT / "checkpoints/stage1/final"),
            "load_adapters": False,
        },
        {
            "name": "Stage 2 (EEVE Adapter)",
            "path": str(PROJECT_ROOT / "checkpoints/stage2/final"),
            "base_model_path": str(PROJECT_ROOT / "checkpoints/stage1/final"),  # Load Stage 1 base!
            "load_adapters": True,
            "adapter_type": "bottleneck",
        },
        {
            "name": "Stage 3 (Hierarchical)",
            "path": str(PROJECT_ROOT / "checkpoints/stage3/final"),
            "base_model_path": str(PROJECT_ROOT / "checkpoints/stage1/final"),  # Load Stage 1 base! (Stage 3 replaces Stage 2 adapters)
            "load_adapters": True,
            "adapter_type": "hierarchical",
        },
    ]

    results = {}

    for config in models:
        if not Path(config["path"]).exists():
            print(f"\n⚠️  모델을 찾을 수 없습니다: {config['path']}")
            continue

        try:
            model, tokenizer = load_model_and_tokenizer(
                config["path"],
                base_model_path=config.get("base_model_path"),
                load_adapters=config.get("load_adapters", False),
                adapter_type=config.get("adapter_type", "bottleneck")
            )

            result = evaluate_model(model, tokenizer, config["name"])
            results[config["name"]] = result

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ 오류 발생: {config['name']}")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Final comparison
    print("\n" + "="*80)
    print("최종 비교 결과")
    print("="*80)
    print(f"{'Model':<30} {'Similar↑':<12} {'Different↓':<12} {'Separation↑':<12}")
    print("-"*80)

    for model_name, result in results.items():
        print(f"{model_name:<30} "
              f"{result['avg_similar']:<12.4f} "
              f"{result['avg_different']:<12.4f} "
              f"{result['separation']:<12.4f}")

    print("="*80)

    if results:
        best_model = max(results.items(), key=lambda x: x[1]['separation'])
        print(f"\n🏆 최고 성능 모델: {best_model[0]}")
        print(f"   구분도: {best_model[1]['separation']:.4f}")
        print("="*80)


if __name__ == "__main__":
    main()
