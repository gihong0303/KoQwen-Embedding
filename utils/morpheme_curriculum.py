"""
Morpheme-guided Curriculum (MGC)

형태소 분석기 기반 커리큘럼 학습
- MeCab 형태소 분석 통합
- 실제 형태소 구조 기반 난이도 측정
- 적응적 커리큘럼 스케줄링

Reference:
- Korean Morphological Analysis (ACL papers)
- Curriculum Learning (Bengio et al., 2009)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Morpheme Analyzer
# ============================================================================

@dataclass
class MorphemeInfo:
    """형태소 분석 결과"""
    token: str
    morphemes: List[Tuple[str, str]]  # (형태소, 품사) 쌍
    stem_count: int
    ending_count: int
    particle_count: int
    affix_count: int
    complexity_score: float
    category: str  # 'easy', 'medium', 'hard'


class MorphemeAnalyzer:
    """
    형태소 분석기 래퍼

    MeCab-ko를 사용하여 토큰을 형태소로 분석하고
    복잡도 점수 및 카테고리를 계산
    """

    # 품사 태그 분류
    STEM_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'NP',  # 명사류
                 'VV', 'VA', 'VX', 'VCP', 'VCN',    # 동사/형용사류
                 'MM', 'MAG', 'MAJ',                # 관형사/부사
                 'IC'}                              # 감탄사

    ENDING_TAGS = {'EP', 'EF', 'EC', 'ETN', 'ETM'}  # 어미류

    PARTICLE_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',  # 조사
                     'JX', 'JC'}

    AFFIX_TAGS = {'XPN', 'XSN', 'XSV', 'XSA', 'XR'}  # 접사

    def __init__(self, use_mecab: bool = True, cache_path: Optional[str] = None):
        """
        Args:
            use_mecab: MeCab 사용 여부 (False면 heuristic)
            cache_path: 분석 결과 캐시 경로
        """
        self.use_mecab = use_mecab
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, MorphemeInfo] = {}

        if use_mecab:
            try:
                from konlpy.tag import Mecab
                self.mecab = Mecab()
                logger.info("MeCab-ko initialized successfully")
            except Exception as e:
                logger.warning(f"MeCab initialization failed: {e}. Using heuristic analysis.")
                self.mecab = None
                self.use_mecab = False
        else:
            self.mecab = None

        # 캐시 로드
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        """캐시 로드"""
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            for token, info in cache_data.items():
                self.cache[token] = MorphemeInfo(**info)
            logger.info(f"Loaded {len(self.cache)} cached morpheme analyses")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """캐시 저장"""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                token: {
                    'token': info.token,
                    'morphemes': info.morphemes,
                    'stem_count': info.stem_count,
                    'ending_count': info.ending_count,
                    'particle_count': info.particle_count,
                    'affix_count': info.affix_count,
                    'complexity_score': info.complexity_score,
                    'category': info.category
                }
                for token, info in self.cache.items()
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def analyze(self, token: str) -> MorphemeInfo:
        """
        단일 토큰 분석

        Args:
            token: 분석할 토큰

        Returns:
            MorphemeInfo: 형태소 분석 결과
        """
        if token in self.cache:
            return self.cache[token]

        if self.use_mecab and self.mecab:
            info = self._analyze_with_mecab(token)
        else:
            info = self._analyze_heuristic(token)

        self.cache[token] = info
        return info

    def _analyze_with_mecab(self, token: str) -> MorphemeInfo:
        """MeCab으로 형태소 분석"""
        try:
            morphemes = self.mecab.pos(token)
        except Exception:
            morphemes = [(token, 'UNKNOWN')]

        stem_count = sum(1 for _, tag in morphemes if tag in self.STEM_TAGS)
        ending_count = sum(1 for _, tag in morphemes if tag in self.ENDING_TAGS)
        particle_count = sum(1 for _, tag in morphemes if tag in self.PARTICLE_TAGS)
        affix_count = sum(1 for _, tag in morphemes if tag in self.AFFIX_TAGS)

        # 복잡도 점수 계산
        complexity = self._compute_complexity(
            len(morphemes), stem_count, ending_count,
            particle_count, affix_count, len(token)
        )

        # 카테고리 결정
        category = self._determine_category(complexity)

        return MorphemeInfo(
            token=token,
            morphemes=morphemes,
            stem_count=stem_count,
            ending_count=ending_count,
            particle_count=particle_count,
            affix_count=affix_count,
            complexity_score=complexity,
            category=category
        )

    def _analyze_heuristic(self, token: str) -> MorphemeInfo:
        """Heuristic 기반 분석 (MeCab 없을 때)"""
        # 간단한 규칙 기반 분석
        length = len(token)

        # 조사 패턴
        particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로',
                    '와', '과', '의', '도', '만', '까지', '부터', '에게', '께']
        particle_count = sum(1 for p in particles if token.endswith(p))

        # 어미 패턴
        endings = ['다', '고', '며', '서', '니', '면', '지', '야', '요', '습니다',
                  '었', '았', '겠', '는다', 'ㄴ다']
        ending_count = sum(1 for e in endings if token.endswith(e))

        # 추정 어간 수
        stem_count = max(1, length // 3)

        complexity = self._compute_complexity(
            length // 2, stem_count, ending_count,
            particle_count, 0, length
        )

        category = self._determine_category(complexity)

        return MorphemeInfo(
            token=token,
            morphemes=[(token, 'UNKNOWN')],
            stem_count=stem_count,
            ending_count=ending_count,
            particle_count=particle_count,
            affix_count=0,
            complexity_score=complexity,
            category=category
        )

    def _compute_complexity(
        self,
        morpheme_count: int,
        stem_count: int,
        ending_count: int,
        particle_count: int,
        affix_count: int,
        token_length: int
    ) -> float:
        """
        복잡도 점수 계산 (0~1)

        가중치:
        - 형태소 개수: 30%
        - 조사/어미 개수: 30%
        - 접사 개수: 20%
        - 토큰 길이: 20%
        """
        # 형태소 개수 점수 (1개: 0, 5개 이상: 1)
        morpheme_score = min(1.0, (morpheme_count - 1) / 4)

        # 조사/어미 점수 (0개: 0, 3개 이상: 1)
        particle_ending_score = min(1.0, (particle_count + ending_count) / 3)

        # 접사 점수 (0개: 0, 2개 이상: 1)
        affix_score = min(1.0, affix_count / 2)

        # 길이 점수 (2자: 0, 10자 이상: 1)
        length_score = min(1.0, max(0, (token_length - 2) / 8))

        complexity = (
            0.30 * morpheme_score +
            0.30 * particle_ending_score +
            0.20 * affix_score +
            0.20 * length_score
        )

        return complexity

    def _determine_category(self, complexity: float) -> str:
        """복잡도에 따른 카테고리 결정"""
        if complexity < 0.33:
            return 'easy'
        elif complexity < 0.66:
            return 'medium'
        else:
            return 'hard'

    def analyze_batch(self, tokens: List[str]) -> List[MorphemeInfo]:
        """배치 분석"""
        return [self.analyze(token) for token in tokens]

    def save_analysis(self, output_path: str):
        """분석 결과 저장"""
        self._save_cache()
        logger.info(f"Saved {len(self.cache)} morpheme analyses to {self.cache_path}")


# ============================================================================
# Adaptive Curriculum Scheduler
# ============================================================================

class AdaptiveCurriculumScheduler:
    """
    적응적 커리큘럼 스케줄러

    각 카테고리의 손실을 모니터링하여
    학습이 필요한 카테고리에 더 많은 가중치 부여
    """

    def __init__(
        self,
        categories: List[str] = None,
        initial_weights: Dict[str, float] = None,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 0.6
    ):
        """
        Args:
            categories: 카테고리 목록
            initial_weights: 초기 가중치
            adaptation_rate: 적응 속도
            min_weight: 최소 가중치
            max_weight: 최대 가중치
        """
        self.categories = categories or ['easy', 'medium', 'hard']
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # 기본: easy에 더 높은 가중치로 시작
            self.weights = {'easy': 0.5, 'medium': 0.3, 'hard': 0.2}

        # 손실 히스토리
        self.loss_history: Dict[str, List[float]] = {cat: [] for cat in self.categories}

        # EMA 손실
        self.ema_losses: Dict[str, float] = {cat: 1.0 for cat in self.categories}
        self.ema_alpha = 0.9

    def update(self, losses_by_category: Dict[str, float]) -> Dict[str, float]:
        """
        손실 기반 가중치 업데이트

        Args:
            losses_by_category: {'easy': 0.5, 'medium': 0.8, 'hard': 1.2}

        Returns:
            updated_weights: 업데이트된 가중치
        """
        # EMA 업데이트
        for cat, loss in losses_by_category.items():
            if cat in self.ema_losses:
                self.ema_losses[cat] = (
                    self.ema_alpha * self.ema_losses[cat] +
                    (1 - self.ema_alpha) * loss
                )
                self.loss_history[cat].append(loss)

        # 손실 비율 기반 가중치 조정
        total_loss = sum(self.ema_losses.values())
        if total_loss > 0:
            for cat in self.categories:
                # 손실이 높은 카테고리에 더 많은 가중치
                loss_ratio = self.ema_losses[cat] / total_loss
                target_weight = loss_ratio

                # 점진적 업데이트
                self.weights[cat] = (
                    (1 - self.adaptation_rate) * self.weights[cat] +
                    self.adaptation_rate * target_weight
                )

                # Clamp
                self.weights[cat] = max(self.min_weight, min(self.max_weight, self.weights[cat]))

        # 정규화
        total_weight = sum(self.weights.values())
        for cat in self.categories:
            self.weights[cat] /= total_weight

        return self.weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """현재 가중치 반환"""
        return self.weights.copy()

    def get_stage_weights(self, stage: int) -> Dict[str, float]:
        """
        스테이지별 기본 가중치 반환

        Stage 1: Easy 강조
        Stage 2: Medium 강조
        Stage 3: Hard 강조
        """
        if stage == 1:
            return {'easy': 0.6, 'medium': 0.3, 'hard': 0.1}
        elif stage == 2:
            return {'easy': 0.2, 'medium': 0.6, 'hard': 0.2}
        elif stage == 3:
            return {'easy': 0.1, 'medium': 0.3, 'hard': 0.6}
        else:
            return {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}


# ============================================================================
# Curriculum Dataset Wrapper
# ============================================================================

class MorphemeCurriculumDataset(Dataset):
    """
    형태소 기반 커리큘럼 데이터셋

    토큰을 난이도별로 분류하고 가중치 샘플링 지원
    """

    def __init__(
        self,
        base_dataset: Dataset,
        analyzer: MorphemeAnalyzer,
        tokenizer,
        stage: int = 1,
        curriculum_weights: Dict[str, float] = None
    ):
        """
        Args:
            base_dataset: 기본 데이터셋
            analyzer: 형태소 분석기
            tokenizer: 토크나이저
            stage: 현재 스테이지 (1, 2, 3)
            curriculum_weights: 카테고리별 샘플링 가중치
        """
        self.base_dataset = base_dataset
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        self.stage = stage

        # 스테이지별 기본 가중치
        scheduler = AdaptiveCurriculumScheduler()
        self.weights = curriculum_weights or scheduler.get_stage_weights(stage)

        # 데이터셋 분석 및 카테고리 할당
        self.sample_categories: List[str] = []
        self.sample_weights: List[float] = []

        self._analyze_dataset()

    def _analyze_dataset(self):
        """데이터셋 분석하여 카테고리 할당"""
        logger.info(f"Analyzing dataset for Stage {self.stage} curriculum...")

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            text = sample.get('text', '')

            # 텍스트를 토큰으로 분해
            tokens = self.tokenizer.tokenize(text)

            # 토큰별 복잡도 평균
            if tokens:
                infos = self.analyzer.analyze_batch(tokens[:10])  # 처음 10개만
                avg_complexity = np.mean([info.complexity_score for info in infos])
            else:
                avg_complexity = 0.5

            # 카테고리 결정
            if avg_complexity < 0.33:
                category = 'easy'
            elif avg_complexity < 0.66:
                category = 'medium'
            else:
                category = 'hard'

            self.sample_categories.append(category)
            self.sample_weights.append(self.weights.get(category, 0.33))

        # 가중치 정규화
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight for w in self.sample_weights]

        # 통계 로깅
        category_counts = Counter(self.sample_categories)
        logger.info(f"Category distribution: {dict(category_counts)}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['category'] = self.sample_categories[idx]
        item['curriculum_weight'] = self.sample_weights[idx]
        return item

    def get_sampler(self) -> Sampler:
        """가중치 기반 샘플러 반환"""
        return torch.utils.data.WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )


# ============================================================================
# Token Difficulty Analyzer
# ============================================================================

class TokenDifficultyAnalyzer:
    """
    토큰 난이도 분석기

    형태소 분석 + 빈도 + 서브워드 복잡도를 종합하여 난이도 계산
    """

    def __init__(
        self,
        analyzer: MorphemeAnalyzer,
        tokenizer,
        frequency_data: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            analyzer: 형태소 분석기
            tokenizer: 토크나이저
            frequency_data: 토큰별 빈도 데이터
        """
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        self.frequency_data = frequency_data or {}

        # 난이도 가중치
        self.morpheme_weight = 0.40
        self.frequency_weight = 0.35
        self.subword_weight = 0.25

    def compute_difficulty(self, token: str) -> Tuple[float, str]:
        """
        토큰 난이도 계산

        Args:
            token: 토큰 문자열

        Returns:
            (difficulty_score, category)
        """
        # 1. 형태소 복잡도 (0~1)
        morpheme_info = self.analyzer.analyze(token)
        morpheme_complexity = morpheme_info.complexity_score

        # 2. 빈도 역수 (0~1, 빈도 낮을수록 어려움)
        frequency = self.frequency_data.get(token, 1)
        max_freq = max(self.frequency_data.values()) if self.frequency_data else 1
        frequency_score = 1.0 - (np.log1p(frequency) / np.log1p(max_freq))

        # 3. 서브워드 복잡도 (분해 시 서브워드 개수)
        subwords = self.tokenizer.tokenize(token)
        subword_count = len(subwords)
        subword_complexity = min(1.0, (subword_count - 1) / 5)

        # 종합 난이도
        difficulty = (
            self.morpheme_weight * morpheme_complexity +
            self.frequency_weight * frequency_score +
            self.subword_weight * subword_complexity
        )

        # 카테고리
        if difficulty < 0.33:
            category = 'easy'
        elif difficulty < 0.66:
            category = 'medium'
        else:
            category = 'hard'

        return difficulty, category

    def categorize_vocabulary(
        self,
        vocab: Set[str],
        output_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        전체 어휘를 카테고리별로 분류

        Args:
            vocab: 어휘 집합
            output_path: 결과 저장 경로

        Returns:
            {
                'easy': [...],
                'medium': [...],
                'hard': [...]
            }
        """
        categories = {'easy': [], 'medium': [], 'hard': []}
        difficulty_scores = {}

        for token in vocab:
            score, category = self.compute_difficulty(token)
            categories[category].append(token)
            difficulty_scores[token] = {
                'score': score,
                'category': category
            }

        # 저장
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'categories': categories,
                    'scores': difficulty_scores,
                    'stats': {
                        'easy': len(categories['easy']),
                        'medium': len(categories['medium']),
                        'hard': len(categories['hard'])
                    }
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved token categories to {output_path}")

        return categories


# ============================================================================
# MGC Loss Weighting
# ============================================================================

class MGCLossWeighting:
    """
    형태소 커리큘럼 기반 손실 가중치

    어려운 토큰에 대해 더 높은 손실 가중치 부여
    """

    def __init__(
        self,
        stage: int = 1,
        easy_weight: float = 1.0,
        medium_weight: float = 1.5,
        hard_weight: float = 2.0
    ):
        """
        Args:
            stage: 현재 스테이지
            easy_weight: Easy 토큰 가중치
            medium_weight: Medium 토큰 가중치
            hard_weight: Hard 토큰 가중치
        """
        self.stage = stage

        # 스테이지별 가중치 조정
        if stage == 1:
            # Easy 강조
            self.weights = {'easy': easy_weight * 1.5, 'medium': medium_weight * 0.8, 'hard': hard_weight * 0.5}
        elif stage == 2:
            # Medium 강조
            self.weights = {'easy': easy_weight * 0.8, 'medium': medium_weight * 1.5, 'hard': hard_weight * 0.8}
        elif stage == 3:
            # Hard 강조
            self.weights = {'easy': easy_weight * 0.5, 'medium': medium_weight * 0.8, 'hard': hard_weight * 1.5}
        else:
            self.weights = {'easy': easy_weight, 'medium': medium_weight, 'hard': hard_weight}

    def get_weight(self, category: str) -> float:
        """카테고리에 대한 가중치 반환"""
        return self.weights.get(category, 1.0)

    def apply_weights(
        self,
        loss: torch.Tensor,
        categories: List[str]
    ) -> torch.Tensor:
        """
        손실에 카테고리별 가중치 적용

        Args:
            loss: [batch_size] 손실 텐서
            categories: 각 샘플의 카테고리

        Returns:
            weighted_loss: 가중치 적용된 손실
        """
        weights = torch.tensor(
            [self.get_weight(cat) for cat in categories],
            device=loss.device
        )
        return (loss * weights).mean()
