"""
Morphological Curriculum Learning (MCL)

형태소 복잡도 기반 커리큘럼 학습
- 어근 → 어근+어미 → 복합 형태 순서로 학습
- 한국어 교착어 특성 활용
- 언어학적으로 자연스러운 학습 순서

Reference:
- Morphological Analysis 효과 검증 (ACL 다수 논문)
- 언어 습득 이론: 단순 → 복잡 순서
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Morphological Analysis
# ============================================================================

class MorphologicalAnalyzer:
    """
    한국어 형태소 분석기 wrapper

    MeCab-ko 또는 규칙 기반 분석 지원
    """

    def __init__(self, use_mecab: bool = True):
        self.use_mecab = use_mecab
        self.mecab = None

        if use_mecab:
            try:
                import MeCab
                # MeCab-ko 사전 사용
                self.mecab = MeCab.Tagger()
                logger.info("MeCab-ko initialized successfully")
            except ImportError:
                logger.warning("MeCab not available, using rule-based analysis")
                self.use_mecab = False

    def analyze(self, text: str) -> List[Tuple[str, str]]:
        """
        형태소 분석

        Args:
            text: 분석할 텍스트

        Returns:
            [(형태소, 품사), ...] 리스트
        """
        if self.use_mecab and self.mecab:
            return self._mecab_analyze(text)
        else:
            return self._rule_based_analyze(text)

    def _mecab_analyze(self, text: str) -> List[Tuple[str, str]]:
        """MeCab 기반 분석"""
        result = []
        parsed = self.mecab.parse(text)

        if parsed is None:
            return []

        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                continue
            try:
                surface, features = line.split('\t')
                pos = features.split(',')[0]
                result.append((surface, pos))
            except ValueError:
                continue

        return result

    def _rule_based_analyze(self, text: str) -> List[Tuple[str, str]]:
        """
        규칙 기반 형태소 분석 (MeCab 없을 때 fallback)

        간단한 휴리스틱 기반 분석
        """
        result = []

        # 한글 단어 추출
        words = re.findall(r'[가-힣]+', text)

        for word in words:
            # 간단한 어미 분리
            morphemes = self._split_word(word)
            result.extend(morphemes)

        return result

    def _split_word(self, word: str) -> List[Tuple[str, str]]:
        """단어를 어근+어미로 간단히 분리"""
        # 일반적인 한국어 어미 패턴
        endings = [
            ('습니다', 'EF'),  # 격식체 종결어미
            ('ㅂ니다', 'EF'),
            ('세요', 'EF'),
            ('았다', 'EP+EF'),
            ('었다', 'EP+EF'),
            ('겠다', 'EP+EF'),
            ('는다', 'EF'),
            ('ㄴ다', 'EF'),
            ('다', 'EF'),
            ('고', 'EC'),  # 연결어미
            ('며', 'EC'),
            ('지만', 'EC'),
            ('어서', 'EC'),
            ('아서', 'EC'),
            ('면', 'EC'),
            ('을', 'JKO'),  # 조사
            ('를', 'JKO'),
            ('이', 'JKS'),
            ('가', 'JKS'),
            ('은', 'JX'),
            ('는', 'JX'),
            ('에', 'JKB'),
            ('에서', 'JKB'),
            ('로', 'JKB'),
            ('으로', 'JKB'),
            ('와', 'JC'),
            ('과', 'JC'),
            ('의', 'JKG'),
        ]

        for ending, pos in endings:
            if word.endswith(ending) and len(word) > len(ending):
                stem = word[:-len(ending)]
                return [(stem, 'VV/NNG'), (ending, pos)]

        # 분리 불가 → 단일 형태소
        return [(word, 'NNG')]


# ============================================================================
# Morphological Complexity Scoring
# ============================================================================

class MorphologicalComplexityScorer:
    """
    형태소 복잡도 점수 계산

    Components:
    1. 형태소 개수 (30%)
    2. 어미/조사 다양성 (30%)
    3. 복합어 여부 (20%)
    4. 존댓말 레벨 (20%)
    """

    def __init__(self, analyzer: MorphologicalAnalyzer = None):
        self.analyzer = analyzer or MorphologicalAnalyzer(use_mecab=True)

        # 존댓말 패턴
        self.honorific_patterns = {
            'formal_high': ['습니다', '십니다', '세요', '시어요'],  # Level 4
            'polite': ['요', '어요', '아요'],  # Level 3
            'informal_high': ['네', '지요'],  # Level 2
            'informal': ['다', '어', '아'],  # Level 1
        }

        # 복합어 판별용 접두사/접미사
        self.compound_markers = [
            '적', '화', '성', '자', '인', '물', '장', '소', '원', '관'
        ]

    def compute_complexity(self, token: str) -> Dict[str, float]:
        """
        토큰의 형태소 복잡도 계산

        Args:
            token: 토큰 문자열

        Returns:
            complexity_scores: 각 요소별 점수 및 총점
        """
        # 형태소 분석
        morphemes = self.analyzer.analyze(token)

        # 1. 형태소 개수 점수 (0-1, 많을수록 높음)
        num_morphemes = len(morphemes)
        morpheme_score = min(num_morphemes / 5.0, 1.0)

        # 2. 어미/조사 다양성 (0-1)
        pos_tags = [pos for _, pos in morphemes]
        affix_types = set()
        for pos in pos_tags:
            if pos.startswith('E'):  # 어미
                affix_types.add('E')
            elif pos.startswith('J'):  # 조사
                affix_types.add('J')
            elif pos.startswith('XS'):  # 접미사
                affix_types.add('XS')
            elif pos.startswith('XP'):  # 접두사
                affix_types.add('XP')
        affix_diversity = len(affix_types) / 4.0

        # 3. 복합어 여부 (0 or 1)
        is_compound = 0.0
        for marker in self.compound_markers:
            if marker in token:
                is_compound = 1.0
                break
        # 또는 NNG가 여러 개
        nng_count = sum(1 for _, pos in morphemes if pos == 'NNG')
        if nng_count >= 2:
            is_compound = 1.0

        # 4. 존댓말 레벨 (0-1)
        honorific_level = 0.0
        for level, (name, patterns) in enumerate(self.honorific_patterns.items()):
            for pattern in patterns:
                if token.endswith(pattern):
                    honorific_level = (4 - level) / 4.0  # formal_high = 1.0
                    break

        # 총 복잡도 점수 (가중 평균)
        total_complexity = (
            0.30 * morpheme_score +
            0.30 * affix_diversity +
            0.20 * is_compound +
            0.20 * honorific_level
        )

        return {
            'morpheme_score': morpheme_score,
            'affix_diversity': affix_diversity,
            'is_compound': is_compound,
            'honorific_level': honorific_level,
            'total_complexity': total_complexity,
            'num_morphemes': num_morphemes
        }

    def categorize_token(self, token: str) -> str:
        """
        토큰을 난이도 카테고리로 분류

        Returns:
            'easy', 'medium', or 'hard'
        """
        scores = self.compute_complexity(token)
        complexity = scores['total_complexity']

        if complexity < 0.3:
            return 'easy'
        elif complexity < 0.6:
            return 'medium'
        else:
            return 'hard'


# ============================================================================
# Morphological Curriculum Dataset
# ============================================================================

class MorphologicalCurriculumDataset:
    """
    형태소 기반 커리큘럼 데이터셋

    Stage별 토큰 필터링:
    - Stage 1: 단일 어근 (예: 집, 학교, 먹다)
    - Stage 2: 어근+조사 (예: 집에, 학교로, 먹고)
    - Stage 3: 복합 구조 (예: 학교에서부터, 먹지않았다)
    """

    def __init__(
        self,
        tokenizer,
        analyzer: MorphologicalAnalyzer = None,
        cache_path: str = None
    ):
        self.tokenizer = tokenizer
        self.analyzer = analyzer or MorphologicalAnalyzer(use_mecab=True)
        self.scorer = MorphologicalComplexityScorer(self.analyzer)
        self.cache_path = cache_path

        # 토큰 → 카테고리 캐시
        self.token_categories: Dict[str, str] = {}
        self.token_scores: Dict[str, Dict] = {}

        if cache_path and Path(cache_path).exists():
            self._load_cache()

    def _load_cache(self):
        """캐시 로드"""
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.token_categories = data.get('categories', {})
                self.token_scores = data.get('scores', {})
                logger.info(f"Loaded {len(self.token_categories)} cached token categories")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def save_cache(self):
        """캐시 저장"""
        if self.cache_path:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'categories': self.token_categories,
                    'scores': self.token_scores
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.token_categories)} token categories to cache")

    def analyze_vocabulary(self, token_ids: List[int] = None):
        """
        어휘 분석 및 카테고리화

        Args:
            token_ids: 분석할 토큰 ID 리스트 (없으면 새 토큰만)
        """
        vocab_size = len(self.tokenizer)
        old_vocab_size = 151669  # Qwen original

        if token_ids is None:
            # 새 토큰만 분석
            token_ids = range(old_vocab_size, vocab_size)

        logger.info(f"Analyzing {len(list(token_ids))} tokens...")

        for token_id in token_ids:
            if token_id >= vocab_size:
                continue

            token = self.tokenizer.decode([token_id])
            token = token.strip()

            if not token:
                continue

            # 이미 캐시됨
            if token in self.token_categories:
                continue

            # 형태소 복잡도 계산
            scores = self.scorer.compute_complexity(token)
            category = self.scorer.categorize_token(token)

            self.token_categories[token] = category
            self.token_scores[token] = scores

        logger.info(f"Analysis complete. Categories: easy={sum(1 for c in self.token_categories.values() if c == 'easy')}, "
                   f"medium={sum(1 for c in self.token_categories.values() if c == 'medium')}, "
                   f"hard={sum(1 for c in self.token_categories.values() if c == 'hard')}")

    def get_tokens_by_stage(self, stage: int) -> Set[str]:
        """
        Stage별 토큰 집합 반환

        Args:
            stage: 1=easy, 2=medium, 3=hard

        Returns:
            토큰 문자열 집합
        """
        category_map = {1: 'easy', 2: 'medium', 3: 'hard'}
        target_category = category_map.get(stage, 'easy')

        return {
            token for token, cat in self.token_categories.items()
            if cat == target_category
        }

    def filter_samples_by_stage(
        self,
        samples: List[str],
        stage: int,
        min_coverage: float = 0.3
    ) -> List[str]:
        """
        Stage에 맞는 샘플 필터링

        Args:
            samples: 텍스트 샘플 리스트
            stage: 학습 단계 (1, 2, 3)
            min_coverage: 최소 타겟 토큰 비율

        Returns:
            필터링된 샘플 리스트
        """
        target_tokens = self.get_tokens_by_stage(stage)
        filtered = []

        for sample in samples:
            # 토큰화
            tokens = self.tokenizer.tokenize(sample)

            # 타겟 토큰 비율 계산
            target_count = sum(1 for t in tokens if t in target_tokens)
            total_count = len(tokens) if tokens else 1

            coverage = target_count / total_count

            if coverage >= min_coverage:
                filtered.append(sample)

        return filtered


# ============================================================================
# MCL Loss Function
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologicalCurriculumLoss(nn.Module):
    """
    형태소 기반 커리큘럼 손실 함수

    1. Stage-aware Contrastive Loss: 단계별 토큰에 집중
    2. Morpheme Consistency Loss: 동일 어근의 변형 간 유사성
    3. Progressive Difficulty Weight: 난이도에 따른 손실 가중치
    """

    def __init__(
        self,
        temperature: float = 0.05,
        stage_weight: float = 2.0,
        consistency_weight: float = 0.3
    ):
        super().__init__()
        self.temperature = temperature
        self.stage_weight = stage_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        token_categories: List[str],
        current_stage: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings1: 첫 번째 pass 임베딩 [batch_size, hidden_dim]
            embeddings2: 두 번째 pass 임베딩 [batch_size, hidden_dim]
            token_categories: 각 샘플의 주요 토큰 카테고리 리스트
            current_stage: 현재 학습 단계 (1, 2, 3)

        Returns:
            loss: 총 손실
            metrics: 개별 손실 값
        """
        batch_size = embeddings1.shape[0]

        # Normalize
        emb1_norm = F.normalize(embeddings1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings2, p=2, dim=1)

        # 1. Base contrastive loss (SimCSE)
        sim_matrix = torch.mm(emb1_norm, emb2_norm.t()) / self.temperature
        labels = torch.arange(batch_size, device=embeddings1.device)
        base_loss = F.cross_entropy(sim_matrix, labels)

        # 2. Stage-aware weighting
        stage_category = {1: 'easy', 2: 'medium', 3: 'hard'}
        target_category = stage_category.get(current_stage, 'easy')

        # 타겟 카테고리에 해당하는 샘플에 높은 가중치
        weights = torch.ones(batch_size, device=embeddings1.device)
        for i, cat in enumerate(token_categories):
            if cat == target_category:
                weights[i] = self.stage_weight

        weighted_loss = (weights * F.cross_entropy(sim_matrix, labels, reduction='none')).mean()

        # 3. Consistency loss (같은 카테고리 내 유사성 강화)
        consistency_loss = torch.tensor(0.0, device=embeddings1.device)

        # 같은 카테고리끼리 그룹화
        category_groups = defaultdict(list)
        for i, cat in enumerate(token_categories):
            category_groups[cat].append(i)

        for cat, indices in category_groups.items():
            if len(indices) < 2:
                continue
            # 그룹 내 임베딩 유사도 (높아야 함)
            group_embs = emb1_norm[indices]
            group_sim = torch.mm(group_embs, group_embs.t())

            # 대각선 제외 평균 유사도
            mask = ~torch.eye(len(indices), dtype=torch.bool, device=group_sim.device)
            avg_sim = group_sim[mask].mean()

            # 유사도가 낮으면 손실 증가
            consistency_loss += (1.0 - avg_sim)

        consistency_loss = consistency_loss / max(len(category_groups), 1)

        # 총 손실
        total_loss = weighted_loss + self.consistency_weight * consistency_loss

        metrics = {
            'base_loss': base_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


# ============================================================================
# Utility Functions
# ============================================================================

def create_morphological_curriculum_config(
    tokenizer,
    output_path: str = 'outputs/morphological_curriculum',
    use_mecab: bool = True
) -> str:
    """
    형태소 커리큘럼 설정 파일 생성

    Args:
        tokenizer: 토크나이저
        output_path: 출력 경로
        use_mecab: MeCab 사용 여부

    Returns:
        설정 파일 경로
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = MorphologicalAnalyzer(use_mecab=use_mecab)
    dataset = MorphologicalCurriculumDataset(
        tokenizer=tokenizer,
        analyzer=analyzer,
        cache_path=str(output_dir / 'token_categories.json')
    )

    # 새 토큰 분석
    dataset.analyze_vocabulary()
    dataset.save_cache()

    # 통계 저장
    stats = {
        'total_tokens': len(dataset.token_categories),
        'easy_count': sum(1 for c in dataset.token_categories.values() if c == 'easy'),
        'medium_count': sum(1 for c in dataset.token_categories.values() if c == 'medium'),
        'hard_count': sum(1 for c in dataset.token_categories.values() if c == 'hard'),
    }

    with open(output_dir / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Created morphological curriculum config at {output_dir}")
    logger.info(f"Stats: {stats}")

    return str(output_dir / 'token_categories.json')
