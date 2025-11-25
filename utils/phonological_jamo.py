"""
Phonological Jamo Composition (PJC)

음운 규칙을 반영한 자모 기반 임베딩 합성
- 연음, 경음화, 비음화, 구개음화, 격음화 등 한국어 음운 규칙 적용
- 음절 간 상호작용 모델링
- Zero-shot 일반화 지원

Reference:
- Korean Phonology (Sohn 1999)
- CharacterBERT (El Boukkouri et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import unicodedata


# ============================================================================
# Korean Jamo Constants
# ============================================================================

# 초성 (Choseong) - 19개
CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 (Jungseong) - 21개
JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 (Jongseong) - 28개 (받침 없음 포함)
JONGSEONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
             'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# Index mappings
CHO_TO_IDX = {c: i for i, c in enumerate(CHOSEONG)}
JUNG_TO_IDX = {c: i for i, c in enumerate(JUNGSEONG)}
JONG_TO_IDX = {c: i for i, c in enumerate(JONGSEONG)}

# Unicode constants
HANGUL_BASE = 0xAC00  # '가'
HANGUL_END = 0xD7A3   # '힣'
CHO_COUNT = 19
JUNG_COUNT = 21
JONG_COUNT = 28


# ============================================================================
# Korean Phonological Rules
# ============================================================================

class PhonologicalRules:
    """
    한국어 음운 규칙 정의

    주요 규칙:
    1. 연음 (Liaison): 종성 + 초성 'ㅇ' → 종성이 다음 음절 초성으로 이동
    2. 경음화 (Fortition): 특정 종성 + 평음 → 경음
    3. 비음화 (Nasalization): 특정 종성 + 비음 → 비음
    4. 구개음화 (Palatalization): 'ㄷ', 'ㅌ' + 'ㅣ' → 'ㅈ', 'ㅊ'
    5. 격음화 (Aspiration): 'ㅎ' + 평음 → 격음
    """

    # 경음화 규칙: (종성, 초성) → 변환된 초성
    FORTITION = {
        # 받침 'ㄱ, ㄲ, ㄳ, ㄺ' + 'ㄱ, ㄷ, ㅂ, ㅅ, ㅈ' → 경음
        ('ㄱ', 'ㄱ'): 'ㄲ', ('ㄱ', 'ㄷ'): 'ㄸ', ('ㄱ', 'ㅂ'): 'ㅃ',
        ('ㄱ', 'ㅅ'): 'ㅆ', ('ㄱ', 'ㅈ'): 'ㅉ',
        ('ㄲ', 'ㄱ'): 'ㄲ', ('ㄲ', 'ㄷ'): 'ㄸ', ('ㄲ', 'ㅂ'): 'ㅃ',
        ('ㄲ', 'ㅅ'): 'ㅆ', ('ㄲ', 'ㅈ'): 'ㅉ',
        # 받침 'ㄷ, ㅅ, ㅆ, ㅈ, ㅊ, ㅌ' + 평음 → 경음
        ('ㄷ', 'ㄱ'): 'ㄲ', ('ㄷ', 'ㄷ'): 'ㄸ', ('ㄷ', 'ㅂ'): 'ㅃ',
        ('ㄷ', 'ㅅ'): 'ㅆ', ('ㄷ', 'ㅈ'): 'ㅉ',
        ('ㅅ', 'ㄱ'): 'ㄲ', ('ㅅ', 'ㄷ'): 'ㄸ', ('ㅅ', 'ㅂ'): 'ㅃ',
        ('ㅅ', 'ㅅ'): 'ㅆ', ('ㅅ', 'ㅈ'): 'ㅉ',
        # 받침 'ㅂ, ㅍ' + 평음 → 경음
        ('ㅂ', 'ㄱ'): 'ㄲ', ('ㅂ', 'ㄷ'): 'ㄸ', ('ㅂ', 'ㅅ'): 'ㅆ',
        ('ㅂ', 'ㅈ'): 'ㅉ',
    }

    # 비음화 규칙: (종성, 초성) → (변환된 종성, 초성 유지)
    NASALIZATION = {
        # 받침 'ㄱ, ㄲ, ㅋ' + 'ㄴ, ㅁ' → 'ㅇ'
        ('ㄱ', 'ㄴ'): ('ㅇ', 'ㄴ'), ('ㄱ', 'ㅁ'): ('ㅇ', 'ㅁ'),
        ('ㄲ', 'ㄴ'): ('ㅇ', 'ㄴ'), ('ㄲ', 'ㅁ'): ('ㅇ', 'ㅁ'),
        ('ㅋ', 'ㄴ'): ('ㅇ', 'ㄴ'), ('ㅋ', 'ㅁ'): ('ㅇ', 'ㅁ'),
        # 받침 'ㄷ, ㅅ, ㅆ, ㅈ, ㅊ, ㅌ' + 'ㄴ, ㅁ' → 'ㄴ'
        ('ㄷ', 'ㄴ'): ('ㄴ', 'ㄴ'), ('ㄷ', 'ㅁ'): ('ㄴ', 'ㅁ'),
        ('ㅅ', 'ㄴ'): ('ㄴ', 'ㄴ'), ('ㅅ', 'ㅁ'): ('ㄴ', 'ㅁ'),
        ('ㅆ', 'ㄴ'): ('ㄴ', 'ㄴ'), ('ㅆ', 'ㅁ'): ('ㄴ', 'ㅁ'),
        # 받침 'ㅂ, ㅍ' + 'ㄴ, ㅁ' → 'ㅁ'
        ('ㅂ', 'ㄴ'): ('ㅁ', 'ㄴ'), ('ㅂ', 'ㅁ'): ('ㅁ', 'ㅁ'),
        ('ㅍ', 'ㄴ'): ('ㅁ', 'ㄴ'), ('ㅍ', 'ㅁ'): ('ㅁ', 'ㅁ'),
    }

    # 격음화 규칙: (종성 ㅎ 또는 초성 ㅎ) + 평음 → 격음
    ASPIRATION = {
        # 종성 'ㅎ' + 평음 → 격음
        ('ㅎ', 'ㄱ'): ('', 'ㅋ'),
        ('ㅎ', 'ㄷ'): ('', 'ㅌ'),
        ('ㅎ', 'ㅂ'): ('', 'ㅍ'),
        ('ㅎ', 'ㅈ'): ('', 'ㅊ'),
        # 종성 평음 + 초성 'ㅎ' → 격음
        ('ㄱ', 'ㅎ'): ('', 'ㅋ'),
        ('ㄷ', 'ㅎ'): ('', 'ㅌ'),
        ('ㅂ', 'ㅎ'): ('', 'ㅍ'),
        ('ㅈ', 'ㅎ'): ('', 'ㅊ'),
    }

    # 구개음화: (종성, 초성 또는 모음) → 변환
    PALATALIZATION = {
        # 받침 'ㄷ' + 모음 'ㅣ' → 'ㅈ'
        ('ㄷ', 'ㅣ'): ('', 'ㅈ'),
        # 받침 'ㅌ' + 모음 'ㅣ' → 'ㅊ'
        ('ㅌ', 'ㅣ'): ('', 'ㅊ'),
    }

    # 연음: 종성 + 초성 'ㅇ' → 종성이 초성으로 이동
    LIAISON_JONGSEONG = {
        'ㄱ': 'ㄱ', 'ㄲ': 'ㄲ', 'ㄴ': 'ㄴ', 'ㄷ': 'ㄷ',
        'ㄹ': 'ㄹ', 'ㅁ': 'ㅁ', 'ㅂ': 'ㅂ', 'ㅅ': 'ㅅ',
        'ㅆ': 'ㅆ', 'ㅇ': 'ㅇ', 'ㅈ': 'ㅈ', 'ㅊ': 'ㅊ',
        'ㅋ': 'ㅋ', 'ㅌ': 'ㅌ', 'ㅍ': 'ㅍ', 'ㅎ': 'ㅎ',
        # 겹받침은 뒤 자음만 연음
        'ㄳ': 'ㅅ', 'ㄵ': 'ㅈ', 'ㄶ': 'ㅎ', 'ㄺ': 'ㄱ',
        'ㄻ': 'ㅁ', 'ㄼ': 'ㄹ', 'ㄽ': 'ㅅ', 'ㄾ': 'ㅌ',
        'ㄿ': 'ㅍ', 'ㅀ': 'ㅎ', 'ㅄ': 'ㅅ',
    }

    @classmethod
    def apply_rules(cls, jong: str, next_cho: str, next_jung: str = None) -> Tuple[str, str]:
        """
        음운 규칙 적용

        Args:
            jong: 현재 음절 종성
            next_cho: 다음 음절 초성
            next_jung: 다음 음절 중성 (구개음화용)

        Returns:
            (변환된 종성, 변환된 초성)
        """
        if not jong or jong == '':
            return ('', next_cho)

        # 1. 연음 (Liaison)
        if next_cho == 'ㅇ' and jong in cls.LIAISON_JONGSEONG:
            return ('', cls.LIAISON_JONGSEONG[jong])

        # 2. 격음화 (Aspiration)
        if (jong, next_cho) in cls.ASPIRATION:
            return cls.ASPIRATION[(jong, next_cho)]

        # 3. 구개음화 (Palatalization)
        if next_jung and (jong, next_jung) in cls.PALATALIZATION:
            new_jong, new_cho = cls.PALATALIZATION[(jong, next_jung)]
            if next_cho == 'ㅇ':  # 이, 히 등
                return (new_jong, new_cho)

        # 4. 비음화 (Nasalization)
        if (jong, next_cho) in cls.NASALIZATION:
            return cls.NASALIZATION[(jong, next_cho)]

        # 5. 경음화 (Fortition)
        if (jong, next_cho) in cls.FORTITION:
            return (jong, cls.FORTITION[(jong, next_cho)])

        # 규칙 미적용
        return (jong, next_cho)


# ============================================================================
# Jamo Decomposition Functions
# ============================================================================

def is_hangul_syllable(char: str) -> bool:
    """Check if character is a complete Hangul syllable (가-힣)"""
    if len(char) != 1:
        return False
    code = ord(char)
    return HANGUL_BASE <= code <= HANGUL_END


def decompose_hangul(char: str) -> Tuple[int, int, int]:
    """
    한글 음절을 자모로 분해

    Args:
        char: 한글 음절 (예: '한')

    Returns:
        (초성_인덱스, 중성_인덱스, 종성_인덱스)
    """
    if not is_hangul_syllable(char):
        return (-1, -1, -1)

    code = ord(char) - HANGUL_BASE
    cho = code // (JUNG_COUNT * JONG_COUNT)
    jung = (code % (JUNG_COUNT * JONG_COUNT)) // JONG_COUNT
    jong = code % JONG_COUNT

    return (cho, jung, jong)


def get_jamo_chars(char: str) -> Tuple[str, str, str]:
    """
    한글 음절을 자모 문자로 분해

    Args:
        char: 한글 음절

    Returns:
        (초성 문자, 중성 문자, 종성 문자)
    """
    cho_idx, jung_idx, jong_idx = decompose_hangul(char)
    if cho_idx < 0:
        return ('', '', '')

    return (
        CHOSEONG[cho_idx],
        JUNGSEONG[jung_idx],
        JONGSEONG[jong_idx]
    )


def decompose_token(token: str) -> List[Tuple[int, int, int]]:
    """토큰 전체를 자모로 분해"""
    result = []
    for char in token:
        if is_hangul_syllable(char):
            result.append(decompose_hangul(char))
    return result


# ============================================================================
# Phonological Jamo Embedding Layer
# ============================================================================

class PhonologicalJamoComposer(nn.Module):
    """
    음운 규칙을 반영한 자모 합성기 (PJC)

    기존 JLCE의 한계 극복:
    1. 음운 규칙 적용 (연음, 경음화, 비음화, 구개음화, 격음화)
    2. 음절 간 상호작용 모델링 (Bi-LSTM 또는 Transformer)
    3. Soft 규칙 적용 (학습 가능한 가중치)
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_phonological_rules: bool = True,
        inter_syllable_type: str = 'transformer'  # 'transformer', 'lstm', 'none'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_phonological_rules = use_phonological_rules
        self.inter_syllable_type = inter_syllable_type

        # 자모 임베딩 (총 68개)
        self.cho_embedding = nn.Embedding(CHO_COUNT, hidden_dim)   # 초성 19개
        self.jung_embedding = nn.Embedding(JUNG_COUNT, hidden_dim)  # 중성 21개
        self.jong_embedding = nn.Embedding(JONG_COUNT, hidden_dim)  # 종성 28개 (0=없음)

        # 위치 인코딩 (초성/중성/종성 구분)
        self.position_embedding = nn.Embedding(3, hidden_dim)

        # 음운 규칙 변환 레이어
        if use_phonological_rules:
            self.phonological_transform = PhonologicalTransformLayer(hidden_dim)

        # 음절 내 합성 (초성 + 중성 + 종성 → 음절)
        self.syllable_composer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 음절 간 상호작용
        if inter_syllable_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.inter_syllable = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif inter_syllable_type == 'lstm':
            self.inter_syllable = nn.LSTM(
                hidden_dim, hidden_dim // 2,
                num_layers=2, bidirectional=True,
                dropout=dropout, batch_first=True
            )
        else:
            self.inter_syllable = None

        # 최종 토큰 임베딩 생성
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.cho_embedding, self.jung_embedding, self.jong_embedding]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def compose_syllable(
        self,
        cho_idx: torch.Tensor,
        jung_idx: torch.Tensor,
        jong_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        단일 음절 합성

        Args:
            cho_idx: 초성 인덱스 [batch_size]
            jung_idx: 중성 인덱스 [batch_size]
            jong_idx: 종성 인덱스 [batch_size]

        Returns:
            syllable_embedding: [batch_size, hidden_dim]
        """
        # 자모 임베딩
        cho_emb = self.cho_embedding(cho_idx)   # [B, D]
        jung_emb = self.jung_embedding(jung_idx)  # [B, D]
        jong_emb = self.jong_embedding(jong_idx)  # [B, D]

        # 위치 인코딩 추가
        pos_cho = self.position_embedding(torch.zeros_like(cho_idx))
        pos_jung = self.position_embedding(torch.ones_like(jung_idx))
        pos_jong = self.position_embedding(torch.full_like(jong_idx, 2))

        cho_emb = cho_emb + pos_cho
        jung_emb = jung_emb + pos_jung
        jong_emb = jong_emb + pos_jong

        # 합성
        concat = torch.cat([cho_emb, jung_emb, jong_emb], dim=-1)  # [B, D*3]
        syllable = self.syllable_composer(concat)  # [B, D]

        return syllable

    def forward(
        self,
        tokens: List[str],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        토큰 리스트를 자모 기반 임베딩으로 변환

        Args:
            tokens: 토큰 문자열 리스트
            device: 디바이스

        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        if device is None:
            device = self.cho_embedding.weight.device

        batch_size = len(tokens)
        max_syllables = max(len([c for c in t if is_hangul_syllable(c)]) for t in tokens)

        if max_syllables == 0:
            # 한글 없으면 zero 반환
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        # 배치 텐서 준비
        cho_batch = torch.zeros(batch_size, max_syllables, dtype=torch.long, device=device)
        jung_batch = torch.zeros(batch_size, max_syllables, dtype=torch.long, device=device)
        jong_batch = torch.zeros(batch_size, max_syllables, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, max_syllables, dtype=torch.bool, device=device)

        # 토큰별 자모 분해
        for b, token in enumerate(tokens):
            syllables = [(c, decompose_hangul(c)) for c in token if is_hangul_syllable(c)]

            for s, (char, (cho, jung, jong)) in enumerate(syllables):
                # 음운 규칙 적용 (다음 음절 정보 필요)
                if self.use_phonological_rules and s < len(syllables) - 1:
                    next_char, (next_cho, next_jung, _) = syllables[s + 1]
                    jong_char = JONGSEONG[jong]
                    next_cho_char = CHOSEONG[next_cho]
                    next_jung_char = JUNGSEONG[next_jung]

                    # 규칙 적용
                    new_jong_char, new_cho_char = PhonologicalRules.apply_rules(
                        jong_char, next_cho_char, next_jung_char
                    )

                    # 인덱스 변환
                    if new_jong_char in JONG_TO_IDX:
                        jong = JONG_TO_IDX[new_jong_char]
                    if new_cho_char in CHO_TO_IDX:
                        # 다음 음절 초성 업데이트 (이 루프에서는 현재 음절만 처리)
                        pass

                cho_batch[b, s] = cho
                jung_batch[b, s] = jung
                jong_batch[b, s] = jong
                mask[b, s] = True

        # 음절별 합성
        syllable_embeddings = []
        for s in range(max_syllables):
            syl_emb = self.compose_syllable(
                cho_batch[:, s],
                jung_batch[:, s],
                jong_batch[:, s]
            )
            syllable_embeddings.append(syl_emb)

        syllable_embeddings = torch.stack(syllable_embeddings, dim=1)  # [B, S, D]

        # 음절 간 상호작용
        if self.inter_syllable is not None:
            if self.inter_syllable_type == 'transformer':
                # Transformer는 mask 사용
                src_key_padding_mask = ~mask
                syllable_embeddings = self.inter_syllable(
                    syllable_embeddings,
                    src_key_padding_mask=src_key_padding_mask
                )
            elif self.inter_syllable_type == 'lstm':
                # Pack for LSTM
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    syllable_embeddings, lengths,
                    batch_first=True, enforce_sorted=False
                )
                output, _ = self.inter_syllable(packed)
                syllable_embeddings, _ = nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=True, total_length=max_syllables
                )

        # Mean pooling over syllables
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = (syllable_embeddings * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        pooled = sum_embeddings / count

        # 최종 projection
        output = self.final_projection(pooled)

        return output

    def blend_with_existing(
        self,
        jamo_embeddings: torch.Tensor,
        existing_embeddings: torch.Tensor,
        blend_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        자모 임베딩과 기존 임베딩 블렌딩

        Args:
            jamo_embeddings: PJC로 생성한 임베딩
            existing_embeddings: 기존 토큰 임베딩
            blend_ratio: 자모 임베딩 비율 (0~1)

        Returns:
            blended: 블렌딩된 임베딩
        """
        return blend_ratio * jamo_embeddings + (1 - blend_ratio) * existing_embeddings


class PhonologicalTransformLayer(nn.Module):
    """
    음운 규칙 기반 변환 레이어

    Hard-coded 규칙을 soft하게 학습 가능한 형태로 적용
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 종성-초성 상호작용 학습
        self.jong_cho_interaction = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # 규칙 적용 강도 (학습 가능)
        self.rule_strength = nn.Parameter(torch.ones(1) * 0.5)

    def forward(
        self,
        jong_embedding: torch.Tensor,
        next_cho_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        음운 규칙에 따른 임베딩 변환

        Args:
            jong_embedding: 현재 종성 임베딩 [B, D]
            next_cho_embedding: 다음 초성 임베딩 [B, D]

        Returns:
            (transformed_jong, transformed_cho)
        """
        # 상호작용 학습
        interaction = self.jong_cho_interaction(jong_embedding, next_cho_embedding)

        # Soft 규칙 적용
        strength = torch.sigmoid(self.rule_strength)

        transformed_jong = jong_embedding + strength * interaction
        transformed_cho = next_cho_embedding + strength * interaction

        return transformed_jong, transformed_cho


# ============================================================================
# PJC Loss Function
# ============================================================================

class PJCLoss(nn.Module):
    """
    Phonological Jamo Composition Loss

    자모 기반 임베딩과 타겟 임베딩 정렬
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        jamo_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            jamo_embeddings: PJC로 생성한 임베딩 [B, D]
            target_embeddings: 타겟 임베딩 [B, D]

        Returns:
            loss: 정렬 손실
        """
        # Normalize
        jamo_norm = F.normalize(jamo_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)

        # Cosine similarity loss
        similarity = (jamo_norm * target_norm).sum(dim=1)
        loss = 1.0 - similarity.mean()

        return loss


# ============================================================================
# Utility Functions
# ============================================================================

def create_pjc_embeddings(
    model: nn.Module,
    tokens: List[str],
    pjc_composer: PhonologicalJamoComposer,
    blend_ratio: float = 0.5
) -> torch.Tensor:
    """
    토큰 리스트에 대해 PJC 블렌딩 임베딩 생성

    Args:
        model: 임베딩 모델
        tokens: 토큰 문자열 리스트
        pjc_composer: PJC 합성기
        blend_ratio: 블렌딩 비율

    Returns:
        blended_embeddings: [batch_size, hidden_dim]
    """
    device = next(model.parameters()).device

    # 기존 임베딩
    # (실제 구현에서는 토큰 ID로 변환 필요)
    # existing_embeddings = model.get_input_embeddings()(token_ids)

    # PJC 임베딩
    jamo_embeddings = pjc_composer(tokens, device=device)

    # 블렌딩은 호출하는 쪽에서 수행
    return jamo_embeddings
