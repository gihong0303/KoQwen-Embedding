# 🇰🇷 한국어 임베딩 확장 프로젝트 (Thunder+EEVE 방식)

**Thunder 토크나이저 확장 + EEVE 어댑터 레이어를 결합한 3단계 학습 파이프라인**

## 📋 프로젝트 개요

Qwen3-Embedding-0.6B 모델에 **Thunder+EEVE 결합 방법론**을 적용하여 한국어 성능을 향상시킨 임베딩 모델을 구축합니다.

### 🎯 Thunder+EEVE 방법론 (7단계)

1. **토크나이저 확장** (Thunder): 한국어 전용 토큰 68,029개를 차집합 추가하고, 새 토큰 임베딩은 기존 서브토큰 임베딩의 평균(Wechsel)으로 초기화
2. **파라미터 확장** (EEVE): 백본 동결 + 얇은 어댑터 레이어(Bottleneck FFN/Gated Adapter) 삽입
3. **역할 분리**: 동결된 백본은 멀티링구얼 능력 유지, 어댑터가 한국어 적응 담당
4. **3단계 학습**:
   - Stage 1: 임베딩 레이어만 SimCSE 무지도 학습
   - Stage 2: EEVE 어댑터만 점진적 학습
   - Stage 3: 극소수 상위 블록 제한 해제 (옵션)
5. **안정성**: 파라미터 증분 최소화로 원본 가중치 손상 방지
6. **효율성**: BFloat16 mixed precision, 최적화된 dataloader
7. **검증**: KOREAN-WEBTEXT(무지도) → K2/KMMLU/HAE_RAE_BENCH(감독)

## 🚀 핵심 특징

### Thunder 토크나이저 확장
- ✅ KORMo-10B 토큰 68,029개 추가 (차집합 방식)
- ✅ 평균 초기화 (서브토큰 임베딩 평균)
- ✅ 기존 Qwen vocab 완전 유지 (151,669 → 219,698)

### EEVE 어댑터 시스템
- ✅ 3가지 어댑터 타입:
  - **Bottleneck**: 기본 adapter (hidden → 256 → hidden)
  - **Gated**: 동적 게이팅 적용
  - **Parallel**: 병렬 구조
- ✅ 백본 완전 동결, 어댑터만 학습
- ✅ 파라미터 증가: ~0.1B (10.6B → 10.7B)

## 📁 프로젝트 구조

```
ko-embedding-expansion/
├── configs/
│   ├── base_config.yaml          # 기본 설정
│   └── training_config.yaml      # 3단계 학습 설정
├── scripts/
│   ├── utils/
│   │   ├── model_utils.py        # 모델 로딩, 확장
│   │   ├── data_utils.py         # 데이터 로딩
│   │   ├── train_utils.py        # 학습 유틸리티
│   │   └── eeve_adapter.py       # EEVE 어댑터 모듈 ⭐NEW
│   ├── 01_analyze_tokenizers.py  # 토크나이저 분석
│   ├── 02_extract_vocab_diff.py  # Vocab 차집합 추출
│   ├── 03_expand_embeddings.py   # Thunder 임베딩 확장
│   ├── 04_train_stage1.py        # Stage 1: 임베딩 학습
│   └── 05_train_stage2.py        # Stage 2: EEVE 어댑터 학습 ⭐NEW
├── outputs/
│   └── koqwen-expanded/          # Stage 0: 확장된 모델
├── checkpoints/
│   ├── stage1/final/             # Stage 1 완료
│   └── stage2/final/             # Stage 2 완료 ⭐NEW
├── tokenizer/
│   ├── vocab_diff.json           # 68,029 토큰
│   └── vocab_diff_stats.json     # 통계
├── run_pipeline.sh               # 전체 파이프라인
├── run_stage1.sh                 # Stage 1 실행
├── run_stage2.sh                 # Stage 2 실행 ⭐NEW
└── README.md
```

## 🚀 빠른 시작

### 1. 전체 파이프라인 실행

```bash
./run_pipeline.sh
```

파이프라인은 다음 단계로 구성됩니다:

**Stage 0: 준비 (Thunder 방식)**
1. 토크나이저 분석
2. Vocab 차집합 추출 (68,029개)
3. 임베딩 확장 + 평균 초기화

**Stage 1: 임베딩 학습**
- 데이터: KOREAN-WEBTEXT (100만 샘플)
- 방법: SimCSE 무지도 학습
- 학습: 임베딩 레이어만 (백본 freeze)
- GPU: 6개, BFloat16

**Stage 2: EEVE 어댑터 학습** ⭐NEW
- 데이터: KOREAN-SyntheticText-1.5B (50만 샘플)
- 방법: 어댑터 레이어 추가 + SimCSE
- 학습: 어댑터만 (백본 + 임베딩 freeze)
- GPU: 6개, BFloat16

### 2. 단계별 실행

```bash
# Stage 0: Thunder 임베딩 확장
python scripts/01_analyze_tokenizers.py
python scripts/02_extract_vocab_diff.py
python scripts/03_expand_embeddings.py

# Stage 1: 임베딩 학습
./run_stage1.sh

# Stage 2: EEVE 어댑터 학습
./run_stage2.sh
```

## 📊 예상 결과

| 지표 | Qwen 원본 | +Thunder (S0) | +Stage1 | +Stage2 (EEVE) | 개선율 |
|------|----------|-------------|---------|---------------|--------|
| Vocab 크기 | 151,669 | 219,698 | 219,698 | 219,698 | +44.8% |
| 파라미터 | 0.6B | 0.6B | 0.6B | ~0.7B | +16% |
| 한국어 토큰 길이 | ~14.2 | ~7.8 | ~7.5 | ~7.0 | ↓ 51% |
| KoSTS 상관계수 | 0.65 | 0.70 | 0.75 | 0.78 | ↑ 20% |
| 영어 STS | 0.82 | 0.81 | 0.81 | 0.81 | ≈ 유지 |

## 🔧 설정 옵션

### EEVE 어댑터 설정

```bash
# run_stage2.sh에서 설정 가능
ADAPTER_TYPE="bottleneck"  # bottleneck, gated, parallel
ADAPTER_SIZE=256           # 어댑터 hidden size
```

#### 어댑터 타입 비교

| 타입 | 구조 | 파라미터 | 특징 |
|------|------|---------|------|
| **Bottleneck** | hidden→256→hidden | 적음 | 기본, 안정적 |
| **Gated** | + 동적 게이트 | 중간 | 적응적 학습 |
| **Parallel** | 병렬 구조 | 많음 | 표현력 높음 |

## 🎓 학습 세부사항

### Stage 0: Thunder 임베딩 확장

```
Qwen vocab (151,669) + KORMo 차집합 (68,029) = 219,698 토큰
새 토큰 초기화: 기존 서브토큰 임베딩의 평균
```

### Stage 1: 임베딩 적응 (SimCSE)

```yaml
데이터: KOREAN-WEBTEXT (1M 샘플)
방법: SimCSE (InfoNCE loss)
학습 파라미터:
  - embed_tokens (input embedding)
  - lm_head (output embedding)
백본: 완전 동결
배치: 48 per GPU (총 288)
Epoch: 2
LR: 5e-5
```

### Stage 2: EEVE 어댑터 학습 ⭐

```yaml
데이터: KOREAN-SyntheticText-1.5B (500K 샘플)
어댑터: Bottleneck (hidden→256→hidden)
학습 파라미터:
  - adapter 레이어만
백본 + 임베딩: 완전 동결
배치: 48 per GPU (총 288)
Epoch: 1
LR: 3e-5 (Stage1보다 낮음)
```

## 📚 사용 데이터셋

모든 데이터셋은 [HAERAE-HUB](https://huggingface.co/HAERAE-HUB)에서 제공:

- **Stage 1**: [KOREAN-WEBTEXT](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT)
- **Stage 2**: [KOREAN-SyntheticText-1.5B](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-SyntheticText-1.5B)
- **Evaluation**: [KoSimpleEval](https://huggingface.co/datasets/HAERAE-HUB/KoSimpleEval)

## 💻 하드웨어 요구사항

- **GPU**: 6개 (GPU 4,5,6,7,8,9 사용)
- **GPU 메모리**: 각 24GB 이상 권장
- **Mixed Precision**: BFloat16 (메모리 절약)
- **디스크**: 약 50GB (모델 + 캐시 + 체크포인트)

## 🛠️ 기술 스택

- **Framework**: PyTorch, Transformers
- **Distributed**: DDP (DistributedDataParallel)
- **Mixed Precision**: BFloat16
- **Loss**: SimCSE (InfoNCE)
- **Adapter**: Custom EEVE-style modules

## 📝 파일 위치

```
outputs/koqwen-expanded/       # Stage 0 완료
checkpoints/stage1/final/      # Stage 1 완료
checkpoints/stage2/final/      # Stage 2 완료
logs/                          # 모든 로그
```

## 🔬 EEVE 어댑터 상세

### Bottleneck Adapter

```python
hidden_states (D)
  ↓
down_proj (D → 256)
  ↓
GELU + Dropout
  ↓
up_proj (256 → D)
  ↓
gate * output + residual
```

- 파라미터: 2 × D × 256
- 각 레이어에 삽입 (24 layers × 약 0.004B = ~0.1B)

### 장점

1. **백본 보존**: 원본 Qwen 가중치 완전 유지
2. **한국어 특화**: 어댑터가 한국어 패턴만 학습
3. **효율성**: 전체 모델의 ~1% 파라미터만 추가
4. **안정성**: 작은 학습률 + 게이트 초기화로 안전하게 학습

## 🔜 다음 단계

현재 구현:
- ✅ Stage 0 (Thunder 임베딩 확장)
- ✅ Stage 1 (임베딩 학습)
- ✅ Stage 2 (EEVE 어댑터 학습)

구현 예정:
- ⏳ Stage 3 (상위 블록 일부 언락)
- ⏳ 평가 스크립트 (KoSimpleEval)
- ⏳ 모델 비교 및 분석 도구
- ⏳ 영어 성능 회귀 모니터링

## 🙏 감사의 말

- [Thunder Team](https://github.com/ibm/thunder) - Thunder 방법론
- [EEVE Team](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) - EEVE 어댑터 아이디어
- [KORMo-Team](https://huggingface.co/KORMo-Team) - KORMo-10B 토크나이저
- [Qwen](https://huggingface.co/Qwen) - Qwen3-Embedding 모델
- [HAERAE-HUB](https://huggingface.co/HAERAE-HUB) - 한국어 데이터셋

---

**프로젝트 상태**: 🎉 Stage 0-2 구현 완료, Thunder+EEVE 방식 적용 완료!
