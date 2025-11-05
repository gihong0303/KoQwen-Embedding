# 한국어 임베딩 확장 학습 파이프라인

## 📁 프로젝트 구조

```
ko-embedding-expansion/
├── scripts/
│   ├── train_stage1.py           # Stage 1: SimCSE 학습
│   ├── train_stage2.py           # Stage 2: EEVE Adapter 학습
│   ├── train_stage3.py           # Stage 3: Hierarchical Adapter 학습
│   ├── evaluate_models.py        # 모든 Stage 평가
│   └── utils/
│       ├── model_utils.py        # 모델 로딩 유틸리티
│       ├── train_utils.py        # 학습 유틸리티
│       ├── data_utils.py         # 데이터 로딩 유틸리티
│       └── eeve_adapter.py       # 어댑터 구현
├── run_stage1.sh                 # Stage 1 실행 스크립트
├── run_stage2.sh                 # Stage 2 실행 스크립트
├── run_stage3.sh                 # Stage 3 실행 스크립트
├── configs/
│   └── training_config.yaml      # 학습 설정
├── checkpoints/                  # 학습된 모델 저장
│   ├── stage1/final/
│   ├── stage2/final/
│   └── stage3/final/
└── logs/                         # 학습 로그
```

## 🚀 학습 실행 방법

### Stage 1: SimCSE 무지도 학습
```bash
./run_stage1.sh
```
- **입력**: `outputs/koqwen-expanded` (어휘 확장 모델)
- **출력**: `checkpoints/stage1/final`
- **특징**: 새로운 한국어 토큰 임베딩 기초 적응
- **GPU**: 6개 사용, 배치 크기 2048

### Stage 2: EEVE Bottleneck Adapter
```bash
./run_stage2.sh
```
- **입력**: `checkpoints/stage1/final`
- **출력**: `checkpoints/stage2/final`
- **특징**: Stage 1에 Bottleneck Adapter 추가
- **GPU**: 6개 사용, 배치 크기 48

### Stage 3: Hierarchical Adapter
```bash
./run_stage3.sh
```
- **입력**: `checkpoints/stage2/final`
- **출력**: `checkpoints/stage3/final`
- **특징**: Stage 2에 Hierarchical Adapter 추가 (language + task 계층)
- **GPU**: 6개 사용, 배치 크기 16, GPU 활용률 97-98%

## 📊 평가 실행

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_models.py
```

평가 결과:
- Stage 0 (Vocab Expanded)
- Stage 1 (SimCSE)
- Stage 2 (EEVE Adapter)
- Stage 3 (Hierarchical Adapter)

## 📈 현재 성능 (2024-11-03 기준)

| 모델 | 유사 문장 점수↑ | 다른 문장 점수↓ | 구분도↑ | 비고 |
|------|----------------|----------------|---------|------|
| Stage 1 (SimCSE) | 0.8579 | 0.4257 | **0.4323** | 🏆 최고 |
| Stage 0 (Vocab) | 0.8624 | 0.4511 | 0.4112 | |
| Stage 3 (Hierarchical) | 0.4967 | 0.1286 | 0.3681 | ⚠️ |
| Stage 2 (EEVE) | 0.4835 | 0.1219 | 0.3617 | ⚠️ |

**⚠️ 주의**: Stage 2와 3는 기술적으로 정상 작동하지만, 어댑터 학습이 오히려 성능을 악화시킴.
SimCSE 학습 방식이 어댑터에 적합하지 않을 수 있음.

## 🔧 기술적 세부사항

### Stage 1
- **데이터**: KOREAN-WEBTEXT (10K 샘플)
- **방법**: SimCSE (dropout 기반 contrastive learning)
- **학습**: 2 epochs, LR 5e-5
- **상태**: ✅ 정상 작동, 최고 성능

### Stage 2
- **데이터**: KOREAN-WEBTEXT (50K 샘플)
- **방법**: Bottleneck Adapter (hidden_size → 256 → hidden_size)
- **학습**: 1 epoch, LR 3e-5
- **상태**: ⚠️ 학습 완료했으나 성능 저하

### Stage 3
- **데이터**: KOREAN-WEBTEXT (50K 샘플)
- **방법**: Hierarchical Adapter (language + task 계층)
- **학습**: 3 epochs, LR 3e-4
- **최적화**: Gradient accumulation, 97-98% GPU 활용
- **상태**: ⚠️ 학습 완료했으나 성능 저하

## 🐛 알려진 문제

1. **Stage 2/3 성능 저하**: 어댑터가 SimCSE 학습 시 임베딩 품질을 악화시킴
   - 증상: 유사 문장 쌍 점수가 매우 낮음 (0.03~0.10)
   - 증상: 다른 문장 쌍에서 음수 코사인 유사도 발생
   - 원인: SimCSE objective가 어댑터 학습에 부적합할 수 있음

2. **개선 방향**:
   - Supervised 데이터 사용 (positive/negative pairs)
   - Learning rate 조정 (현재 너무 높을 수 있음)
   - Temperature 조정
   - 다른 loss 함수 시도 (Triplet loss, InfoNCE 등)

## 📝 로그 파일

- `logs/stage1_enhanced.log`: Stage 1 학습 로그
- `logs/stage2.log`: Stage 2 학습 로그
- `logs/stage3_optimized.log`: Stage 3 학습 로그
- `logs/evaluation_correct.log`: 최종 평가 로그

## 🔄 재학습 방법

특정 Stage를 처음부터 다시 학습하려면:

```bash
# 체크포인트 백업 (선택사항)
mv checkpoints/stage2 checkpoints/stage2_backup

# 재학습
./run_stage2.sh
```

## ⚙️ 설정 변경

`configs/training_config.yaml`에서 모든 하이퍼파라미터 조정 가능:
- 데이터셋 크기
- 배치 크기
- Learning rate
- 에폭 수
- 어댑터 크기
- 등등

---

**생성일**: 2024-11-03
**마지막 업데이트**: Stage 3 학습 완료 및 평가
