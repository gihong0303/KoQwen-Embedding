# Hyperparameter Tuning

v3 이후 성능 개선을 위한 하이퍼파라미터 튜닝 실험

## Grid Search 계획

### 현재 Baseline (v3)
- Temperature: 0.05
- Learning Rate: 5e-5
- Batch Size: 64
- **결과**: v3 학습 완료 후 확인

### 튜닝 실험 목록

#### Temperature Variations
1. `temp_003.yaml` - Temperature: 0.03 (sharper)
2. **`temp_005.yaml`** - Temperature: 0.05 (baseline, v3)
3. `temp_007.yaml` - Temperature: 0.07 (softer)

#### Learning Rate Variations
1. `lr_3e5.yaml` - Learning Rate: 3e-5 (conservative)
2. **`lr_5e5.yaml`** - Learning Rate: 5e-5 (baseline, v3)
3. `lr_7e5.yaml` - Learning Rate: 7e-5 (aggressive)

#### Batch Size Variations
1. `batch_48.yaml` - Batch Size: 48 (smaller)
2. **`batch_64.yaml`** - Batch Size: 64 (baseline, v3)
3. `batch_96.yaml` - Batch Size: 96 (larger)

## 실행 순서

1. **v3 학습 완료 & 평가** (현재 진행 중)
   - 결과 확인 후 다음 단계 결정

2. **Best 2-3 조합 선정**
   - v3 결과 기반으로 가장 유망한 조합 선택

3. **추가 실험 실행**
   ```bash
   # 예시
   ./run_tuning_experiment.sh configs/tuning/temp_003.yaml
   ./run_tuning_experiment.sh configs/tuning/lr_7e5.yaml
   ```

4. **최종 모델 선정**
   - 모든 실험 결과 비교
   - 최고 성능 모델을 최종 모델로

## 예상 실험 시간
- 각 실험: ~2-3시간
- 총 실험 (3개): ~6-9시간
