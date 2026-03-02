# 파이프라인 분석 및 개선 방향

> **적용된 수정 (2026-03)**: LSTM/Transformer seq_len→365, seq_cols 최소 2개, `final_features_ranked` 사용

## 1. 현재 문제 요약

| 문제 | 현상 | 원인 |
|------|------|------|
| LSTM/Transformer inf | CV MAPE = inf | 시퀀스 컬럼 부족 (아래 상세) |
| MAPE 25% → 목표 10% | Baseline 21% 수준 | Feature/Target/모델 한계 |
| Summer MAPE 37% | 계절별 편차 큼 | 여름 가격 변동성 + 데이터 부족 |
| 학습 빠름 | ~30초 수준 | HPO 없음, 단일 설정만 |

---

## 2. LSTM/Transformer 실패 원인 (inf)

### 2.1 시퀀스 구성 로직

```python
# deep_models.py
_get_lag_sequence_cols(features, base="price_per_kg_mean", max_lag=30)
# → price_per_kg_mean_lag1 ~ lag30 중 features에 있는 것만
# → lag <= 30 조건
```

### 2.2 Feature Selection 이후 상태

- **Lag clustering**: `price_per_kg_mean` base당 **1개**만 유지 (예: lag6 또는 lag21)
- **Seasonal lags**: lag364, 365, 366는 `max_lag=30` 때문에 **제외됨**
- 결과: `price_per_kg_mean_lag*` 컬럼이 **1~2개**만 존재
- 조건: `len(seq_cols) >= 5` → **항상 미달** → fold 스킵 → `fold_mapes = []` → **inf 반환**

### 2.3 개선 방향

1. **max_lag 확대**: `seq_len: 365`로 두어 seasonal lag 포함
2. **seq_cols 최소 개수 완화**: `>= 5` → `>= 2` 또는 `>= 3`
3. **다중 base 시퀀스**: `price_per_kg_mean` 뿐 아니라 `domae_amt_mean`, `weather_TA_AVG` 등 lag를 합쳐 시퀀스 구성

---

## 3. MAPE 25% → 10% 달성 방향

### 3.1 농산물 가격 예측의 한계

- **본질적 변동성**: 날씨, 수급, 유통 이슈로 단기 변동이 큼
- **MAPE 10% 이하**는 현실적으로 매우 어려운 목표
- 참고: KREI 등 공공 예측도 15~25% 수준

### 3.2 Feature Selection 쪽 허점

| 단계 | 이슈 | 개선 |
|------|------|------|
| Stage 1 CCF+MI | 상관/상호정보만으로 유의 lag 선별 | 도메인(수확 시기, 유통) 반영 feature 추가 |
| Stage 2~4 | 전 기간 통합 기준 | **계절별(특히 여름) 가중** 또는 별도 모델 |
| Stage 5 Common | 품종 공통만 강조 | 품종별 feature set 분리 검토 |
| Stage 6 Lag cluster | base당 1개로 과도한 축소 | `top_k_per_base: 2` 등으로 완화 |

### 3.3 Training 쪽 허점

| 항목 | 현재 | 개선 |
|------|------|------|
| **Hyperparameter** | 고정값 (n_estimators=500 등) | Optuna/GridSearch HPO |
| **앙상블** | LGB+CatBoost 2개만 stacking | LSTM/Transformer 복구 후 4~5 model stacking |
| **계절 대응** | 없음 | Summer 가중 학습 또는 계절별 모델 |
| **Target** | log1p(price) | Quantile/분위수 회귀, 또는 MAPE 직접 최소화 |

---

## 4. Summer MAPE 37% 개선

### 4.1 원인 추정

- 여름: 냉장/저장 비용, 수급 변동, 품질 이슈로 **가격 변동 폭이 큼**
- 학습 데이터에서 여름 비중이 낮거나, 계절 정보가 feature에 약함

### 4.2 개선 방향

1. **계절 feature 강화**: `date_month`, `is_summer` 등 명시적 계절 변수
2. **Summer 가중 학습**: `sample_weight`로 여름 샘플 가중
3. **계절별 모델**: Spring/Summer/Fall/Winter 별도 모델 후 앙상블
4. **데이터 확장**: 여름 데이터 비중 확대 (가능하다면)

---

## 5. 학습 시간 & 앙상블 탐색

### 5.1 현재 구조

- **HPO 없음**: config 고정값만 사용
- **앙상블**: LGB + CatBoost 2개, OOF corr < 0.95일 때만 Ridge stacking
- **탐색 범위**: 모델 조합, 하이퍼파라미터, feature set 등 거의 없음

### 5.2 개선 방향

1. **Optuna HPO**: LGB, CatBoost에 `n_trials` 적용 (config에 이미 필드 있으나 미사용)
2. **다중 앙상블**: LGB + CatBoost + ElasticNet + (LSTM/Transformer 복구 시) 4~5 model stacking
3. **Feature subset 탐색**: 상위 20/30/40개 feature로 각각 학습 후 비교
4. **CV 강화**: `n_folds` 5 → 10, `valid_days` 30 → 60 등

---

## 6. 우선 적용 권장 순서

| 순위 | 항목 | 예상 효과 | 난이도 |
|------|------|-----------|--------|
| 1 | LSTM/Transformer seq_cols 조건 완화 + max_lag 확대 | inf 해소, deep 모델 참여 | 낮음 |
| 2 | Optuna HPO (LGB, CatBoost) | MAPE 2~5%p 개선 가능 | 중간 |
| 3 | Summer 가중/계절 feature 강화 | Summer MAPE 5~10%p 개선 | 중간 |
| 4 | Lag cluster 완화 (top_k_per_base: 2) | Feature 다양성 확보 | 낮음 |
| 5 | 4~5 model stacking | MAPE 1~3%p 추가 개선 | 중간 |

---

## 7. MAPE 10% 목표에 대한 현실성

- 농산물 경매가 예측에서 **MAPE 10% 이하**는 매우 도전적인 목표
- Baseline(naive lag1) 21%에서 10%로는 **약 50% 오차 감소** 필요
- 단기적으로는 **15~18%**를 목표로 두고, 10%는 장기 목표로 두는 것이 현실적
