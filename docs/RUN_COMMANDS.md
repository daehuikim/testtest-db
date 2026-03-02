# 실행 명령어

## 1. 데이터 수집 (2020-01-01 ~ 2025-12-31)

```bash
source .venv/bin/activate  # 또는 conda activate dataset
cd daebong-dataset-main

python scripts/run_collect_all.py
```

기본 기간: 2020-01-01 ~ 2025-12-31  
다른 기간 사용 시:
```bash
python scripts/run_collect_all.py --start 2020-01-01 --end 2025-12-31
```

수집 후 시계열 통합:
```bash
python scripts/run_collect_all.py --combine
```

---

## 2. Feature Selection

```bash
python scripts/run_feature_selection_pipeline.py --save-merged
```

- `config/feature_selection_config.yaml`의 `data_range`, `seasonal_lags` 확인
- 출력: `reports/feature_selection_pipeline_report.json`, `temp/merged_stage0.csv`

---

## 3. 학습

```bash
python scripts/run_training_pipeline.py --skip-shap
```

옵션:
- `--seed 42`: split 고정 (기본)
- `--config config/training_config.yaml`
- `--data-path temp/merged_stage0.csv`
- `--cv purged`: CV 방법 (expanding | timeseries | purged)
- `--model lgb catboost lstm`: 실행할 모델만 지정
- `--no-deep`: LSTM/Transformer 스킵
- `--plot`: 학습 후 inference + 시계열 플롯 생성

학습 완료 시 최고 모델 체크포인트 저장: `checkpoints/best_model/checkpoint.joblib`

---

## 4. Inference & Time Series Plot

학습 후 체크포인트로 test 구간 inference 및 시계열 플롯 생성 (blue=actual train, red=predicted test).

지원 모드: 단일/스태킹, 계절별 라우팅(4개), 월별 라우팅(12개). 체크포인트에 저장된 설정에 따라 자동 적용.

```bash
python scripts/run_inference_and_plot.py
```

옵션:
- `--checkpoint checkpoints/best_model/checkpoint.joblib`
- `--data temp/merged_stage0.csv`
- `--output reports/inference_timeseries.png`

또는 학습 시 `--plot` 옵션으로 한 번에 실행:

```bash
python scripts/run_training_pipeline.py --skip-shap --plot
```

---

## 5. 카테고리컬 보정 실험

mean vs median, domae 가락도매 필터, 저거래건수 처리(exclude vs fill_week) 등 조합 실험:

```bash
python scripts/run_categorical_merge_experiments.py
```

옵션:
- `--analyze-only`: 카테고리컬 컬럼 분석만 (보고서: reports/categorical_analysis_report.md)
- `--skip-training`: merge만 수행, 학습 스킵
- `--exp exp_0 exp_1`: 특정 실험만 실행

출력: reports/categorical_experiments_report.md, reports/categorical_experiments_comparison.png

---

## 전체 파이프라인 (순서)

```bash
# 1. 데이터 수집
python scripts/run_collect_all.py

# 2. Feature selection
python scripts/run_feature_selection_pipeline.py --save-merged

# 3. 학습
python scripts/run_training_pipeline.py --skip-shap
```
