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
