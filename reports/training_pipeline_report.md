# Training Pipeline Report

## Baseline
| Model | MAPE (%) |
|-------|----------|
| naive_lag1 | 28.58 |
| seasonal_7d | 26.33 |

## CV Scores
| Model | Mean | Std | Worst |
|-------|------|-----|-------|
| lgb | 33.27 | 5.46 | 41.34 |
| catboost | 31.08 | 4.94 | 39.69 |
| elasticnet | 34.55 | 4.84 | 42.72 |

## Best Model: stacking
Test MAPE: 30.32%

## Season MAPE
| Season | MAPE (%) |
|--------|----------|
| spring | 22.63 |
| summer | 33.14 |
| fall | 31.68 |
| winter | 31.75 |