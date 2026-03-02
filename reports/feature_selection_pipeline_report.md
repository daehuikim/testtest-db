# Feature Selection Pipeline 결과

## 품종 (공통 feature 적용)
감홍, 기타, 로얄부사, 루비에스, 미시마, 미야비, 미얀마, 시나노골드, 시나노스위트, 아리수, 아오리, 알프스오토메, 양광, 요까, 홍로, 홍옥, 후지, 후지후브락스

## Stage별 감소
| Stage | 유지 feature 수 |
|-------|----------------|
| 0 (Maximal lag) | 1470 |
| 1 (CCF+MI) | 976 |
| 2 (Elastic Net) | 560 |
| 3 (Rolling Perm) | 2 |
| 4 (Stability) | 1 |
| 5 (Common) | 1 |

## 최종 Feature Set

1. `price_per_kg_mean_lag2`