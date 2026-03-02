# Feature Selection Pipeline 조절 가이드

## 0. GPU 사용

```json
"use_gpu": true
```
→ LightGBM 사용 Stage (2 fallback, 3, 4, 5)에서 GPU 가속. CUDA 필요.

## 1. Feature 수 조절 (너무 적게/많이 나올 때)

### config/feature_selection_config.json (또는 .yaml) 수정

| 상황 | 조절 위치 | 조절 방법 |
|------|----------|----------|
| **너무 적게 나옴** | `stage3_rolling_permutation` | `min_kept` ↑ (예: 50→80), `top_percentile` ↑ (예: 50→70), `stability_lambda` ↓ (예: 0.3→0.1) |
| | `stage4_stability` | `min_kept` ↑ (예: 30→50) |
| | `stage5_common` | `min_kept` ↑ (예: 20→40), `min_variety_ratio` ↓ (예: 0.5→0.3), `top_k_per_variety` ↑ (예: 80→100) |
| **너무 많이 나옴** | `stage1_prefilter` | `ccf_threshold` ↑ (예: 0.15→0.2), `mi_percentile` ↑ (예: 75→85) |
| | `stage3_rolling_permutation` | `min_kept` ↓, `top_percentile` ↓, `stability_lambda` ↑ |
| | `stage5_common` | `min_variety_ratio` ↑ (예: 0.5→0.7), `top_k_per_variety` ↓ |

### Elastic Net 수렴 경고 (ConvergenceWarning)

**방법 1: LightGBM으로 대체 (권장, GPU 사용 가능)**
```json
"stage2_elasticnet": {
  "use_lightgbm_fallback": true
}
```
→ Elastic Net 대신 LightGBM importance 사용. GPU 가속.

**방법 2: Elastic Net 유지**
- `max_iter` ↑ (50000)
- `tol` ↑ (0.001 → 0.01) 수렴 허용 오차 완화
- `alphas`에 더 큰 값 추가 (100.0, 1000.0)

---

## 2. Popular 품종만 사용하기

### config에 variety_filter 추가

```json
"variety_filter": {
  "min_pct": 1.0,
  "whitelist": null,
  "max_varieties": null
}
```

### 옵션 3가지 (하나만 사용)

| 옵션 | 설명 | 예시 |
|------|------|------|
| `whitelist` | 지정 품종만 사용 | `["후지", "홍로", "미시마", "아오리"]` |
| `min_pct` | 전체 대비 비율 N% 이상인 품종만 | `1.0` → 1% 이상 |
| `max_varieties` | 상위 N개 품종만 | `10` → Top 10 |

---

## 3. 품종별 비율 (auction 기준)

| 품종 | 비율 | 행 수 |
|------|------|-------|
| 후지 | 21.0% | 62,441 |
| 홍로 | 20.4% | 60,641 |
| 미시마 | 16.8% | 50,052 |
| 기타 | 13.3% | 39,650 |
| 미얀마 | 6.8% | 20,177 |
| 아오리 | 6.3% | 18,645 |
| 미야비 | 2.8% | 8,202 |
| 감홍 | 2.3% | 6,979 |
| 시나노골드 | 1.9% | 5,544 |
| 양광 | 1.8% | 5,341 |
| 후지후브락스 | 1.8% | 5,215 |
| 시나노스위트 | 1.0% | 2,944 |
| 썸머킹 | 1.0% | 2,929 |
| 아리수 | 0.7% | 2,078 |
| 홍옥 | 0.4% | 1,248 |
| 알프스오토메 | 0.4% | 1,191 |
| 요까 | 0.3% | 1,039 |
| 로얄부사 | 0.3% | 1,031 |
| 그 외 | <0.3% | 각 100~350 |

### Popular 품종 추천 (whitelist 예시)

```json
"whitelist": ["후지", "홍로", "미시마", "아오리", "미야비", "감홍", "시나노골드", "양광"]
```

또는 `min_pct: 1.0` → 1% 이상 12개 품종만 사용.

---

## 4. 빠른 확인

품종별 비율 재확인:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/auction/auction_20240101_20251231.csv', encoding='utf-8-sig')
df = df[df['품종'].notna() & df['품종'].str.strip().ne('')]
cnt = df['품종'].value_counts()
pct = (cnt / len(df) * 100).round(1)
for v, c in cnt.items():
    print(f'{v}: {pct[v]:.1f}%')
"
```
