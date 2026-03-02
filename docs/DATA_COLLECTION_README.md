# 데이터 수집 가이드 (2024-01-01 ~ 2025-12-31)

## 개요

`run_collect_all.py`는 auction, somae, domae, weather 4개 소스의 데이터를 병렬로 수집하고,  
`combine_raw_data.py`로 날짜 기준 시계열 raw 데이터를 통합합니다.

---

## 사전 준비

### 1. 환경 변수 (.env)

프로젝트 루트에 `.env` 파일을 생성하고 다음 키를 설정하세요:

```env
# 소매 API (공공데이터포털)
RETAIL_API_BASE_URL=http://211.237.50.150:7080/openapi
RETAIL_API_CODE=Grid_20141225000000000163_1
RETAIL_API_KEY=your_retail_api_key

# 도매 API
WHOLESALE_API_BASE_URL=http://211.237.50.150:7080/openapi
WHOLESALE_API_CODE=Grid_20150406000000000217_1
WHOLESALE_API_KEY=your_wholesale_api_key

# 기상청 API
WEATHER_API_KEY=your_weather_api_key
WEATHER_STATION_CODE=108

# 제품 코드 (선택)
PRODUCT_CODE=411

# 호환성 (API_KEY 하나로 통일 시)
API_KEY=your_api_key
```

- **auction**: API 키 불필요 (가락시장만 수집)
- **somae**: `RETAIL_API_KEY` 또는 `API_KEY`
- **domae**: `WHOLESALE_API_KEY` 또는 `API_KEY`
- **weather**: `WEATHER_API_KEY` 또는 `WEATHER_AUTH_KEY`, `WEATHER_STATION_CODE` (108=서울)

### 2. 패키지 설치

```bash
source .venv/bin/activate  # 또는 activate
pip install -r requirements.txt
```

---

## 실행 방법

### 전체 수집 (2024-01-01 ~ 2025-12-31)

```bash
cd daebong-dataset-main
source .venv/bin/activate
python scripts/run_collect_all.py
```

### 기간 지정

```bash
python scripts/run_collect_all.py --start 2024-01-01 --end 2025-12-31
```

### 특정 소스만 수집

```bash
# auction만 (API 키 불필요)
python scripts/run_collect_all.py --only auction

# auction + weather
python scripts/run_collect_all.py --only auction weather
```

### 시계열 통합

수집 완료 후:

```bash
python scripts/combine_raw_data.py --start 20240101 --end 20251231
```

출력: `data/raw/combined/raw_combined.csv`

---

## 병목 개선 요약

| 소스   | 기존 방식              | 개선 방식                    | 예상 단축 |
|--------|------------------------|------------------------------|-----------|
| auction | 날짜×시장 순차 + 1~2초 sleep | **가락시장만** + 날짜별 병렬 | **대폭 단축** |
| somae  | 날짜 순차              | 날짜 배치 병렬 (10 workers)  | **약 7배** |
| domae  | 날짜 순차              | 날짜 배치 병렬 (10 workers)  | **약 7배** |
| weather | -                    | 월별 병렬 (6 workers)        | -         |

자세한 분석: [BOTTLENECK_ANALYSIS.md](./BOTTLENECK_ANALYSIS.md)

---

## 출력 구조

```
data/raw/
├── auction/   # auction_20240101_20251231.csv
├── somae/     # somae_20240101_20251231.csv
├── domae/     # domae_20240101_20251231.csv
├── weather/   # weather_20240101_20251231.csv
└── combined/  # raw_combined.csv (날짜 기준 통합)
```

---

## 로그

주요 단계(API 요청, 저장, 에러)는 로그로 기록됩니다.  
재실행 시 기존 파일은 덮어쓰므로, 필요 시 백업 후 실행하세요.
