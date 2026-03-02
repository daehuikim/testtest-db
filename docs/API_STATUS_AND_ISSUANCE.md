# API 상태 및 키 발급 안내

## 요약

| 스크립트 | 상태 | API 키 | 비고 |
|----------|------|--------|------|
| **auction** | ✅ 정상 | 불필요 | 가락시장 웹 크롤링 |
| **somae** | ✅ 정상 | API_KEY | 2014~2023 데이터 (2024는 API 미제공 가능) |
| **domae** | ✅ 정상 | API_KEY | 2014~2023 데이터 (2024는 API 미제공 가능) |
| **weather** | ⚠️ 별도 키 | WEATHER_API_KEY | 기상청 별도 발급 |

---

## .env 설정 예시

```env
# 농림축산식품 공공데이터 (somae, domae)
API_KEY=발급받은_농림축산식품_API키

# 기상청 (weather) - 별도 발급
WEATHER_API_KEY=발급받은_기상청_API키
WEATHER_STATION_CODE=108

# 제품 코드 (선택, 기본 411=사과)
PRODUCT_CODE=411
```

---

## 1. AUCTION (경매) ✅

- **출처**: https://at.agromarket.kr (한국농수산식품유통공사)
- **API 키**: 불필요 (공개 웹페이지)
- **가락시장**: whsal_cd=110001

---

## 2. SOMAE (소매가격) ✅

- **API**: Grid_20141225000000000163_1
- **키**: `API_KEY` (농림축산식품 공공데이터)
- **발급**: https://data.mafra.go.kr/main.do → 회원가입 → 오픈 API 신청
- **데이터**: 2014~2023 (2024일부는 API 미제공 가능)

---

## 3. DOMAE (도매가격) ✅

- **API**: Grid_20150406000000000217_1
- **키**: `API_KEY` (농림축산식품 공공데이터)
- **발급**: somae와 동일 (같은 포털)
- **데이터**: 2014~2023 (2024일부는 API 미제공 가능)

---

## 4. WEATHER (기상) ⚠️

- **API**: kma_sfcdd3.php (지상관측 일자료)
- **키**: `WEATHER_API_KEY` (농림 API와 별도)
- **발급**: https://apihub.kma.go.kr/ → 회원가입 → 인증키 발급
- **대안**: 공공데이터포털(data.go.kr)에서 기상청 API 활용신청

---

## 테스트

```bash
python scripts/test_apis.py all
```

---

## 2024~2025 데이터 참고

- **domae/somae**: 농림축산식품 API는 2024년 데이터를 아직 제공하지 않을 수 있음. **2023년까지는 정상 수집 가능** (사과 411, 후지 등).
- **auction**: 2024~2025 실시간 수집 가능.
- **weather**: 기상청 API로 2024~2025 데이터 수집 가능 (WEATHER_API_KEY 별도 발급). 403일 경우 일일 호출 한도 초과.
