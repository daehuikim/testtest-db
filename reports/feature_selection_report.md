# Feature Selection 리포트

## 1. 목표
- **Target**: `price_per_kg_mean` (일별 품종별 **1kg당 경락가** 평균)
  - 단량(20kg, 10kg 등) 파싱 후 단량당 경락가/kg로 환산한 값의 평균
- **도매/소매가**: 당일 경매 후 수집되므로 **lag 1~7일**만 사용
- **품종별 모델**: 사용자가 품종을 선택하면 해당 품종 전용 ML 모델 학습
- **데이터 소스**: auction, domae, somae, weather

## 2. 컬럼 프로파일 (컬럼별 5개 샘플)

### 2.1 후보 컬럼 (candidate)
| 컬럼 | dtype | n_unique | null% | 분류 | 샘플(5개) |
|------|-------|----------|-------|------|-----------|
| auction_quantity_sum | float64 | 1887 | 0.0% | numeric_ok | 1491.0, 855.0, 66.0, 6324.0, 501.0 |
| auction_quantity_mean | float64 | 2837 | 0.0% | numeric_ok | 28.132075471698112, 7.137931034482759, 19.53846153846154,... |
| auction_transaction_count | int64 | 431 | 0.0% | numeric_ok | 53, 103, 173, 201, 326 |
| domae_amt_mean_lag1 | float64 | 303 | 90.91% | numeric_ok | 88450.0, 64020.0, 84580.0, 96430.0, 89190.0 |
| domae_amt_median_lag1 | float64 | 164 | 90.91% | numeric_ok | 89500.0, 67300.0, 85150.0, 84250.0, 87500.0 |
| domae_amt_std_lag1 | float64 | 318 | 90.91% | numeric_ok | 10065.259283517958, 8190.678984200409, 9921.043851889324,... |
| domae_market_count_lag1 | float64 | 7 | 90.91% | numeric_ok | 10.0, 8.0, 4.0, 7.0, 2.0 |
| domae_amt_mean_lag2 | float64 | 247 | 92.84% | numeric_ok | 88450.0, 65280.0, 83250.0, 96430.0, 89000.0 |
| domae_amt_median_lag2 | float64 | 143 | 92.84% | numeric_ok | 89500.0, 65650.0, 84500.0, 81500.0, 87500.0 |
| domae_amt_std_lag2 | float64 | 257 | 92.84% | numeric_ok | 10065.259283517958, 7306.2834449381835, 8729.8275406155, ... |
| domae_market_count_lag2 | float64 | 7 | 92.84% | numeric_ok | 10.0, 8.0, 4.0, 7.0, 2.0 |
| domae_amt_mean_lag3 | float64 | 244 | 92.79% | numeric_ok | 88450.0, 64890.0, 82120.0, 96370.0, 88800.0 |
| domae_amt_median_lag3 | float64 | 143 | 92.79% | numeric_ok | 89500.0, 67300.0, 79500.0, 80950.0, 86150.0 |
| domae_amt_std_lag3 | float64 | 257 | 92.79% | numeric_ok | 10065.259283517958, 8190.678984200409, 9921.043851889324,... |
| domae_market_count_lag3 | float64 | 6 | 92.79% | numeric_ok | 10.0, 8.0, 6.0, 4.0, 2.0 |
| domae_amt_mean_lag4 | float64 | 249 | 92.81% | numeric_ok | 88450.0, 64850.0, 84580.0, 95820.0, 88800.0 |
| domae_amt_median_lag4 | float64 | 142 | 92.81% | numeric_ok | 89500.0, 64800.0, 86150.0, 78700.0, 86500.0 |
| domae_amt_std_lag4 | float64 | 255 | 92.81% | numeric_ok | 10065.259283517958, 8190.678984200409, 10381.527397792257... |
| domae_market_count_lag4 | float64 | 6 | 92.81% | numeric_ok | 10.0, 8.0, 6.0, 4.0, 2.0 |
| domae_amt_mean_lag5 | float64 | 242 | 92.98% | numeric_ok | 82590.0, 64850.0, 81190.0, 96430.0, 88800.0 |
| domae_amt_median_lag5 | float64 | 140 | 92.98% | numeric_ok | 83000.0, 64800.0, 84500.0, 84000.0, 82700.0 |
| domae_amt_std_lag5 | float64 | 249 | 92.98% | numeric_ok | 7573.117661248313, 7430.986176514901, 8729.8275406155, 14... |
| domae_market_count_lag5 | float64 | 7 | 92.98% | numeric_ok | 10.0, 8.0, 6.0, 7.0, 2.0 |
| domae_amt_mean_lag6 | float64 | 239 | 92.92% | numeric_ok | 87880.0, 64590.0, 84580.0, 99225.0, 88800.0 |
| domae_amt_median_lag6 | float64 | 144 | 92.92% | numeric_ok | 89500.0, 68800.0, 82500.0, 84000.0, 85800.0 |
| domae_amt_std_lag6 | float64 | 246 | 92.92% | numeric_ok | 10024.170788648804, 8527.836771420993, 9340.38424144199, ... |
| domae_market_count_lag6 | float64 | 7 | 92.92% | numeric_ok | 10.0, 8.0, 4.0, 7.0, 2.0 |
| domae_amt_mean_lag7 | float64 | 298 | 91.07% | numeric_ok | 88450.0, 64020.0, 85650.0, 96370.0, 88800.0 |
| domae_amt_median_lag7 | float64 | 163 | 91.07% | numeric_ok | 89500.0, 67300.0, 85150.0, 80950.0, 87500.0 |
| domae_amt_std_lag7 | float64 | 312 | 91.07% | numeric_ok | 10065.259283517958, 8527.836771420993, 9340.38424144199, ... |
| domae_market_count_lag7 | float64 | 7 | 91.07% | numeric_ok | 10.0, 8.0, 4.0, 7.0, 2.0 |
| somae_amt_mean_lag1 | float64 | 361 | 90.12% | numeric_ok | 28766.666666666668, 21892.55319148936, 24885.263157894737... |
| somae_amt_median_lag1 | float64 | 90 | 90.12% | numeric_ok | 26600.0, 21100.0, 23700.0, 27800.0, 24600.0 |
| somae_amt_std_lag1 | float64 | 363 | 90.12% | numeric_ok | 7550.861357909891, 5295.935535489924, 5555.309396064476, ... |
| somae_market_count_lag1 | float64 | 55 | 90.12% | numeric_ok | 93.0, 42.0, 91.0, 72.0, 25.0 |
| somae_amt_mean_lag2 | float64 | 286 | 92.22% | numeric_ok | 28766.666666666668, 22063.82978723404, 25807.36842105263,... |
| somae_amt_median_lag2 | float64 | 81 | 92.22% | numeric_ok | 26600.0, 21700.0, 23700.0, 24850.0, 24600.0 |
| somae_amt_std_lag2 | float64 | 286 | 92.22% | numeric_ok | 7550.861357909891, 5243.918158031782, 5535.8453217780325,... |
| somae_market_count_lag2 | float64 | 51 | 92.22% | numeric_ok | 93.0, 57.0, 53.0, 67.0, 80.0 |
| somae_amt_mean_lag3 | float64 | 288 | 92.16% | numeric_ok | 26578.65168539326, 21035.05154639175, 25455.78947368421, ... |
| somae_amt_median_lag3 | float64 | 79 | 92.16% | numeric_ok | 25000.0, 20800.0, 23200.0, 27150.0, 24600.0 |
| somae_amt_std_lag3 | float64 | 288 | 92.16% | numeric_ok | 7661.733009368293, 5492.059442904695, 5358.284811898294, ... |
| somae_market_count_lag3 | float64 | 49 | 92.16% | numeric_ok | 89.0, 50.0, 92.0, 67.0, 65.0 |
| somae_amt_mean_lag4 | float64 | 282 | 92.3% | numeric_ok | 27577.52808988764, 21892.55319148936, 24801.052631578947,... |
| somae_amt_median_lag4 | float64 | 81 | 92.3% | numeric_ok | 25900.0, 18500.0, 23700.0, 27900.0, 24600.0 |
| somae_amt_std_lag4 | float64 | 283 | 92.3% | numeric_ok | 6898.645334889188, 5295.935535489924, 5328.302802044938, ... |
| somae_market_count_lag4 | float64 | 45 | 92.3% | numeric_ok | 89.0, 34.0, 88.0, 72.0, 36.0 |
| somae_amt_mean_lag5 | float64 | 276 | 92.46% | numeric_ok | 28766.666666666668, 21939.36170212766, 25455.78947368421,... |
| somae_amt_median_lag5 | float64 | 79 | 92.46% | numeric_ok | 26600.0, 20800.0, 23200.0, 27900.0, 24100.0 |
| somae_amt_std_lag5 | float64 | 277 | 92.46% | numeric_ok | 7550.861357909891, 5295.935535489924, 5638.263665220435, ... |
| somae_market_count_lag5 | float64 | 47 | 92.46% | numeric_ok | 93.0, 40.0, 91.0, 84.0, 65.0 |
| somae_amt_mean_lag6 | float64 | 279 | 92.41% | numeric_ok | 28766.666666666668, 21892.55319148936, 25434.736842105263... |
| somae_amt_median_lag6 | float64 | 86 | 92.41% | numeric_ok | 26600.0, 18500.0, 23700.0, 27900.0, 24600.0 |
| somae_amt_std_lag6 | float64 | 279 | 92.41% | numeric_ok | 7550.861357909891, 5295.935535489924, 5701.675162972172, ... |
| somae_market_count_lag6 | float64 | 48 | 92.41% | numeric_ok | 93.0, 40.0, 81.0, 67.0, 82.0 |
| somae_amt_mean_lag7 | float64 | 351 | 90.45% | numeric_ok | 28766.666666666668, 21892.55319148936, 25729.473684210527... |
| somae_amt_median_lag7 | float64 | 91 | 90.45% | numeric_ok | 26600.0, 20800.0, 23700.0, 27800.0, 24600.0 |
| somae_amt_std_lag7 | float64 | 351 | 90.45% | numeric_ok | 7550.861357909891, 5295.935535489924, 5555.309396064476, ... |
| somae_market_count_lag7 | float64 | 52 | 90.45% | numeric_ok | 93.0, 74.0, 95.0, 67.0, 36.0 |
| weather_WS_AVG | float64 | 125 | 41.05% | numeric_ok | 2.6, 1.3333333333333333, 0.9333333333333335, 3.1999999999... |
| weather_WR_DAY | float64 | 389 | 41.05% | numeric_ok | 2264.0, 1071.6666666666667, 729.3333333333334, 1455.0, 29... |
| weather_WD_MAX | float64 | 63 | 41.05% | numeric_ok | 30.0, 11.0, 21.0, 10.666666666666666, 16.333333333333332 |
| weather_WS_MAX | float64 | 187 | 41.05% | numeric_ok | 6.1000000000000005, 8.566666666666666, 2.2666666666666666... |
| weather_WS_MAX_TM | float64 | 384 | 41.05% | numeric_ok | 1789.3333333333333, 944.3333333333334, 1708.3333333333333... |
| weather_WD_INS | float64 | 68 | 41.05% | numeric_ok | 30.666666666666668, 14.0, 26.666666666666668, 19.66666666... |
| weather_WS_INS | float64 | 245 | 41.05% | numeric_ok | 10.333333333333334, 8.700000000000001, 6.933333333333334,... |
| weather_WS_INS_TM | float64 | 371 | 41.05% | numeric_ok | 1380.3333333333333, 1678.3333333333333, 1559.666666666666... |
| weather_TA_AVG | float64 | 337 | 41.05% | numeric_ok | 1.8666666666666665, 17.400000000000002, 25.40000000000000... |
| weather_TA_MAX | float64 | 343 | 41.05% | numeric_ok | 5.2, 17.3, 32.666666666666664, 6.933333333333334, 2.06666... |
| weather_TA_MAX_TM | float64 | 346 | 41.05% | numeric_ok | 1299.6666666666667, 1614.6666666666667, 1055.333333333333... |
| weather_TA_MIN | float64 | 352 | 41.05% | numeric_ok | -0.10000000000000002, 10.466666666666667, 16.266666666666... |
| weather_TA_MIN_TM | float64 | 365 | 41.05% | numeric_ok | 1122.0, 1406.3333333333333, 477.3333333333333, 638.333333... |
| weather_TD_AVG | float64 | 348 | 41.05% | numeric_ok | -2.233333333333333, 7.2, 21.900000000000002, -7.233333333... |
| weather_TS_AVG | float64 | 348 | 41.05% | numeric_ok | 1.8333333333333333, 15.033333333333333, 34.33333333333333... |
| weather_TG_MIN | float64 | 350 | 41.05% | numeric_ok | -2.0, 10.666666666666666, 20.633333333333336, -7.23333333... |
| weather_HM_AVG | float64 | 360 | 41.05% | numeric_ok | 74.66666666666667, 81.43333333333334, 83.93333333333332, ... |
| weather_HM_MIN | float64 | 169 | 41.05% | numeric_ok | 56.333333333333336, 48.666666666666664, 25.66666666666666... |
| weather_HM_MIN_TM | float64 | 367 | 41.05% | numeric_ok | 1376.6666666666667, 1320.3333333333333, 1459.333333333333... |
| weather_PV_AVG | float64 | 311 | 41.05% | numeric_ok | 5.2, 12.533333333333333, 26.7, 7.2, 1.6666666666666667 |
| weather_EV_S | float64 | 184 | 41.05% | numeric_ok | 1.25, 7.35, 6.449999999999999, 2.7, 1.2 |
| weather_EV_L | float64 | 150 | 41.05% | numeric_ok | 0.8500000000000001, 3.85, 5.95, 0.4, 2.35 |
| weather_FG_DUR | float64 | 25 | 94.53% | numeric_ok | 2.5, 2.08, 6.33, 0.25, 1.92 |
| weather_PA_AVG | float64 | 318 | 41.05% | numeric_ok | 995.8000000000001, 1003.3333333333334, 986.5666666666666,... |
| weather_PS_AVG | float64 | 319 | 41.05% | numeric_ok | 1019.0666666666666, 1012.4, 1013.5333333333333, 1024.5333... |
| weather_PS_MAX | float64 | 318 | 41.05% | numeric_ok | 1021.4, 1017.3333333333334, 1009.9, 1031.8333333333333, 1... |
| weather_PS_MAX_TM | float64 | 335 | 41.05% | numeric_ok | 2353.0, 1906.0, 2328.6666666666665, 852.6666666666666, 21... |
| weather_PS_MIN | float64 | 330 | 41.05% | numeric_ok | 1016.2666666666668, 1010.3666666666667, 1011.9, 1024.4666... |
| weather_PS_MIN_TM | float64 | 363 | 41.05% | numeric_ok | 1363.0, 1597.0, 571.3333333333334, 1601.6666666666667, 14... |
| weather_CA_TOT | float64 | 234 | 41.05% | numeric_ok | 6.266666666666667, 9.5, 3.5666666666666664, 2.30000000000... |
| weather_SS_DAY | float64 | 243 | 41.05% | numeric_ok | 2.6, 11.4, 10.966666666666667, 9.766666666666666, 2.66666... |
| weather_SS_DUR | float64 | 115 | 41.05% | numeric_ok | 9.7, 11.4, 13.6, 12.666666666666666, 9.666666666666666 |
| weather_SI_DAY | float64 | 388 | 41.05% | numeric_ok | 6.265000000000001, 18.835, 16.990000000000002, 20.59, 11.885 |
| weather_SI_60M_MAX | float64 | 312 | 41.05% | numeric_ok | 1.26, 2.145, 1.6, 3.1399999999999997, 2.0549999999999997 |
| weather_SI_60M_MAX_TM | float64 | 14 | 41.05% | numeric_ok | 1100.0, 1200.0, 1000.0, 1400.0, 1500.0 |
| weather_RN_DAY | float64 | 113 | 74.09% | numeric_ok | 0.35, 15.533333333333333, 4.233333333333333, 1.8, 1.53333... |
| weather_RN_D99 | float64 | 65 | 79.56% | numeric_ok | 0.0, 24.5, 0.7, 11.5, 2.9 |
| weather_RN_DUR | float64 | 98 | 78.82% | numeric_ok | 2.0, 4.5, 1.75, 10.0, 0.72 |
| weather_RN_60M_MAX | float64 | 66 | 85.82% | numeric_ok | 2.9333333333333336, 2.8333333333333335, 10.06666666666666... |
| weather_RN_60M_MAX_TM | float64 | 65 | 90.69% | numeric_ok | 1062.0, 1502.3333333333333, 1850.0, 1703.6666666666667, 5... |
| weather_RN_10M_MAX | float64 | 59 | 85.82% | numeric_ok | 0.7333333333333334, 3.966666666666667, 2.6333333333333333... |
| weather_RN_10M_MAX_TM | float64 | 64 | 90.69% | numeric_ok | 1127.6666666666667, 1400.0, 1858.0, 1926.0, 613.666666666... |
| weather_SD_NEW | float64 | 10 | 97.82% | numeric_ok | 1.8, 0.3, 1.6, 6.5, 0.7 |
| weather_SD_NEW_TM | float64 | 13 | 97.82% | numeric_ok | 2355.0, 755.0, 953.0, 2255.0, 556.0 |
| weather_SD_MAX | float64 | 12 | 97.44% | numeric_ok | 1.8, 0.3, 1.6, 6.5, 0.7 |
| weather_SD_MAX_TM | float64 | 13 | 97.44% | numeric_ok | 2355.0, 755.0, 953.0, 2255.0, 955.0 |
| date_year | int32 | 2 | 0.0% | numeric_ok | 2024, 2025 |
| date_month | int32 | 12 | 0.0% | numeric_ok | 1, 3, 6, 9, 12 |
| date_day | int32 | 31 | 0.0% | numeric_ok | 3, 11, 20, 29, 28 |
| date_dayofweek | int32 | 6 | 0.0% | numeric_ok | 2, 3, 4, 5, 1 |

### 2.2 제외 컬럼 (drop)
| 컬럼 | dtype | n_unique | null% | 분류 | 샘플(5개) |
|------|-------|----------|-------|------|-----------|
| weather_SS_CMB | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_RN_POW_MAX | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_RN_POW_MAX_TM | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_TE_05 | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_TE_10 | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_TE_15 | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_TE_30 | float64 | 0 | 100.0% | constant | None, None, None, None, None |
| weather_TE_50 | float64 | 0 | 100.0% | constant | None, None, None, None, None |

## 3. Feature Importance (품종: 후지)

### 3.1 Gini (RandomForestClassifier, target 5-quantile 구간화)
| 순위 | 컬럼 | 설명 | 수치 |
|------|------|------|------|
| 1 | auction_quantity_mean | 해당일 품종별 거래당 평균 수량 | 0.0388 |
| 2 | auction_quantity_sum | 해당일 품종별 경매 수량 합계 | 0.0386 |
| 3 | auction_transaction_count | 해당일 품종별 경매 거래 건수 | 0.0332 |
| 4 | domae_amt_std_lag7 | 7일 전 도매가 표준편차 | 0.0197 |
| 5 | somae_amt_mean_lag1 | 1일 전 소매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0173 |
| 6 | date_day | 일 | 0.0172 |
| 7 | weather_TG_MIN | 최저 초상온도 | 0.0163 |
| 8 | date_month | 월 (계절성/수확시기) | 0.0161 |
| 9 | somae_amt_std_lag1 | 1일 전 소매가 표준편차 | 0.0148 |
| 10 | weather_PS_MIN_TM | 최저 지면기압 시각 | 0.0137 |
| 11 | domae_amt_std_lag1 | 1일 전 도매가 표준편차 | 0.0136 |
| 12 | date_dayofweek | 요일 (0=월) | 0.0134 |
| 13 | weather_WD_MAX | 최대 풍향 | 0.0134 |
| 14 | weather_HM_AVG | 평균 상대습도(%) | 0.0134 |
| 15 | somae_amt_median_lag1 | 1일 전 소매가 중앙값 | 0.0131 |
| 16 | weather_TA_MAX | 최고 기온 | 0.0129 |
| 17 | weather_TA_MIN | 최저 기온 | 0.0128 |
| 18 | somae_amt_mean_lag7 | 7일 전 소매가 평균 | 0.0128 |
| 19 | domae_amt_mean_lag7 | 7일 전 도매가 평균 | 0.0128 |
| 20 | weather_WS_INS | 순간 풍속 | 0.0127 |
| 21 | weather_HM_MIN | 최소 상대습도 | 0.0127 |
| 22 | weather_TS_AVG | 평균 지표면온도 | 0.0124 |
| 23 | weather_CA_TOT | 전운량 | 0.0122 |
| 24 | weather_WS_INS_TM | 순간 풍속 시각 | 0.0122 |
| 25 | domae_amt_std_lag2 | 2일 전 도매가 표준편차 | 0.0121 |
| 26 | weather_TA_MAX_TM | 최고 기온 시각 | 0.0120 |
| 27 | weather_HM_MIN_TM | 최소 습도 시각 | 0.0120 |
| 28 | domae_amt_mean_lag1 | 1일 전 도매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0119 |
| 29 | weather_SI_DAY | 일사량 | 0.0118 |
| 30 | weather_PV_AVG | 평균 현지기압(hPa) | 0.0118 |
| 31 | weather_SI_60M_MAX | 60분 최대 일사량 | 0.0115 |
| 32 | weather_PS_MAX_TM | 최고 지면기압 시각 | 0.0113 |
| 33 | weather_WR_DAY | 풍향 일교차 | 0.0113 |
| 34 | weather_SS_DUR | 일조 지속시간 | 0.0111 |
| 35 | somae_amt_std_lag6 | 6일 전 소매가 표준편차 | 0.0111 |
| 36 | weather_WD_INS | 순간 풍향 | 0.0110 |
| 37 | weather_EV_S | 소형증발량 | 0.0110 |
| 38 | weather_PS_MAX | 최고 지면기압 | 0.0108 |
| 39 | somae_amt_std_lag7 | 7일 전 소매가 표준편차 | 0.0108 |
| 40 | weather_TD_AVG | 평균 이슬점온도 | 0.0105 |
| 41 | weather_TA_AVG | 평균 기온(℃) | 0.0105 |
| 42 | weather_WS_MAX_TM | 최대 풍속 시각(HHMM) | 0.0104 |
| 43 | weather_SS_DAY | 일조시간 | 0.0102 |
| 44 | somae_amt_median_lag7 | 7일 전 소매가 중앙값 | 0.0102 |
| 45 | domae_amt_median_lag1 | 1일 전 도매가 중앙값 | 0.0098 |
| 46 | weather_TA_MIN_TM | 최저 기온 시각 | 0.0095 |
| 47 | somae_amt_mean_lag4 | 4일 전 소매가 평균 | 0.0094 |
| 48 | weather_PS_AVG | 평균 지면기압 | 0.0092 |
| 49 | somae_amt_std_lag3 | 3일 전 소매가 표준편차 | 0.0089 |
| 50 | somae_amt_mean_lag3 | 3일 전 소매가 평균 | 0.0086 |
| 51 | weather_EV_L | 대형증발량 | 0.0085 |
| 52 | somae_amt_mean_lag2 | 2일 전 소매가 평균 | 0.0084 |
| 53 | domae_amt_mean_lag2 | 2일 전 도매가 평균 | 0.0084 |
| 54 | domae_amt_std_lag6 | 6일 전 도매가 표준편차 | 0.0084 |
| 55 | weather_WS_AVG | 풍속 평균 (m/s) | 0.0083 |
| 56 | domae_amt_std_lag3 | 3일 전 도매가 표준편차 | 0.0082 |
| 57 | weather_WS_MAX | 최대 풍속 | 0.0082 |
| 58 | domae_amt_median_lag6 | 6일 전 도매가 중앙값 | 0.0082 |
| 59 | somae_amt_mean_lag6 | 6일 전 소매가 평균 | 0.0080 |
| 60 | domae_amt_std_lag4 | 4일 전 도매가 표준편차 | 0.0079 |
| 61 | weather_PA_AVG | 평균 해면기압 | 0.0078 |
| 62 | domae_amt_median_lag5 | 5일 전 도매가 중앙값 | 0.0078 |
| 63 | domae_amt_mean_lag3 | 3일 전 도매가 평균 | 0.0078 |
| 64 | somae_market_count_lag3 | 3일 전 소매 시장 수 | 0.0077 |
| 65 | domae_amt_median_lag4 | 4일 전 도매가 중앙값 | 0.0077 |
| 66 | domae_amt_median_lag3 | 3일 전 도매가 중앙값 | 0.0076 |
| 67 | somae_amt_std_lag2 | 2일 전 소매가 표준편차 | 0.0076 |
| 68 | domae_amt_median_lag2 | 2일 전 도매가 중앙값 | 0.0076 |
| 69 | somae_market_count_lag1 | 1일 전 소매 시장 수 | 0.0076 |
| 70 | domae_amt_median_lag7 | 7일 전 도매가 중앙값 | 0.0075 |
| 71 | weather_PS_MIN | 최저 지면기압 | 0.0073 |
| 72 | domae_amt_mean_lag6 | 6일 전 도매가 평균 | 0.0072 |
| 73 | somae_amt_median_lag5 | 5일 전 소매가 중앙값 | 0.0071 |
| 74 | somae_amt_median_lag6 | 6일 전 소매가 중앙값 | 0.0071 |
| 75 | somae_amt_std_lag5 | 5일 전 소매가 표준편차 | 0.0071 |
| 76 | somae_amt_median_lag4 | 4일 전 소매가 중앙값 | 0.0071 |
| 77 | domae_amt_std_lag5 | 5일 전 도매가 표준편차 | 0.0070 |
| 78 | weather_SI_60M_MAX_TM | 60분 최대 일사 시각 | 0.0069 |
| 79 | domae_amt_mean_lag5 | 5일 전 도매가 평균 | 0.0069 |
| 80 | domae_amt_mean_lag4 | 4일 전 도매가 평균 | 0.0066 |
| 81 | somae_amt_median_lag3 | 3일 전 소매가 중앙값 | 0.0063 |
| 82 | somae_amt_median_lag2 | 2일 전 소매가 중앙값 | 0.0063 |
| 83 | somae_market_count_lag7 | 7일 전 소매 시장 수 | 0.0063 |
| 84 | somae_market_count_lag5 | 5일 전 소매 시장 수 | 0.0063 |
| 85 | somae_market_count_lag6 | 6일 전 소매 시장 수 | 0.0063 |
| 86 | somae_amt_std_lag4 | 4일 전 소매가 표준편차 | 0.0062 |
| 87 | somae_amt_mean_lag5 | 5일 전 소매가 평균 | 0.0059 |
| 88 | somae_market_count_lag2 | 2일 전 소매 시장 수 | 0.0056 |
| 89 | somae_market_count_lag4 | 4일 전 소매 시장 수 | 0.0055 |
| 90 | weather_RN_D99 | 99% 일강수량 | 0.0052 |
| 91 | weather_RN_DAY | 일강수량(mm) | 0.0048 |
| 92 | weather_RN_DUR | 강수 지속시간 | 0.0041 |
| 93 | date_year | 연도 | 0.0028 |
| 94 | weather_RN_60M_MAX_TM | 60분 최대 강수 시각 | 0.0021 |
| 95 | weather_RN_60M_MAX | 60분 최대 강수량 | 0.0020 |
| 96 | weather_RN_10M_MAX | 10분 최대 강수량 | 0.0017 |
| 97 | weather_RN_10M_MAX_TM | 10분 최대 강수 시각 | 0.0013 |
| 98 | weather_FG_DUR | 안개 지속시간 | 0.0010 |
| 99 | weather_SD_MAX_TM | 적설 최대 시각 | 0.0010 |
| 100 | domae_market_count_lag3 | 3일 전 도매 시장 수 | 0.0009 |
| 101 | domae_market_count_lag7 | 7일 전 도매 시장 수 | 0.0005 |
| 102 | domae_market_count_lag1 | 1일 전 도매 시장 수 | 0.0005 |
| 103 | domae_market_count_lag5 | 5일 전 도매 시장 수 | 0.0004 |
| 104 | domae_market_count_lag4 | 4일 전 도매 시장 수 | 0.0004 |
| 105 | weather_SD_MAX | 적설 깊이 | 0.0004 |
| 106 | weather_SD_NEW | 신적설량 | 0.0004 |
| 107 | domae_market_count_lag2 | 2일 전 도매 시장 수 | 0.0003 |
| 108 | domae_market_count_lag6 | 6일 전 도매 시장 수 | 0.0003 |
| 109 | weather_SD_NEW_TM | 신적설 시각 | 0.0002 |

### 3.2 Mutual Information
| 순위 | 컬럼 | 설명 | 수치 |
|------|------|------|------|
| 1 | date_month | 월 (계절성/수확시기) | 0.1927 |
| 2 | weather_SS_DUR | 일조 지속시간 | 0.1557 |
| 3 | auction_quantity_sum | 해당일 품종별 경매 수량 합계 | 0.1545 |
| 4 | auction_transaction_count | 해당일 품종별 경매 거래 건수 | 0.1341 |
| 5 | domae_amt_std_lag7 | 7일 전 도매가 표준편차 | 0.1250 |
| 6 | weather_PV_AVG | 평균 현지기압(hPa) | 0.1194 |
| 7 | somae_amt_mean_lag7 | 7일 전 소매가 평균 | 0.1176 |
| 8 | domae_amt_median_lag7 | 7일 전 도매가 중앙값 | 0.1156 |
| 9 | domae_amt_std_lag3 | 3일 전 도매가 표준편차 | 0.1136 |
| 10 | domae_amt_mean_lag7 | 7일 전 도매가 평균 | 0.1058 |
| 11 | somae_amt_mean_lag1 | 1일 전 소매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.1027 |
| 12 | somae_amt_median_lag1 | 1일 전 소매가 중앙값 | 0.0998 |
| 13 | weather_HM_MIN | 최소 상대습도 | 0.0980 |
| 14 | weather_TD_AVG | 평균 이슬점온도 | 0.0933 |
| 15 | weather_HM_AVG | 평균 상대습도(%) | 0.0930 |
| 16 | weather_TA_MAX | 최고 기온 | 0.0921 |
| 17 | weather_WD_MAX | 최대 풍향 | 0.0916 |
| 18 | domae_amt_mean_lag6 | 6일 전 도매가 평균 | 0.0849 |
| 19 | weather_TA_AVG | 평균 기온(℃) | 0.0840 |
| 20 | weather_TS_AVG | 평균 지표면온도 | 0.0814 |
| 21 | weather_WS_MAX | 최대 풍속 | 0.0808 |
| 22 | domae_amt_median_lag4 | 4일 전 도매가 중앙값 | 0.0789 |
| 23 | auction_quantity_mean | 해당일 품종별 거래당 평균 수량 | 0.0753 |
| 24 | weather_TG_MIN | 최저 초상온도 | 0.0748 |
| 25 | somae_amt_mean_lag2 | 2일 전 소매가 평균 | 0.0745 |
| 26 | somae_amt_median_lag2 | 2일 전 소매가 중앙값 | 0.0720 |
| 27 | date_dayofweek | 요일 (0=월) | 0.0707 |
| 28 | domae_amt_median_lag3 | 3일 전 도매가 중앙값 | 0.0704 |
| 29 | domae_amt_median_lag6 | 6일 전 도매가 중앙값 | 0.0700 |
| 30 | domae_amt_mean_lag4 | 4일 전 도매가 평균 | 0.0699 |
| 31 | weather_WR_DAY | 풍향 일교차 | 0.0692 |
| 32 | domae_amt_std_lag6 | 6일 전 도매가 표준편차 | 0.0692 |
| 33 | domae_amt_std_lag1 | 1일 전 도매가 표준편차 | 0.0688 |
| 34 | weather_WD_INS | 순간 풍향 | 0.0679 |
| 35 | somae_market_count_lag1 | 1일 전 소매 시장 수 | 0.0677 |
| 36 | domae_amt_std_lag4 | 4일 전 도매가 표준편차 | 0.0661 |
| 37 | domae_amt_mean_lag3 | 3일 전 도매가 평균 | 0.0660 |
| 38 | domae_amt_std_lag2 | 2일 전 도매가 표준편차 | 0.0638 |
| 39 | domae_amt_median_lag1 | 1일 전 도매가 중앙값 | 0.0623 |
| 40 | somae_amt_std_lag7 | 7일 전 소매가 표준편차 | 0.0623 |
| 41 | somae_amt_median_lag7 | 7일 전 소매가 중앙값 | 0.0616 |
| 42 | weather_CA_TOT | 전운량 | 0.0606 |
| 43 | domae_amt_mean_lag1 | 1일 전 도매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0603 |
| 44 | somae_amt_mean_lag3 | 3일 전 소매가 평균 | 0.0584 |
| 45 | somae_market_count_lag2 | 2일 전 소매 시장 수 | 0.0572 |
| 46 | somae_amt_median_lag4 | 4일 전 소매가 중앙값 | 0.0554 |
| 47 | weather_TA_MIN | 최저 기온 | 0.0547 |
| 48 | somae_amt_mean_lag4 | 4일 전 소매가 평균 | 0.0525 |
| 49 | weather_PS_AVG | 평균 지면기압 | 0.0523 |
| 50 | somae_amt_std_lag4 | 4일 전 소매가 표준편차 | 0.0493 |
| 51 | domae_amt_median_lag2 | 2일 전 도매가 중앙값 | 0.0490 |
| 52 | somae_amt_median_lag3 | 3일 전 소매가 중앙값 | 0.0489 |
| 53 | domae_amt_median_lag5 | 5일 전 도매가 중앙값 | 0.0487 |
| 54 | weather_WS_AVG | 풍속 평균 (m/s) | 0.0483 |
| 55 | weather_PA_AVG | 평균 해면기압 | 0.0477 |
| 56 | weather_PS_MAX | 최고 지면기압 | 0.0471 |
| 57 | domae_amt_mean_lag2 | 2일 전 도매가 평균 | 0.0466 |
| 58 | weather_TA_MIN_TM | 최저 기온 시각 | 0.0466 |
| 59 | weather_EV_S | 소형증발량 | 0.0463 |
| 60 | weather_SI_60M_MAX | 60분 최대 일사량 | 0.0460 |
| 61 | somae_amt_std_lag3 | 3일 전 소매가 표준편차 | 0.0444 |
| 62 | weather_SI_DAY | 일사량 | 0.0425 |
| 63 | somae_amt_median_lag6 | 6일 전 소매가 중앙값 | 0.0423 |
| 64 | somae_amt_std_lag6 | 6일 전 소매가 표준편차 | 0.0403 |
| 65 | somae_amt_std_lag1 | 1일 전 소매가 표준편차 | 0.0399 |
| 66 | domae_amt_mean_lag5 | 5일 전 도매가 평균 | 0.0388 |
| 67 | weather_PS_MIN_TM | 최저 지면기압 시각 | 0.0379 |
| 68 | somae_amt_mean_lag5 | 5일 전 소매가 평균 | 0.0344 |
| 69 | domae_amt_std_lag5 | 5일 전 도매가 표준편차 | 0.0300 |
| 70 | weather_HM_MIN_TM | 최소 습도 시각 | 0.0254 |
| 71 | weather_EV_L | 대형증발량 | 0.0249 |
| 72 | somae_amt_mean_lag6 | 6일 전 소매가 평균 | 0.0243 |
| 73 | weather_RN_D99 | 99% 일강수량 | 0.0231 |
| 74 | somae_amt_std_lag2 | 2일 전 소매가 표준편차 | 0.0223 |
| 75 | weather_WS_INS_TM | 순간 풍속 시각 | 0.0204 |
| 76 | somae_amt_median_lag5 | 5일 전 소매가 중앙값 | 0.0192 |
| 77 | weather_SI_60M_MAX_TM | 60분 최대 일사 시각 | 0.0192 |
| 78 | weather_SS_DAY | 일조시간 | 0.0155 |
| 79 | somae_market_count_lag7 | 7일 전 소매 시장 수 | 0.0153 |
| 80 | somae_market_count_lag5 | 5일 전 소매 시장 수 | 0.0133 |
| 81 | somae_market_count_lag4 | 4일 전 소매 시장 수 | 0.0106 |
| 82 | weather_SD_MAX | 적설 깊이 | 0.0098 |
| 83 | weather_PS_MIN | 최저 지면기압 | 0.0090 |
| 84 | somae_market_count_lag6 | 6일 전 소매 시장 수 | 0.0089 |
| 85 | weather_SD_MAX_TM | 적설 최대 시각 | 0.0064 |
| 86 | weather_WS_MAX_TM | 최대 풍속 시각(HHMM) | 0.0057 |
| 87 | weather_PS_MAX_TM | 최고 지면기압 시각 | 0.0056 |
| 88 | weather_SD_NEW | 신적설량 | 0.0048 |
| 89 | weather_SD_NEW_TM | 신적설 시각 | 0.0034 |
| 90 | somae_market_count_lag3 | 3일 전 소매 시장 수 | 0.0020 |
| 91 | domae_market_count_lag2 | 2일 전 도매 시장 수 | 0.0013 |
| 92 | weather_WS_INS | 순간 풍속 | 0.0012 |
| 93 | domae_market_count_lag1 | 1일 전 도매 시장 수 | 0.0000 |
| 94 | domae_market_count_lag7 | 7일 전 도매 시장 수 | 0.0000 |
| 95 | domae_market_count_lag3 | 3일 전 도매 시장 수 | 0.0000 |
| 96 | domae_market_count_lag5 | 5일 전 도매 시장 수 | 0.0000 |
| 97 | domae_market_count_lag6 | 6일 전 도매 시장 수 | 0.0000 |
| 98 | domae_market_count_lag4 | 4일 전 도매 시장 수 | 0.0000 |
| 99 | somae_amt_std_lag5 | 5일 전 소매가 표준편차 | 0.0000 |
| 100 | weather_TA_MAX_TM | 최고 기온 시각 | 0.0000 |
| 101 | weather_FG_DUR | 안개 지속시간 | 0.0000 |
| 102 | weather_RN_10M_MAX_TM | 10분 최대 강수 시각 | 0.0000 |
| 103 | weather_RN_10M_MAX | 10분 최대 강수량 | 0.0000 |
| 104 | weather_RN_60M_MAX_TM | 60분 최대 강수 시각 | 0.0000 |
| 105 | weather_RN_60M_MAX | 60분 최대 강수량 | 0.0000 |
| 106 | weather_RN_DUR | 강수 지속시간 | 0.0000 |
| 107 | weather_RN_DAY | 일강수량(mm) | 0.0000 |
| 108 | date_year | 연도 | 0.0000 |
| 109 | date_day | 일 | 0.0000 |

### 3.3 Correlation (|r|)
| 순위 | 컬럼 | 설명 | 수치 |
|------|------|------|------|
| 1 | date_month | 월 (계절성/수확시기) | 0.2025 |
| 2 | auction_quantity_mean | 해당일 품종별 거래당 평균 수량 | 0.1732 |
| 3 | auction_quantity_sum | 해당일 품종별 경매 수량 합계 | 0.1658 |
| 4 | weather_TA_MIN | 최저 기온 | 0.1433 |
| 5 | weather_TA_AVG | 평균 기온(℃) | 0.1407 |
| 6 | weather_TG_MIN | 최저 초상온도 | 0.1383 |
| 7 | weather_TS_AVG | 평균 지표면온도 | 0.1357 |
| 8 | weather_TA_MAX | 최고 기온 | 0.1348 |
| 9 | domae_amt_std_lag3 | 3일 전 도매가 표준편차 | 0.1290 |
| 10 | weather_TD_AVG | 평균 이슬점온도 | 0.1289 |
| 11 | domae_amt_std_lag2 | 2일 전 도매가 표준편차 | 0.1246 |
| 12 | weather_PV_AVG | 평균 현지기압(hPa) | 0.1165 |
| 13 | weather_RN_D99 | 99% 일강수량 | 0.1136 |
| 14 | weather_RN_60M_MAX_TM | 60분 최대 강수 시각 | 0.1045 |
| 15 | weather_PS_MAX | 최고 지면기압 | 0.0993 |
| 16 | weather_RN_10M_MAX_TM | 10분 최대 강수 시각 | 0.0979 |
| 17 | domae_amt_std_lag1 | 1일 전 도매가 표준편차 | 0.0967 |
| 18 | weather_TA_MIN_TM | 최저 기온 시각 | 0.0943 |
| 19 | domae_amt_std_lag7 | 7일 전 도매가 표준편차 | 0.0940 |
| 20 | domae_amt_mean_lag1 | 1일 전 도매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0936 |
| 21 | domae_amt_mean_lag3 | 3일 전 도매가 평균 | 0.0925 |
| 22 | domae_amt_mean_lag2 | 2일 전 도매가 평균 | 0.0922 |
| 23 | domae_amt_std_lag5 | 5일 전 도매가 표준편차 | 0.0898 |
| 24 | somae_market_count_lag7 | 7일 전 소매 시장 수 | 0.0895 |
| 25 | weather_PS_AVG | 평균 지면기압 | 0.0894 |
| 26 | domae_amt_mean_lag5 | 5일 전 도매가 평균 | 0.0880 |
| 27 | weather_PS_MIN | 최저 지면기압 | 0.0861 |
| 28 | auction_transaction_count | 해당일 품종별 경매 거래 건수 | 0.0845 |
| 29 | domae_amt_median_lag2 | 2일 전 도매가 중앙값 | 0.0799 |
| 30 | domae_amt_median_lag1 | 1일 전 도매가 중앙값 | 0.0797 |
| 31 | date_year | 연도 | 0.0795 |
| 32 | domae_amt_std_lag4 | 4일 전 도매가 표준편차 | 0.0795 |
| 33 | weather_PA_AVG | 평균 해면기압 | 0.0779 |
| 34 | domae_amt_std_lag6 | 6일 전 도매가 표준편차 | 0.0779 |
| 35 | domae_amt_mean_lag7 | 7일 전 도매가 평균 | 0.0771 |
| 36 | domae_amt_median_lag5 | 5일 전 도매가 중앙값 | 0.0758 |
| 37 | domae_amt_mean_lag6 | 6일 전 도매가 평균 | 0.0730 |
| 38 | weather_RN_DUR | 강수 지속시간 | 0.0716 |
| 39 | weather_SS_DAY | 일조시간 | 0.0710 |
| 40 | domae_amt_median_lag3 | 3일 전 도매가 중앙값 | 0.0702 |
| 41 | weather_SD_NEW | 신적설량 | 0.0702 |
| 42 | domae_amt_median_lag6 | 6일 전 도매가 중앙값 | 0.0674 |
| 43 | somae_amt_std_lag3 | 3일 전 소매가 표준편차 | 0.0619 |
| 44 | somae_market_count_lag6 | 6일 전 소매 시장 수 | 0.0611 |
| 45 | weather_SI_60M_MAX | 60분 최대 일사량 | 0.0602 |
| 46 | weather_HM_AVG | 평균 상대습도(%) | 0.0597 |
| 47 | somae_market_count_lag3 | 3일 전 소매 시장 수 | 0.0575 |
| 48 | domae_market_count_lag1 | 1일 전 도매 시장 수 | 0.0573 |
| 49 | somae_amt_std_lag2 | 2일 전 소매가 표준편차 | 0.0572 |
| 50 | domae_amt_median_lag7 | 7일 전 도매가 중앙값 | 0.0569 |
| 51 | domae_market_count_lag7 | 7일 전 도매 시장 수 | 0.0566 |
| 52 | domae_amt_mean_lag4 | 4일 전 도매가 평균 | 0.0555 |
| 53 | somae_market_count_lag2 | 2일 전 소매 시장 수 | 0.0554 |
| 54 | somae_amt_std_lag1 | 1일 전 소매가 표준편차 | 0.0509 |
| 55 | weather_SD_MAX | 적설 깊이 | 0.0493 |
| 56 | somae_amt_mean_lag7 | 7일 전 소매가 평균 | 0.0470 |
| 57 | weather_EV_L | 대형증발량 | 0.0460 |
| 58 | somae_market_count_lag5 | 5일 전 소매 시장 수 | 0.0454 |
| 59 | weather_EV_S | 소형증발량 | 0.0454 |
| 60 | domae_market_count_lag4 | 4일 전 도매 시장 수 | 0.0446 |
| 61 | domae_amt_median_lag4 | 4일 전 도매가 중앙값 | 0.0433 |
| 62 | somae_market_count_lag1 | 1일 전 소매 시장 수 | 0.0424 |
| 63 | weather_WS_AVG | 풍속 평균 (m/s) | 0.0421 |
| 64 | domae_market_count_lag6 | 6일 전 도매 시장 수 | 0.0413 |
| 65 | weather_WR_DAY | 풍향 일교차 | 0.0410 |
| 66 | weather_SI_DAY | 일사량 | 0.0400 |
| 67 | somae_market_count_lag4 | 4일 전 소매 시장 수 | 0.0397 |
| 68 | somae_amt_std_lag7 | 7일 전 소매가 표준편차 | 0.0396 |
| 69 | weather_SI_60M_MAX_TM | 60분 최대 일사 시각 | 0.0375 |
| 70 | weather_CA_TOT | 전운량 | 0.0357 |
| 71 | date_day | 일 | 0.0343 |
| 72 | weather_WD_MAX | 최대 풍향 | 0.0322 |
| 73 | somae_amt_median_lag7 | 7일 전 소매가 중앙값 | 0.0314 |
| 74 | weather_WS_INS_TM | 순간 풍속 시각 | 0.0313 |
| 75 | somae_amt_std_lag6 | 6일 전 소매가 표준편차 | 0.0312 |
| 76 | date_dayofweek | 요일 (0=월) | 0.0303 |
| 77 | somae_amt_median_lag1 | 1일 전 소매가 중앙값 | 0.0283 |
| 78 | somae_amt_mean_lag6 | 6일 전 소매가 평균 | 0.0269 |
| 79 | weather_RN_10M_MAX | 10분 최대 강수량 | 0.0268 |
| 80 | somae_amt_std_lag5 | 5일 전 소매가 표준편차 | 0.0266 |
| 81 | weather_HM_MIN | 최소 상대습도 | 0.0259 |
| 82 | somae_amt_mean_lag5 | 5일 전 소매가 평균 | 0.0256 |
| 83 | weather_FG_DUR | 안개 지속시간 | 0.0249 |
| 84 | weather_WD_INS | 순간 풍향 | 0.0249 |
| 85 | weather_SS_DUR | 일조 지속시간 | 0.0229 |
| 86 | somae_amt_mean_lag4 | 4일 전 소매가 평균 | 0.0227 |
| 87 | weather_WS_MAX | 최대 풍속 | 0.0222 |
| 88 | somae_amt_std_lag4 | 4일 전 소매가 표준편차 | 0.0207 |
| 89 | somae_amt_median_lag3 | 3일 전 소매가 중앙값 | 0.0193 |
| 90 | domae_market_count_lag5 | 5일 전 도매 시장 수 | 0.0189 |
| 91 | weather_PS_MIN_TM | 최저 지면기압 시각 | 0.0176 |
| 92 | weather_RN_DAY | 일강수량(mm) | 0.0164 |
| 93 | somae_amt_median_lag2 | 2일 전 소매가 중앙값 | 0.0161 |
| 94 | weather_RN_60M_MAX | 60분 최대 강수량 | 0.0133 |
| 95 | somae_amt_median_lag4 | 4일 전 소매가 중앙값 | 0.0131 |
| 96 | weather_HM_MIN_TM | 최소 습도 시각 | 0.0121 |
| 97 | somae_amt_median_lag5 | 5일 전 소매가 중앙값 | 0.0105 |
| 98 | weather_SD_MAX_TM | 적설 최대 시각 | 0.0100 |
| 99 | somae_amt_mean_lag1 | 1일 전 소매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0088 |
| 100 | weather_WS_INS | 순간 풍속 | 0.0072 |
| 101 | weather_TA_MAX_TM | 최고 기온 시각 | 0.0052 |
| 102 | weather_SD_NEW_TM | 신적설 시각 | 0.0046 |
| 103 | somae_amt_mean_lag2 | 2일 전 소매가 평균 | 0.0045 |
| 104 | weather_WS_MAX_TM | 최대 풍속 시각(HHMM) | 0.0043 |
| 105 | domae_market_count_lag3 | 3일 전 도매 시장 수 | 0.0043 |
| 106 | domae_market_count_lag2 | 2일 전 도매 시장 수 | 0.0039 |
| 107 | somae_amt_median_lag6 | 6일 전 소매가 중앙값 | 0.0034 |
| 108 | weather_PS_MAX_TM | 최고 지면기압 시각 | 0.0019 |
| 109 | somae_amt_mean_lag3 | 3일 전 소매가 평균 | 0.0004 |

## 4. 통합 순위 (가중 평균)
가중치: Gini 40%, Mutual Info 35%, Correlation 25%

| 순위 | 컬럼 | 설명 | 통합점수 |
|------|------|------|----------|
| 1 | date_month | 월 (계절성/수확시기) | 0.6500 |
| 2 | auction_quantity_mean | 해당일 품종별 거래당 평균 수량 | 0.5402 |
| 3 | auction_quantity_sum | 해당일 품종별 경매 수량 합계 | 0.4000 |
| 4 | auction_transaction_count | 해당일 품종별 경매 거래 건수 | 0.2298 |
| 5 | weather_SS_DUR | 일조 지속시간 | 0.1897 |
| 6 | domae_amt_std_lag7 | 7일 전 도매가 표준편차 | 0.1832 |
| 7 | somae_amt_mean_lag1 | 1일 전 소매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.1143 |
| 8 | weather_TG_MIN | 최저 초상온도 | 0.1134 |
| 9 | weather_TA_MIN | 최저 기온 | 0.0935 |
| 10 | weather_PV_AVG | 평균 현지기압(hPa) | 0.0925 |
| 11 | weather_TA_AVG | 평균 기온(℃) | 0.0782 |
| 12 | weather_TA_MAX | 최고 기온 | 0.0781 |
| 13 | somae_amt_mean_lag7 | 7일 전 소매가 평균 | 0.0767 |
| 14 | domae_amt_std_lag3 | 3일 전 도매가 표준편차 | 0.0738 |
| 15 | date_day | 일 | 0.0734 |
| 16 | weather_TS_AVG | 평균 지표면온도 | 0.0714 |
| 17 | domae_amt_mean_lag7 | 7일 전 도매가 평균 | 0.0632 |
| 18 | domae_amt_std_lag1 | 1일 전 도매가 표준편차 | 0.0617 |
| 19 | weather_TD_AVG | 평균 이슬점온도 | 0.0600 |
| 20 | somae_amt_median_lag1 | 1일 전 소매가 중앙값 | 0.0591 |
| 21 | weather_HM_AVG | 평균 상대습도(%) | 0.0573 |
| 22 | weather_WD_MAX | 최대 풍향 | 0.0548 |
| 23 | domae_amt_median_lag7 | 7일 전 도매가 중앙값 | 0.0545 |
| 24 | somae_amt_std_lag1 | 1일 전 소매가 표준편차 | 0.0545 |
| 25 | date_dayofweek | 요일 (0=월) | 0.0496 |
| 26 | weather_HM_MIN | 최소 상대습도 | 0.0491 |
| 27 | weather_PS_MIN_TM | 최저 지면기압 시각 | 0.0480 |
| 28 | domae_amt_std_lag2 | 2일 전 도매가 표준편차 | 0.0479 |
| 29 | domae_amt_mean_lag1 | 1일 전 도매가 평균 (당일 경매 후 수집되므로 lag 사용) | 0.0349 |
| 30 | weather_PS_MAX | 최고 지면기압 | 0.0334 |
| 31 | domae_amt_mean_lag6 | 6일 전 도매가 평균 | 0.0318 |
| 32 | weather_CA_TOT | 전운량 | 0.0293 |
| 33 | weather_TA_MIN_TM | 최저 기온 시각 | 0.0286 |
| 34 | weather_RN_D99 | 99% 일강수량 | 0.0285 |
| 35 | domae_amt_mean_lag3 | 3일 전 도매가 평균 | 0.0277 |
| 36 | weather_WR_DAY | 풍향 일교차 | 0.0273 |
| 37 | weather_WS_MAX | 최대 풍속 | 0.0266 |
| 38 | weather_WS_INS | 순간 풍속 | 0.0263 |
| 39 | domae_amt_median_lag1 | 1일 전 도매가 중앙값 | 0.0262 |
| 40 | domae_amt_median_lag4 | 4일 전 도매가 중앙값 | 0.0262 |
| 41 | domae_amt_std_lag6 | 6일 전 도매가 표준편차 | 0.0257 |
| 42 | weather_RN_60M_MAX_TM | 60분 최대 강수 시각 | 0.0255 |
| 43 | weather_PS_AVG | 평균 지면기압 | 0.0255 |
| 44 | domae_amt_mean_lag2 | 2일 전 도매가 평균 | 0.0251 |
| 45 | domae_amt_median_lag6 | 6일 전 도매가 중앙값 | 0.0249 |
| 46 | domae_amt_median_lag3 | 3일 전 도매가 중앙값 | 0.0248 |
| 47 | weather_WS_INS_TM | 순간 풍속 시각 | 0.0247 |
| 48 | weather_WD_INS | 순간 풍향 | 0.0244 |
| 49 | weather_SI_60M_MAX | 60분 최대 일사량 | 0.0243 |
| 50 | domae_amt_std_lag4 | 4일 전 도매가 표준편차 | 0.0242 |
| 51 | somae_amt_mean_lag2 | 2일 전 소매가 평균 | 0.0241 |
| 52 | weather_SI_DAY | 일사량 | 0.0232 |
| 53 | weather_RN_10M_MAX_TM | 10분 최대 강수 시각 | 0.0232 |
| 54 | somae_amt_std_lag7 | 7일 전 소매가 표준편차 | 0.0227 |
| 55 | weather_HM_MIN_TM | 최소 습도 시각 | 0.0224 |
| 56 | domae_amt_mean_lag4 | 4일 전 도매가 평균 | 0.0215 |
| 57 | domae_amt_median_lag2 | 2일 전 도매가 중앙값 | 0.0214 |
| 58 | weather_TA_MAX_TM | 최고 기온 시각 | 0.0214 |
| 59 | domae_amt_std_lag5 | 5일 전 도매가 표준편차 | 0.0211 |
| 60 | somae_amt_median_lag7 | 7일 전 소매가 중앙값 | 0.0211 |
| 61 | somae_amt_median_lag2 | 2일 전 소매가 중앙값 | 0.0210 |
| 62 | weather_EV_S | 소형증발량 | 0.0210 |
| 63 | weather_PA_AVG | 평균 해면기압 | 0.0205 |
| 64 | somae_amt_std_lag6 | 6일 전 소매가 표준편차 | 0.0202 |
| 65 | weather_SS_DAY | 일조시간 | 0.0202 |
| 66 | domae_amt_median_lag5 | 5일 전 도매가 중앙값 | 0.0200 |
| 67 | domae_amt_mean_lag5 | 5일 전 도매가 평균 | 0.0200 |
| 68 | somae_market_count_lag1 | 1일 전 소매 시장 수 | 0.0198 |
| 69 | somae_amt_std_lag3 | 3일 전 소매가 표준편차 | 0.0197 |
| 70 | somae_market_count_lag7 | 7일 전 소매 시장 수 | 0.0197 |
| 71 | weather_PS_MIN | 최저 지면기압 | 0.0191 |
| 72 | weather_PS_MAX_TM | 최고 지면기압 시각 | 0.0188 |
| 73 | somae_amt_mean_lag4 | 4일 전 소매가 평균 | 0.0187 |
| 74 | somae_amt_mean_lag3 | 3일 전 소매가 평균 | 0.0182 |
| 75 | weather_WS_AVG | 풍속 평균 (m/s) | 0.0177 |
| 76 | weather_EV_L | 대형증발량 | 0.0172 |
| 77 | somae_market_count_lag2 | 2일 전 소매 시장 수 | 0.0170 |
| 78 | weather_WS_MAX_TM | 최대 풍속 시각(HHMM) | 0.0160 |
| 79 | somae_amt_std_lag2 | 2일 전 소매가 표준편차 | 0.0158 |
| 80 | date_year | 연도 | 0.0156 |
| 81 | somae_amt_median_lag4 | 4일 전 소매가 중앙값 | 0.0155 |
| 82 | somae_market_count_lag3 | 3일 전 소매 시장 수 | 0.0155 |
| 83 | somae_amt_mean_lag6 | 6일 전 소매가 평균 | 0.0148 |
| 84 | somae_market_count_lag6 | 6일 전 소매 시장 수 | 0.0146 |
| 85 | somae_amt_std_lag4 | 4일 전 소매가 표준편차 | 0.0145 |
| 86 | somae_amt_median_lag3 | 3일 전 소매가 중앙값 | 0.0145 |
| 87 | weather_RN_DUR | 강수 지속시간 | 0.0142 |
| 88 | weather_SD_NEW | 신적설량 | 0.0138 |
| 89 | somae_market_count_lag5 | 5일 전 소매 시장 수 | 0.0134 |
| 90 | somae_amt_median_lag6 | 6일 전 소매가 중앙값 | 0.0133 |
| 91 | weather_SI_60M_MAX_TM | 60분 최대 일사 시각 | 0.0133 |
| 92 | domae_market_count_lag1 | 1일 전 도매 시장 수 | 0.0129 |
| 93 | somae_amt_mean_lag5 | 5일 전 소매가 평균 | 0.0128 |
| 94 | somae_amt_median_lag5 | 5일 전 소매가 중앙값 | 0.0127 |
| 95 | weather_SD_MAX | 적설 깊이 | 0.0126 |
| 96 | domae_market_count_lag7 | 7일 전 도매 시장 수 | 0.0126 |
| 97 | somae_market_count_lag4 | 4일 전 소매 시장 수 | 0.0125 |
| 98 | somae_amt_std_lag5 | 5일 전 소매가 표준편차 | 0.0120 |
| 99 | domae_market_count_lag4 | 4일 전 도매 시장 수 | 0.0116 |
| 100 | domae_market_count_lag6 | 6일 전 도매 시장 수 | 0.0112 |
| 101 | weather_RN_10M_MAX | 10분 최대 강수량 | 0.0107 |
| 102 | weather_SD_MAX_TM | 적설 최대 시각 | 0.0107 |
| 103 | weather_FG_DUR | 안개 지속시간 | 0.0106 |
| 104 | weather_RN_DAY | 일강수량(mm) | 0.0104 |
| 105 | domae_market_count_lag5 | 5일 전 도매 시장 수 | 0.0103 |
| 106 | weather_RN_60M_MAX | 60분 최대 강수량 | 0.0102 |
| 107 | domae_market_count_lag3 | 3일 전 도매 시장 수 | 0.0101 |
| 108 | weather_SD_NEW_TM | 신적설 시각 | 0.0101 |
| 109 | domae_market_count_lag2 | 2일 전 도매 시장 수 | 0.0099 |

## 5. 권장 Feature 후보

위 통합 순위 전체를 확인하여 초기 feature set 선정. 품종별 importance 재실행 권장.
