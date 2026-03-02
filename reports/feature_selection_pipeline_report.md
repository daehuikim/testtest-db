# Feature Selection Pipeline 결과

## 품종 (공통 feature 적용)
감홍, 기타, 로얄부사, 미시마, 미야비, 미얀마, 시나노골드, 시나노스위트, 아오리, 알프스오토메, 양광, 홍로, 후지

## Stage별 감소
| Stage | 유지 feature 수 |
|-------|----------------|
| 0 (Maximal lag) | 1470 |
| 1 (CCF+MI) | 976 |
| 2 (Elastic Net) | 488 |
| 3 (Rolling Perm) | 244 |
| 4 (Stability) | 104 |
| 5 (Common) | 94 |

## 최종 Feature Set

1. `auction_quantity_mean`
2. `auction_quantity_sum`
3. `auction_transaction_count`
4. `price_per_kg_mean_lag1`
5. `price_per_kg_mean_lag10`
6. `price_per_kg_mean_lag11`
7. `price_per_kg_mean_lag13`
8. `price_per_kg_mean_lag14`
9. `price_per_kg_mean_lag16`
10. `price_per_kg_mean_lag2`
11. `price_per_kg_mean_lag20`
12. `price_per_kg_mean_lag21`
13. `price_per_kg_mean_lag26`
14. `price_per_kg_mean_lag27`
15. `price_per_kg_mean_lag28`
16. `price_per_kg_mean_lag29`
17. `price_per_kg_mean_lag3`
18. `price_per_kg_mean_lag31`
19. `price_per_kg_mean_lag32`
20. `price_per_kg_mean_lag33`
21. `price_per_kg_mean_lag35`
22. `price_per_kg_mean_lag38`
23. `price_per_kg_mean_lag4`
24. `price_per_kg_mean_lag42`
25. `price_per_kg_mean_lag44`
26. `price_per_kg_mean_lag49`
27. `price_per_kg_mean_lag54`
28. `price_per_kg_mean_lag55`
29. `price_per_kg_mean_lag6`
30. `price_per_kg_mean_lag7`
31. `price_per_kg_mean_lag9`
32. `weather_CA_TOT_lag6`
33. `weather_EV_L_lag13`
34. `weather_EV_L_lag5`
35. `weather_FG_DUR_lag1`
36. `weather_HM_AVG_lag12`
37. `weather_HM_AVG_lag19`
38. `weather_HM_MIN_TM_lag20`
39. `weather_HM_MIN_TM_lag3`
40. `weather_HM_MIN_TM_lag9`
41. `weather_HM_MIN_lag11`
42. `weather_HM_MIN_lag5`
43. `weather_HM_MIN_lag7`
44. `weather_PA_AVG_lag2`
45. `weather_PS_AVG_lag18`
46. `weather_PS_MAX_TM_lag1`
47. `weather_PS_MAX_TM_lag5`
48. `weather_PS_MAX_TM_lag7`
49. `weather_PS_MAX_lag20`
50. `weather_PS_MIN_TM_lag10`
51. `weather_PS_MIN_TM_lag6`
52. `weather_PS_MIN_lag11`
53. `weather_RN_60M_MAX_lag10`
54. `weather_RN_DAY_lag15`
55. `weather_SI_60M_MAX_TM_lag9`
56. `weather_SI_60M_MAX_lag1`
57. `weather_SI_60M_MAX_lag8`
58. `weather_SI_DAY_lag12`
59. `weather_SS_DAY_lag11`
60. `weather_SS_DAY_lag16`
61. `weather_SS_DAY_lag18`
62. `weather_SS_DAY_lag7`
63. `weather_SS_DUR_lag10`
64. `weather_TA_AVG_lag21`
65. `weather_TA_MAX_TM_lag12`
66. `weather_TA_MAX_lag12`
67. `weather_TA_MAX_lag20`
68. `weather_TA_MAX_lag6`
69. `weather_TA_MIN_TM_lag1`
70. `weather_TA_MIN_TM_lag2`
71. `weather_TA_MIN_TM_lag20`
72. `weather_TA_MIN_TM_lag5`
73. `weather_TA_MIN_lag13`
74. `weather_TD_AVG_lag15`
75. `weather_TG_MIN_lag10`
76. `weather_TM_lag20`
77. `weather_TM_lag9`
78. `weather_TS_AVG_lag1`
79. `weather_TS_AVG_lag13`
80. `weather_WD_INS_lag19`
81. `weather_WD_INS_lag2`
82. `weather_WD_INS_lag20`
83. `weather_WD_INS_lag9`
84. `weather_WD_MAX_lag1`
85. `weather_WR_DAY_lag3`
86. `weather_WR_DAY_lag9`
87. `weather_WS_INS_TM_lag13`
88. `weather_WS_INS_TM_lag4`
89. `weather_WS_INS_TM_lag8`
90. `weather_WS_INS_lag12`
91. `weather_WS_INS_lag2`
92. `weather_WS_MAX_TM_lag17`
93. `weather_WS_MAX_TM_lag7`
94. `weather_WS_MAX_lag14`