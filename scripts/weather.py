import os
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

"""
#START7777
#--------------------------------------------------------------------------------------------------
#  기상청 지상관측 일자료 [입력인수형태][예] ?tm=20100715&stn=0&help=1
#--------------------------------------------------------------------------------------------------
#  1. TM            : 관측일 (KST)
#  2. STN           : 국내 지점번호
#  3. WS_AVG        : 일 평균 풍속 (m/s)
#  4. WR_DAY        : 일 풍정 (m)
#  5. WD_MAX        : 최대풍향
#  6. WS_MAX        : 최대풍속 (m/s)
#  7. WS_MAX_TM     : 최대풍속 시각 (시분)
#  8. WD_INS        : 최대순간풍향
#  9. WS_INS        : 최대순간풍속 (m/s)
# 10. WS_INS_TM     : 최대순간풍속 시각 (시분)
# 11. TA_AVG        : 일 평균기온 (C)
# 12. TA_MAX        : 최고기온 (C)
# 13. TA_MAX_TM     : 최고기온 시가 (시분)
# 14. TA_MIN        : 최저기온 (C)
# 15. TA_MIN_TM     : 최저기온 시각 (시분)
# 16. TD_AVG        : 일 평균 이슬점온도 (C)
# 17. TS_AVG        : 일 평균 지면온도 (C)
# 18. TG_MIN        : 일 최저 초상온도 (C)
# 19. HM_AVG        : 일 평균 상대습도 (%)
# 20. HM_MIN        : 최저습도 (%)
# 21. HM_MIN_TM     : 최저습도 시각 (시분)
# 22. PV_AVG        : 일 평균 수증기압 (hPa)
# 23. EV_S          : 소형 증발량 (mm)
# 24. EV_L          : 대형 증발량 (mm)
# 25. FG_DUR        : 안개계속시간 (hr)
# 26. PA_AVG        : 일 평균 현지기압 (hPa)
# 27. PS_AVG        : 일 평균 해면기압 (hPa)
# 28. PS_MAX        : 최고 해면기압 (hPa)
# 29. PS_MAX_TM     : 최고 해면기압 시각 (시분)
# 30. PS_MIN        : 최저 해면기압 (hPa)
# 31. PS_MIN_TM     : 최저 해면기압 시각 (시분)
# 32. CA_TOT        : 일 평균 전운량 (1/10)
# 33. SS_DAY        : 일조합 (hr)
# 34. SS_DUR        : 가조시간 (hr)
# 35. SS_CMB        : 캄벨 일조 (hr)
# 36. SI_DAY        : 일사합 (MJ/m2)
# 37. SI_60M_MAX    : 최대 1시간일사 (MJ/m2)
# 38. SI_60M_MAX_TM : 최대 1시간일사 시각 (시분)
# 39. RN_DAY        : 일 강수량 (mm)
# 40. RN_D99        : 9-9 강수량 (mm)
# 41. RN_DUR        : 강수계속시간 (hr)
# 42. RN_60M_MAX    : 1시간 최다강수량 (mm)
# 43. RN_60M_MAX_TM : 1시간 최다강수량 시각 (시분)
# 44. RN_10M_MAX    : 10분간 최다강수량 (mm)
# 45. RN_10M_MAX_TM : 10분간 최다강수량 시각 (시분)
# 46. RN_POW_MAX    : 최대 강우강도 (mm/h)
# 47. RN_POW_MAX_TM : 최대 강우강도 시각 (시분)
# 48. SD_NEW        : 최심 신적설 (cm)
# 49. SD_NEW_TM     : 최심 신적설 시각 (시분)
# 50. SD_MAX        : 최심 적설 (cm)
# 51. SD_MAX_TM     : 최심 적설 시각 (시분)
# 52. TE_05         : 0.5m 지중온도 (C) 
# 53. TE_10         : 1.0m 지중온도 (C)
# 54. TE_15         : 1.5m 지중온도 (C)
# 55. TE_30         : 3.0m 지중온도 (C)
# 56. TE_50         : 5.0m 지중온도 (C)
"""

columns = [
    "TM",
    "STN",
    "WS_AVG",
    "WR_DAY",
    "WD_MAX",
    "WS_MAX",
    "WS_MAX_TM",
    "WD_INS",
    "WS_INS",
    "WS_INS_TM",
    "TA_AVG",
    "TA_MAX",
    "TA_MAX_TM",
    "TA_MIN",
    "TA_MIN_TM",
    "TD_AVG",
    "TS_AVG",
    "TG_MIN",
    "HM_AVG",
    "HM_MIN",
    "HM_MIN_TM",
    "PV_AVG",
    "EV_S",
    "EV_L",
    "FG_DUR",
    "PA_AVG",
    "PS_AVG",
    "PS_MAX",
    "PS_MAX_TM",
    "PS_MIN",
    "PS_MIN_TM",
    "CA_TOT",
    "SS_DAY",
    "SS_DUR",
    "SS_CMB",
    "SI_DAY",
    "SI_60M_MAX",
    "SI_60M_MAX_TM",
    "RN_DAY",
    "RN_D99",
    "RN_DUR",
    "RN_60M_MAX",
    "RN_60M_MAX_TM",
    "RN_10M_MAX",
    "RN_10M_MAX_TM",
    "RN_POW_MAX",
    "RN_POW_MAX_TM",
    "SD_NEW",
    "SD_NEW_TM",
    "SD_MAX",
    "SD_MAX_TM",
    "TE_05",
    "TE_10",
    "TE_15",
    "TE_30",
    "TE_50",
]

output_dir = "./data/raw/weather"

auth_key = os.getenv("WEATHER_AUTH_KEY")
base_url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?authKey={auth_key}"

start_date_str = "20260113"
end_date_str = "20260114"
stations = [
    (136, "안동"),
    (276, "청송"),
    (272, "영주"),
]

stn_param = ":".join(str(stn[0]) for stn in stations)
url = base_url + f"&tm1={start_date_str}" + f"&tm2={end_date_str}" + f"&stn={stn_param}"

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    lines = response.text.strip().split("\n")
    data_lines = [line for line in lines if not line.startswith("#")]

    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=columns,
    )

    # 결측값(-9, -9.0, -99.0 등) 처리
    df = df.replace([-9, -9.0, -99.0], np.nan)

    os.makedirs(output_dir, exist_ok=True)

    for stn_id, stn_name in stations:
        stn_df = df[df["STN"] == stn_id].sort_values("TM").reset_index(drop=True)
        if not stn_df.empty:
            filename = f"weather_{start_date_str}_{end_date_str}_{stn_id}.csv"
            filepath = os.path.join(output_dir, filename)
            stn_df.to_csv(filepath, index=False)
            print(f"저장 완료: {filepath} ({len(stn_df)}행)")

except requests.RequestException as e:
    print(f"Error fetching weather data: {e}")
