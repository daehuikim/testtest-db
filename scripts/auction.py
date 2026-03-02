import io
import time
import random
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta


def get_url(
    date_string: str,
    whsal_cd: str,
    large_cd: str = "06",
    mid_cd: str = "01",
    page_size: str = "1000",
):
    return f"https://at.agromarket.kr/domeinfo/sanRealtime.do?pageNo=1&saledateBefore={date_string}&largeCdBefore=&midCdBefore=&smallCdBefore=&saledate={date_string}&whsalCd={whsal_cd}&cmpCd=&sanCd=&smallCdSearch=&largeCd={large_cd}&midCd={mid_cd}&smallCd=&pageSize={page_size}"


def get_empty_data(date_str: str):
    return {
        "경락일시": date_str,
        "도매시장": np.nan,
        "법인": np.nan,
        "부류": np.nan,
        "품목": np.nan,
        "품종": np.nan,
        "출하지": np.nan,
        "단량": np.nan,
        "수량": np.nan,
        "단량당 경락가(원)": np.nan,
    }


start_date_str = "20140818"
end_date_str = "20250918"
whsal_cd_list = [
    "110001",
    "110008",
    "210001",
    "210005",
    "210009",
    "220001",
    "230001",
    "230003",
    "240001",
    "240004",
    "250001",
    "250003",
    "310101",
    "310401",
    "310901",
    "311201",
    "320101",
    "320201",
    "320301",
    "330101",
    "330201",
    "340101",
    "350101",
    "350301",
    "350402",
    "360301",
    "370101",
    "370401",
    "371501",
    "380101",
    "380201",
    "380303",
    "380401",
]

start_date = date(
    int(start_date_str[:4]), int(start_date_str[4:6]), int(start_date_str[6:])
)
end_date = date(int(end_date_str[:4]), int(end_date_str[4:6]), int(end_date_str[6:]))

all_dfs = []
last_month = start_date.month

current_date = start_date
while current_date <= end_date:
    date_string = current_date.strftime("%Y-%m-%d")
    date_string_only_number = current_date.strftime("%Y%m%d")
    date_dfs = []

    if current_date.month > last_month:
        print(
            f"--- Processing month: {current_date.year} {current_date.month} ({date_string}) ---"
        )
        last_month = current_date.month

    for whsal_cd in whsal_cd_list:
        url = get_url(date_string, whsal_cd)

        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()

            dfs_from_html = pd.read_html(
                io.StringIO(res.text), thousands=",", encoding="utf-8"
            )

            if dfs_from_html and not dfs_from_html[0].empty:
                if "경매가" not in str(dfs_from_html[0].iloc[0, 0]):
                    df = dfs_from_html[0]
                    # '경락일시' 컬럼 추가
                    df["경락일시"] = date_string_only_number
                    date_dfs.append(df)
            else:
                print(f"Skipping {url}: No data table found.")

        except Exception as e:
            print(f"Error processing {url}: {e}")

    if len(date_dfs) == 0:
        date_dfs.append(pd.DataFrame([get_empty_data(date_string_only_number)]))
    all_dfs.extend(date_dfs)
    random_sec = random.uniform(1, 2)
    time.sleep(random_sec)
    current_date += timedelta(days=1)

# 모든 DataFrame을 하나로 합치기
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)

    # 최종 데이터프레임의 '경락일시'와 '도매시장' 컬럼이 맨 앞으로 오도록 순서 변경
    cols = ["경락일시", "도매시장"] + [
        col for col in final_df.columns if col not in ["경락일시", "도매시장"]
    ]
    final_df = final_df[cols]
    final_df["수량"] = final_df["수량"].astype(float)
    final_df["단량당 경락가(원)"] = final_df["단량당 경락가(원)"].astype(float)

    output_filename = f"./data/raw/auction/auction_{start_date_str}_{end_date_str}.csv"
    final_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print("\n데이터가 성공적으로 저장되었습니다.")
else:
    print("\n수집된 데이터가 없습니다.")
