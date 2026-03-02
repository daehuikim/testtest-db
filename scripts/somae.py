import os
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

"""
Input Parameters
항목명	타입	샘플데이터	항목설명
API_KEY	기본	sample	발급받은 API_KEY
TYPE	기본	xml	요청파일 타입 xml, json
API_URL	기본	Grid_20141225000000000163_1	OpenAPI 서비스 URL
START_INDEX	기본	1	요청시작위치
END_INDEX	기본	5	요청종료위치
EXAMIN_DE	필수	20150401	조사일
FRMPRD_CATGORY_CD	선택	100	농산물 부류 코드
PRDLST_CD	선택	151	품목 코드
SPCIES_CD	선택	00	품종 코드
GRAD_CD	선택	04	등급 코드
AREA_CD	선택	1101	지역 코드
MRKT_CD	선택	0110402	시장 코드
"""

"""
Output Parameters
항목명	항목설명
EXAMIN_DE	가격조사 일자
FRMPRD_CATGORY_NM	농산물 부류명
FRMPRD_CATGORY_CD	부류코드 [ 100(식량작물), 200(채소류), 300(특용작물), 400(과일류), 500(축산물), 600(수산물) ]
PRDLST_CD	품목코드
PRDLST_NM	품목명
SPCIES_CD	품종코드 [ 01(일반계) …..
SPCIES_NM	품종명
GRAD_CD	등급코드 [ 01(상품) …..
GRAD_NM	등급명
EXAMIN_UNIT	조사단위 ( Kg, g, ..)
AREA_CD	시도별 구분
AREA_NM	지역명
MRKT_CD	시도의 시장
MRKT_NM	시장명
AMT	조사 가격(원)
"""


def get_empty_data(examin_de: str):
    return {
        "ROW_NUM": np.nan,
        "EXAMIN_DE": examin_de,
        "FRMPRD_CATGORY_NM": np.nan,
        "FRMPRD_CATGORY_CD": np.nan,
        "PRDLST_CD": np.nan,
        "PRDLST_NM": np.nan,
        "SPCIES_CD": np.nan,
        "SPCIES_NM": np.nan,
        "GRAD_CD": np.nan,
        "GRAD_NM": np.nan,
        "EXAMIN_UNIT": np.nan,
        "AREA_CD": np.nan,
        "AREA_NM": np.nan,
        "MRKT_CD": np.nan,
        "MRKT_NM": np.nan,
        "AMT": np.nan,
    }


api_key = os.getenv("API_KEY")
base_url = (
    f"http://211.237.50.150:7080/openapi/{api_key}/json/Grid_20141225000000000163_1/"
)

START_INDEX = 1
END_INDEX = 1000
FRMPRD_CATGORY_CD = "400"
PRDLST_CD = "411"

start_date_str = "20140818"
end_date_str = "20250918"

start_date = date(
    int(start_date_str[:4]), int(start_date_str[4:6]), int(start_date_str[6:])
)
end_date = date(int(end_date_str[:4]), int(end_date_str[4:6]), int(end_date_str[6:]))

all_data = []

current_date = start_date
last_year = current_date.year
while current_date <= end_date:
    examin_de = current_date.strftime("%Y%m%d")

    if current_date.year > last_year:
        print(f"Fetching data for year: {current_date.year} (at {examin_de})")
        last_year = current_date.year

    try:
        response = requests.get(
            base_url
            + f"{START_INDEX}/{END_INDEX}"
            + f"?EXAMIN_DE={examin_de}"
            + f"&FRMPRD_CATGORY_CD={FRMPRD_CATGORY_CD}"
            + f"&PRDLST_CD={PRDLST_CD}"
        )
        response.raise_for_status()

        data = response.json()

        if (
            "Grid_20141225000000000163_1" in data
            and "row" in data["Grid_20141225000000000163_1"]
        ):
            rows = data["Grid_20141225000000000163_1"]["row"]
            if len(rows) > 0:
                all_data.extend(rows)
            else:
                empty_row = get_empty_data(examin_de)
                all_data.append(empty_row)
            if data["Grid_20141225000000000163_1"]["totalCnt"] > END_INDEX:
                print(
                    f"Total count is greater than end index for {examin_de}: {data['Grid_20141225000000000163_1']['totalCnt']}"
                )
        else:
            print(f"No 'row' data found for {examin_de}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {examin_de}: {e}")
        break
    except ValueError as e:
        print(f"Error decoding JSON for {examin_de}: {e}")
        break

    current_date += timedelta(days=1)

if all_data:
    df = pd.DataFrame(all_data)
    df.drop("ROW_NUM", axis=1, inplace=True)

    # DataFrame을 CSV 파일로 저장
    output_filename = f"./data/raw/somae/somae_{start_date_str}_{end_date_str}.csv"
    df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"\nData successfully saved to {output_filename}")
else:
    print("\nNo data was collected. CSV file was not created.")
