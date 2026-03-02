"""
도매(domae) 데이터 수집기 - 병렬 처리
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from .base import BaseCollector

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

load_dotenv()

logger = logging.getLogger(__name__)

# .env 호환: WHOLESALE_API_*, PRODUCT_CODE, API_KEY
WHOLESALE_API_CODE = "Grid_20150406000000000217_1"
START_INDEX = 1
END_INDEX = 1000
FRMPRD_CATGORY_CD = "400"


def _get_empty_data(examin_de: str) -> Dict[str, Any]:
    return {
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


def _fetch_single_date(
    examin_de: str,
    base_url: str,
    grid_key: str,
    prdlst_cd: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """단일 날짜 도매 데이터 조회"""
    url = (
        f"{base_url}{START_INDEX}/{END_INDEX}"
        f"?EXAMIN_DE={examin_de}"
        f"&FRMPRD_CATGORY_CD={FRMPRD_CATGORY_CD}"
        f"&PRDLST_CD={prdlst_cd}"
    )
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        if grid_key in data and "row" in data[grid_key]:
            rows = data[grid_key]["row"]
            if rows:
                return rows
            return []  # 데이터 없으면 빈 리스트 (빈 행 추가 안 함)
        if grid_key in data and data[grid_key].get("totalCnt", 0) > END_INDEX:
            logger.warning("totalCnt > END_INDEX for %s", examin_de)
    except Exception as e:
        logger.debug("Error %s: %s", examin_de, e)
    return []


class DomaeCollector(BaseCollector):
    """도매 데이터 수집기 - 날짜 배치 병렬 처리"""

    def __init__(
        self,
        start_date: date,
        end_date: date,
        output_dir: str = "./data/raw/domae",
        max_workers: int = 10,
        seed: int = 42,
    ):
        super().__init__(start_date, end_date, output_dir, seed)
        self.max_workers = max_workers
        api_key = os.getenv("API_KEY")
        base = "http://211.237.50.150:7080/openapi"
        code = WHOLESALE_API_CODE
        if not api_key:
            raise ValueError("API_KEY 환경변수가 필요합니다.")
        self.base_url = f"{base.rstrip('/')}/{api_key}/json/{code}/"
        self.grid_key = code
        self.prdlst_cd = os.getenv("PRODUCT_CODE", "411")  # 411=사과

    def collect(self) -> pd.DataFrame:
        dates = [
            d.strftime("%Y%m%d")
            for d in self.date_range(self.start_date, self.end_date)
        ]
        all_data = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_fetch_single_date, d, self.base_url, self.grid_key, self.prdlst_cd): d
                for d in dates
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="domae", unit="day", unit_scale=False):
                rows = future.result()
                all_data.extend(rows)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if "ROW_NUM" in df.columns:
            df = df.drop(columns=["ROW_NUM"])
        return df
