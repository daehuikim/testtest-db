"""
경매(auction) 데이터 수집기 - 병렬 처리
"""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .base import BaseCollector

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

# 가락시장 (서울) - whsal_cd
GARAK_MARKET_CD = "110001"
WHSAL_CD_LIST = [GARAK_MARKET_CD]  # 가락시장만 수집


def _get_url(
    date_string: str,
    whsal_cd: str,
    large_cd: str = "06",
    mid_cd: str = "01",
    page_size: str = "1000",
) -> str:
    return (
        f"https://at.agromarket.kr/domeinfo/sanRealtime.do?"
        f"pageNo=1&saledateBefore={date_string}&largeCdBefore=&midCdBefore=&smallCdBefore=&"
        f"saledate={date_string}&whsalCd={whsal_cd}&cmpCd=&sanCd=&smallCdSearch=&"
        f"largeCd={large_cd}&midCd={mid_cd}&smallCd=&pageSize={page_size}"
    )


def _get_empty_data(date_str: str) -> dict:
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


def _fetch_single_market(
    date_string: str,
    date_string_num: str,
    whsal_cd: str,
    timeout: int = 30,
) -> Optional[pd.DataFrame]:
    """단일 시장·날짜 데이터 조회"""
    url = _get_url(date_string, whsal_cd)
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        dfs = pd.read_html(io.StringIO(res.text), thousands=",", encoding="utf-8")
        if dfs and not dfs[0].empty and "경매가" not in str(dfs[0].iloc[0, 0]):
            df = dfs[0].copy()
            df["경락일시"] = date_string_num
            return df
    except Exception as e:
        logger.debug("Error %s: %s", url, e)
    return None


def _fetch_single_date(
    current_date: date,
    whsal_cd_list: List[str],
    max_workers: int = 15,
    min_sleep: float = 0.2,
) -> List[pd.DataFrame]:
    """한 날짜에 대해 모든 시장 병렬 조회"""
    date_string = current_date.strftime("%Y-%m-%d")
    date_string_num = current_date.strftime("%Y%m%d")
    date_dfs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_single_market, date_string, date_string_num, whsal_cd
            ): whsal_cd
            for whsal_cd in whsal_cd_list
        }
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                date_dfs.append(df)
            time.sleep(min_sleep)  # rate limit 완화

    return date_dfs


class AuctionCollector(BaseCollector):
    """경매 데이터 수집기 - 날짜별 시장 병렬 처리"""

    def __init__(
        self,
        start_date: date,
        end_date: date,
        output_dir: str = "./data/raw/auction",
        whsal_cd_list: Optional[List[str]] = None,
        max_workers_per_date: int = 15,
        min_sleep: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(start_date, end_date, output_dir, seed)
        self.whsal_cd_list = whsal_cd_list or WHSAL_CD_LIST
        self.max_workers_per_date = max_workers_per_date
        self.min_sleep = min_sleep

    def collect(self) -> pd.DataFrame:
        all_dfs = []
        total_days = (self.end_date - self.start_date).days + 1

        for current_date in tqdm(
            self.date_range(self.start_date, self.end_date),
            total=total_days,
            desc="auction (가락시장)",
            unit="day",
            unit_scale=False,
        ):
            date_dfs = _fetch_single_date(
                current_date,
                self.whsal_cd_list,
                max_workers=self.max_workers_per_date,
                min_sleep=self.min_sleep,
            )
            if not date_dfs:
                date_dfs.append(
                    pd.DataFrame([_get_empty_data(current_date.strftime("%Y%m%d"))])
                )
            all_dfs.extend(date_dfs)

        if not all_dfs:
            return pd.DataFrame()

        final_df = pd.concat(all_dfs, ignore_index=True)
        cols = ["경락일시", "도매시장"] + [
            c for c in final_df.columns if c not in ("경락일시", "도매시장")
        ]
        final_df = final_df[cols]
        final_df["수량"] = pd.to_numeric(final_df["수량"], errors="coerce")
        final_df["단량당 경락가(원)"] = pd.to_numeric(
            final_df["단량당 경락가(원)"], errors="coerce"
        )
        return final_df
