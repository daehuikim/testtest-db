"""
기상(weather) 데이터 수집기 - kma_sfcdd3 (지상관측 일자료)
문서: 기상청 API 허브 > 지상관측 > 지상관측데이터 일통계
사과 생산지: 안동(136), 청송(276), 영주(272)
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import StringIO
from typing import List, Tuple

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

COLUMNS = [
    "TM", "STN", "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM",
    "WD_INS", "WS_INS", "WS_INS_TM", "TA_AVG", "TA_MAX", "TA_MAX_TM",
    "TA_MIN", "TA_MIN_TM", "TD_AVG", "TS_AVG", "TG_MIN", "HM_AVG",
    "HM_MIN", "HM_MIN_TM", "PV_AVG", "EV_S", "EV_L", "FG_DUR",
    "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
    "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB", "SI_DAY", "SI_60M_MAX",
    "SI_60M_MAX_TM", "RN_DAY", "RN_D99", "RN_DUR", "RN_60M_MAX",
    "RN_60M_MAX_TM", "RN_10M_MAX", "RN_10M_MAX_TM", "RN_POW_MAX",
    "RN_POW_MAX_TM", "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
    "TE_05", "TE_10", "TE_15", "TE_30", "TE_50",
]

# 사과 생산지: 안동(136), 청송(276), 영주(272)
DEFAULT_STATIONS = [(136, "안동"), (276, "청송"), (272, "영주")]


def _fetch_weather_range(
    start_str: str,
    end_str: str,
    stn_param: str,
    base_url: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """날짜 범위 기상 데이터 조회 (kma_sfcdd3)"""
    url = f"{base_url}&tm1={start_str}&tm2={end_str}&stn={stn_param}"
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        lines = res.text.strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        if not data_lines:
            return pd.DataFrame()
        df = pd.read_csv(
            StringIO("\n".join(data_lines)),
            sep=r"\s+",
            header=None,
            names=COLUMNS,
        )
        df = df.replace([-9, -9.0, -99.0], np.nan)
        return df
    except Exception as e:
        logger.debug("Error %s-%s: %s", start_str, end_str, e)
        return pd.DataFrame()


def _month_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    """월 단위 날짜 범위 생성"""
    ranges = []
    current = start.replace(day=1)
    while current <= end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(
            day=1
        ) - timedelta(days=1)
        actual_end = min(month_end, end)
        actual_start = max(current, start)
        if actual_start <= actual_end:
            ranges.append((actual_start, actual_end))
        if month_end >= end:
            break
        current = month_end + timedelta(days=1)
    return ranges


class WeatherCollector(BaseCollector):
    """기상 데이터 수집기 - kma_sfcdd3 (월별 병렬)"""

    def __init__(
        self,
        start_date: date,
        end_date: date,
        output_dir: str = "./data/raw/weather",
        stations: List[Tuple[int, str]] = None,
        max_workers: int = 6,
        seed: int = 42,
    ):
        super().__init__(start_date, end_date, output_dir, seed)
        self.stations = stations or DEFAULT_STATIONS
        self.max_workers = max_workers
        auth_key = os.getenv("WEATHER_API_KEY") or os.getenv("WEATHER_AUTH_KEY")
        if not auth_key:
            raise ValueError("WEATHER_API_KEY 또는 WEATHER_AUTH_KEY 환경변수가 필요합니다.")
        self.base_url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?authKey={auth_key}"
        self.stn_param = ":".join(str(s[0]) for s in self.stations)

    def collect(self) -> pd.DataFrame:
        ranges = _month_ranges(self.start_date, self.end_date)
        all_dfs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_weather_range,
                    s.strftime("%Y%m%d"),
                    e.strftime("%Y%m%d"),
                    self.stn_param,
                    self.base_url,
                ): (s, e)
                for s, e in ranges
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="weather",
                unit="month",
                unit_scale=False,
            ):
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
