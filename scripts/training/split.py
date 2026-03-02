#!/usr/bin/env python3
"""
데이터 분할: 전체 데이터에서 test ~70일만 제외, 나머지 전부 train.

- Train: 전체 731일 중 test ~70일 제외한 나머지 전부
- Test: 전체 unique date에서 계절 균형 랜덤 ~70일
- seed 고정 시 split 고정 → 모델별로 동일한 split으로 비교 가능
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 봄(3-5), 여름(6-8), 가을(9-11), 겨울(12-2)
SEASON_MAP = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}


def _month_to_season(month: int) -> str:
    for name, months in SEASON_MAP.items():
        if month in months:
            return name
    return "unknown"


def stratified_test_dates(
    all_dates: pd.Series,
    n_days: int = 70,
    seed: int = 42,
) -> List[str]:
    """
    전체 unique date에서 계절 균형 있게 n_days개 선정.
    봄/여름/가을/겨울 각각 비슷한 비율로 샘플링.
    seed 고정 시 동일 split 보장.
    """
    dates = pd.to_datetime(all_dates.astype(str), format="%Y%m%d", errors="coerce")
    valid = dates.notna()
    dates = dates[valid].dt.strftime("%Y%m%d").unique().tolist()
    if not dates:
        return []

    df = pd.DataFrame({"date": dates})
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["month"] = df["dt"].dt.month
    df["season"] = df["month"].apply(_month_to_season)

    rng = np.random.default_rng(seed)
    seasons = df["season"].unique().tolist()
    n_per_season = max(1, n_days // len(seasons))
    remainder = n_days - n_per_season * len(seasons)

    selected = []
    for i, s in enumerate(seasons):
        pool = df[df["season"] == s]["date"].tolist()
        n = n_per_season + (1 if i < remainder else 0)
        n = min(n, len(pool))
        if n > 0 and pool:
            chosen = rng.choice(pool, size=n, replace=False)
            selected.extend(chosen.tolist() if hasattr(chosen, "tolist") else [chosen])

    return sorted(selected)


def train_test_split(
    df: pd.DataFrame,
    test_days: int = 70,
    test_ratio: Optional[float] = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Train: 전체 데이터에서 test 날짜만 제외한 나머지 전부
    Test: 전체 unique date에서 계절 균형 랜덤 ~test_days (seed 고정 시 split 고정)

    Returns: (train_df, test_df, test_date_list)
    """
    df = df.copy()
    df["date"] = df["date"].astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)

    all_dates = df["date"].dropna().unique()
    n_total = len(all_dates)
    n_test = test_days if test_days > 0 else max(1, int(n_total * (test_ratio or 0.1)))
    n_test = min(n_test, n_total)


    test_date_list = stratified_test_dates(pd.Series(all_dates), n_days=n_test, seed=seed)
    test_mask = df["date"].isin(test_date_list)
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        "Split (seed=%d): 전체 %d일 중 train %d rows (%d일), test %d rows (%d일, 계절 균형)",
        seed,
        n_total,
        len(train_df),
        n_total - len(test_date_list),
        len(test_df),
        len(test_date_list),
    )
    return train_df, test_df, test_date_list
