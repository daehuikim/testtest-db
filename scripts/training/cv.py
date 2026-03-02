#!/usr/bin/env python3
"""
Expanding Walk-forward CV: Train 구간 내에서 validation 30일 고정.
"""

import logging
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"


def expanding_walk_fold_indices(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 5,
    valid_days: int = 30,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Expanding window: 각 fold에서 train은 누적, valid는 30일 고정.
    Yields: (train_idx, valid_idx) per fold.
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    valid_dates = df["_dt"].dropna().unique()
    valid_dates = np.sort(valid_dates)

    if len(valid_dates) < valid_days * (n_folds + 1):
        logger.warning("데이터 부족: fold 수 또는 valid_days 감소")
        n_folds = max(1, len(valid_dates) // (valid_days * 2))

    step = max(1, (len(valid_dates) - valid_days) // n_folds)
    for i in range(n_folds):
        valid_end_idx = step * (i + 1) + valid_days
        if valid_end_idx > len(valid_dates):
            break
        valid_start_idx = valid_end_idx - valid_days
        valid_dates_fold = valid_dates[valid_start_idx:valid_end_idx]
        train_dates_fold = valid_dates[:valid_start_idx]

        train_mask = df["_dt"].isin(train_dates_fold)
        valid_mask = df["_dt"].isin(valid_dates_fold)
        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]

        if len(train_idx) < 30 or len(valid_idx) < 10:
            continue
        yield train_idx, valid_idx
