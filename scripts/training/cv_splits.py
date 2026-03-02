#!/usr/bin/env python3
"""
다양한 시계열 CV: Expanding, TimeSeriesSplit, Purged Group Time Series Split.
"""

import logging
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def expanding_walk_fold_indices(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 5,
    n_splits: Optional[int] = None,  # n_folds와 동일 (호환용)
    valid_days: int = 30,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Expanding window: train 누적, valid 30일 고정."""
    if n_splits is not None:
        n_folds = n_splits
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    valid_dates = df["_dt"].dropna().unique()
    valid_dates = np.sort(valid_dates)

    if len(valid_dates) < valid_days * (n_folds + 1):
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


def timeseries_split_indices(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    n_folds: Optional[int] = None,  # n_splits와 동일 (호환용)
    **kwargs,  # valid_days 등 무시
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """sklearn TimeSeriesSplit 스타일: expanding window."""
    if n_folds is not None:
        n_splits = n_folds
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < 100:
        yield np.arange(0, n // 2), np.arange(n // 2, n)
        return

    test_size = max(20, n // (n_splits + 1))
    for i in range(1, n_splits + 1):
        train_end = n - (n_splits - i + 1) * test_size - test_size
        test_end = train_end + test_size
        if train_end < 30 or test_end > n:
            continue
        train_idx = np.arange(0, train_end)
        valid_idx = np.arange(train_end, test_end)
        if len(train_idx) >= 30 and len(valid_idx) >= 10:
            yield train_idx, valid_idx


def purged_group_timeseries_indices(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    n_folds: Optional[int] = None,  # n_splits와 동일 (호환용, 무시됨)
    valid_days: int = 30,
    purge_days: int = 7,
    embargo_days: int = 3,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Purged Group Time Series Split.
    - purge: valid 구간과 겹치는 train 샘플 제거
    - embargo: valid 직전 embargo_days도 train에서 제외 (leakage 방지)
    """
    if n_folds is not None:
        n_splits = n_folds
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    valid_dates = df["_dt"].dropna().unique()
    valid_dates = np.sort(valid_dates)

    if len(valid_dates) < valid_days * (n_splits + 1):
        n_splits = max(1, len(valid_dates) // (valid_days * 2))

    step = max(1, (len(valid_dates) - valid_days) // n_splits)
    for i in range(n_splits):
        valid_end_idx = step * (i + 1) + valid_days
        if valid_end_idx > len(valid_dates):
            break
        valid_start_idx = valid_end_idx - valid_days
        valid_dates_fold = valid_dates[valid_start_idx:valid_end_idx]
        train_dates_fold = valid_dates[:valid_start_idx]

        # Purge + Embargo: valid 직전 purge_days+embargo_days 이내 train 제외
        valid_min = valid_dates_fold.min()
        cutoff = valid_min - pd.Timedelta(days=purge_days + embargo_days)
        train_dates_fold = [d for d in train_dates_fold if d <= cutoff]

        if not train_dates_fold:
            continue

        train_mask = df["_dt"].isin(train_dates_fold)
        valid_mask = df["_dt"].isin(valid_dates_fold)
        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]

        if len(train_idx) < 30 or len(valid_idx) < 10:
            continue
        yield train_idx, valid_idx


def get_cv_folds(
    df: pd.DataFrame,
    method: str = "expanding",
    **kwargs,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """CV method에 따라 fold indices 생성."""
    if method == "expanding":
        return expanding_walk_fold_indices(df, **kwargs)
    elif method == "timeseries":
        return timeseries_split_indices(df, **kwargs)
    elif method == "purged":
        return purged_group_timeseries_indices(df, **kwargs)
    else:
        return expanding_walk_fold_indices(df, **kwargs)
