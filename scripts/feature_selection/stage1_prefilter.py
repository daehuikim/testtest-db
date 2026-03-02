#!/usr/bin/env python3
"""
Stage 1: Statistical Pre-screening
(A) Cross-correlation: |ρ| > threshold 유의 lag만 유지
(B) Mutual Information: 상위 percentile만 유지
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"
EXCLUDE_COLS = {"date", "품종", "price_per_kg_mean", "price_per_kg_median", "price_per_kg_std"}


from .config_loader import load_config


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Feature 후보 컬럼 (수치형, exclude 제외)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in EXCLUDE_COLS]


def stage1_prefilter(
    df: pd.DataFrame,
    ccf_threshold: float = 0.15,
    mi_percentile: float = 75,
    config: Optional[dict] = None,
) -> Set[str]:
    """
    CCF + MI 기반 pre-filter.
    Returns: 유지할 feature set.
    """
    cfg = config or load_config()
    stage1 = cfg.get("stage1_prefilter", {})
    ccf_threshold = stage1.get("ccf_threshold", ccf_threshold)
    mi_percentile = stage1.get("mi_percentile", mi_percentile)

    features = get_feature_cols(df)
    if not features:
        return set()

    # 결측 처리: 컬럼별 median, 남은 NaN은 0
    use = df[[TARGET_COL] + features].copy()
    use = use.dropna(subset=[TARGET_COL])
    valid_features = []
    for c in features:
        if c not in use.columns:
            continue
        if use[c].isna().all():
            use = use.drop(columns=[c])
        else:
            use[c] = use[c].fillna(use[c].median())
            valid_features.append(c)
    use = use.fillna(0)
    features = valid_features
    y = use[TARGET_COL]
    X = use[features]

    # (A) Cross-correlation
    corrs = X.corrwith(y).abs()
    ccf_pass = set(corrs[corrs >= ccf_threshold].index.tolist())

    # (B) Mutual Information
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=features)
    thresh = np.percentile(mi_series, 100 - mi_percentile)
    mi_pass = set(mi_series[mi_series >= thresh].index.tolist())

    # Union: CCF OR MI 통과
    kept = ccf_pass | mi_pass
    logger.info(
        "Stage 1: %d -> %d (CCF pass: %d, MI pass: %d)",
        len(features), len(kept), len(ccf_pass), len(mi_pass),
    )
    return kept
