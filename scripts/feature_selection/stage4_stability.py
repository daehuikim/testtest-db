#!/usr/bin/env python3
"""
Stage 4: Temporal Stability Filtering
연도/시즌별 importance → Jaccard, rank correlation
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"
RANDOM_STATE = 42


from .config_loader import load_config
from .device_utils import fit_lgb_with_fallback


def _get_device(config) -> str:
    cfg = config or load_config()
    return "gpu" if cfg.get("use_gpu") else "cpu"


def stage4_stability(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_jaccard: float = 0.5,
    min_rank_corr: float = 0.3,
    min_kept: int = 30,
    config: Optional[dict] = None,
) -> Set[str]:
    """
    연도별 importance rank → 안정적인 feature 유지.
    min_kept 미만으로 줄이지 않음.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return set(feature_cols)

    cfg = config or load_config()
    stage4 = cfg.get("stage4_stability", {})
    min_jaccard = stage4.get("min_jaccard", min_jaccard)
    min_rank_corr = stage4.get("min_rank_corr", min_rank_corr)
    min_kept = stage4.get("min_kept", min_kept)

    # feature 적을 때는 스킵
    if len(feature_cols) <= min_kept:
        return set(feature_cols)

    data = df.sort_values("date").reset_index(drop=True)
    data["year"] = pd.to_datetime(data["date"], format="%Y%m%d").dt.year
    years = data["year"].unique()
    if len(years) < 2:
        return set(feature_cols)

    ranks_per_year = []
    for yr in years:
        sub = data[data["year"] == yr]
        if len(sub) < 30:
            continue
        X = sub[feature_cols].fillna(sub[feature_cols].median())
        y = sub[TARGET_COL]
        device = _get_device(config)
        params = dict(n_estimators=50, max_depth=5, random_state=RANDOM_STATE, verbosity=-1)
        if device == "gpu":
            params["device"] = "gpu"
        model = lgb.LGBMRegressor(**params)
        model = fit_lgb_with_fallback(model, X, y, device)
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        ranks_per_year.append(imp.rank(ascending=False))

    if len(ranks_per_year) < 2:
        return set(feature_cols)

    rank_df = pd.DataFrame(ranks_per_year)
    corrs = rank_df.T.corr()
    mean_corr = (corrs.sum().sum() - len(corrs)) / (len(corrs) * (len(corrs) - 1)) if corrs.size > 1 else 1.0

    # k = 상위 k개 (최소 min_kept)
    k = max(min_kept, min(30, len(feature_cols) // 2))
    mean_rank = pd.DataFrame(ranks_per_year).mean()
    rank_std = pd.DataFrame(ranks_per_year).std().fillna(0)

    # 안정성: rank_std가 낮거나 mean_rank가 상위 k 안
    stable = (rank_std < rank_std.median()) | (mean_rank <= k)
    kept = set(mean_rank[stable].index.tolist())

    if len(kept) < min_kept:
        # 상위 min_kept개 by mean_rank (낮을수록 좋음)
        top_idx = mean_rank.nsmallest(min_kept).index.tolist()
        kept = set(top_idx)

    jaccard = 0.0
    logger.info("Stage 4: %d -> %d (Stability, min_kept=%d)", len(feature_cols), len(kept), min_kept)
    return kept
