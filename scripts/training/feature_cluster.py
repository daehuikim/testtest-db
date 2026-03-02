#!/usr/bin/env python3
"""
Lag feature clustering: base별 그룹화 후 importance 상위 대표값만 유지.
예: price_per_kg_mean_lag1, lag2, ... lag55 → 1개 (가장 순위 높은 lag)
"""

import logging
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_PATTERN = re.compile(r"^(.+)_lag(\d+)$")


def _parse_feature(f: str) -> Tuple[str, Optional[int]]:
    """(base, lag) 반환. lag 없으면 (f, None)."""
    m = LAG_PATTERN.match(f)
    if m:
        return m.group(1), int(m.group(2))
    return f, None


def reduce_by_lag_representative(
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str = "price_per_kg_mean",
    top_k_per_base: int = 1,
    max_final: int = 50,
    seasonal_lags: Optional[List[int]] = None,
    use_gpu: bool = True,
    random_state: int = 42,
) -> List[str]:
    """
    base별로 그룹화 후 LGBM importance 기준 상위 top_k_per_base개만 유지.
    seasonal_lags(364,365,366 등)는 clustering 제외, 항상 유지.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return features

    seasonal_lags = seasonal_lags or [364, 365, 366]
    seasonal_feature_names = [f"price_per_kg_mean_lag{d}" for d in seasonal_lags]

    # Group by base (seasonal은 제외하고 clustering)
    groups: dict[str, List[str]] = defaultdict(list)
    no_lag: List[str] = []
    auction_no_lag: List[str] = []  # auction_* (lag 없음) → clustering 대상
    always_keep: List[str] = []
    for f in features:
        if f in seasonal_feature_names:
            always_keep.append(f)
            continue
        base, lag = _parse_feature(f)
        if lag is None:
            if f.startswith("auction_"):
                auction_no_lag.append(f)
            else:
                no_lag.append(f)
        else:
            groups[base].append(f)

    # Single-lag or no-lag: keep all (auction은 별도 clustering)
    to_keep = list(no_lag)
    multi_lag_bases = {b: feats for b, feats in groups.items() if len(feats) > 1}
    has_auction_cluster = len(auction_no_lag) >= 2

    if not multi_lag_bases and not has_auction_cluster:
        logger.info("Lag clustering: 다중 lag/auction 그룹 없음")
        return features

    # Compute importance
    data = train_df[train_df[target_col].notna()].fillna(train_df.median(numeric_only=True))
    X = data[features]
    y = data[target_col]

    params = dict(
        n_estimators=200,
        max_depth=6,
        verbosity=-1,
        random_state=random_state,
    )
    if use_gpu:
        params["device"] = "gpu"

    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
    except Exception as e:
        err = str(e).lower()
        if use_gpu and ("opencl" in err or "cuda" in err or "device" in err):
            logger.warning("LGBM GPU 실패, CPU로 importance 계산: %s", str(e)[:40])
            params["device"] = "cpu"
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
        else:
            raise

    imp = pd.Series(model.feature_importances_, index=features)

    for base, feats in multi_lag_bases.items():
        sub = imp[feats].nlargest(top_k_per_base)
        to_keep.extend(sub.index.tolist())

    # auction_* (no lag) clustering: 2개 이상이면 importance 상위 2개만
    if has_auction_cluster:
        auction_top_k = min(2, len(auction_no_lag))
        auction_kept = imp.reindex(auction_no_lag).fillna(0).nlargest(auction_top_k).index.tolist()
        to_keep.extend(auction_kept)
        logger.info("Auction clustering: %s -> %s (top %d)", auction_no_lag, auction_kept, auction_top_k)
    else:
        to_keep.extend(auction_no_lag)

    # Single-lag bases: keep
    for base, feats in groups.items():
        if len(feats) == 1:
            to_keep.append(feats[0])

    kept = list(dict.fromkeys(always_keep + to_keep))  # seasonal 먼저
    if max_final > 0 and len(kept) > max_final:
        # seasonal 유지 + 나머지 중 importance 상위
        rest = [f for f in kept if f not in always_keep]
        n_rest = max(0, max_final - len(always_keep))
        if n_rest < len(rest):
            imp_rest = imp.reindex(rest).fillna(0)
            rest = imp_rest.nlargest(n_rest).index.tolist()
        kept = always_keep + rest
    logger.info(
        "Lag clustering: %d -> %d (base %d개, seasonal %d개, max_final=%d)",
        len(features),
        len(kept),
        len(groups) + (1 if no_lag else 0),
        len(always_keep),
        max_final,
    )
    return kept
