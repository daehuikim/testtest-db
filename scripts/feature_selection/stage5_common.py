#!/usr/bin/env python3
"""
Stage 5: Common Feature Selection across Varieties
품종 리스트업 → 공통적으로 중요한 feature만 최종 확정
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

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


def get_variety_list(df: pd.DataFrame, min_samples: int = 50) -> List[str]:
    """충분한 샘플이 있는 품종 리스트."""
    counts = df.groupby("품종").size()
    return counts[counts >= min_samples].index.tolist()


def importance_per_variety(
    df: pd.DataFrame,
    feature_cols: List[str],
    varieties: List[str],
    config: Optional[dict] = None,
) -> Dict[str, pd.Series]:
    """품종별 feature importance (GPU 지원)."""
    try:
        import lightgbm as lgb
    except ImportError:
        return {}

    device = _get_device(config)
    params = dict(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbosity=-1)
    if device == "gpu":
        params["device"] = "gpu"

    result = {}
    for var in varieties:
        sub = df[df["품종"] == var].sort_values("date").reset_index(drop=True)
        if len(sub) < 30:
            continue
        X = sub[feature_cols].fillna(sub[feature_cols].median())
        y = sub[TARGET_COL]
        model = lgb.LGBMRegressor(**params)
        model = fit_lgb_with_fallback(model, X, y, device)
        result[var] = pd.Series(model.feature_importances_, index=feature_cols)
    return result


def stage5_common(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_variety_ratio: float = 0.5,
    top_k_per_variety: int = 80,
    min_kept: int = 20,
    config: Optional[dict] = None,
) -> Set[str]:
    """
    품종별 importance → 공통 중요 feature. min_kept 미만으로 줄이지 않음.
    """
    cfg = config or load_config()
    stage5 = cfg.get("stage5_common", {})
    min_variety_ratio = stage5.get("min_variety_ratio", min_variety_ratio)
    top_k_per_variety = stage5.get("top_k_per_variety", top_k_per_variety)
    min_kept = stage5.get("min_kept", min_kept)

    varieties = get_variety_list(df, min_samples=50)
    if not varieties:
        return set(feature_cols)

    logger.info("Stage 5: 품종 %d개 분석 중", len(varieties))

    imp_per_var = importance_per_variety(df, feature_cols, varieties, config)
    if not imp_per_var:
        return set(feature_cols)

    n_varieties = len(imp_per_var)
    min_count = max(1, int(n_varieties * min_variety_ratio))
    feature_votes = {f: 0 for f in feature_cols}
    for var, imp in imp_per_var.items():
        top = imp.nlargest(min(top_k_per_variety, len(imp))).index.tolist()
        for f in top:
            feature_votes[f] = feature_votes.get(f, 0) + 1

    kept = {f for f, c in feature_votes.items() if c >= min_count}

    if len(kept) < min_kept:
        # vote 수 상위 min_kept개 유지
        sorted_by_votes = sorted(feature_votes.items(), key=lambda x: -x[1])
        kept = {f for f, _ in sorted_by_votes[:min_kept]}

    logger.info("Stage 5: %d -> %d (공통 중요, min_kept=%d)", len(feature_cols), len(kept), min_kept)
    return kept
