#!/usr/bin/env python3
"""
Stage 2: Sparse filtering (multicollinearity 제거)
(A) Elastic Net: TimeSeriesSplit CV, non-zero coefficient
(B) LightGBM fallback: GPU 지원, importance 기반 (수렴 이슈 회피)
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config_loader import load_config

logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"
RANDOM_STATE = 42


def _get_device(config: Optional[dict]) -> str:
    """use_gpu면 'gpu', 아니면 'cpu'."""
    cfg = config or load_config()
    return "gpu" if cfg.get("use_gpu") else "cpu"


def _stage2_lightgbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_ratio: float = 0.5,
    device: str = "cpu",
) -> Set[str]:
    """LightGBM importance 기반. GPU 실패 시 CPU fallback."""
    try:
        import lightgbm as lgb
    except ImportError:
        return set()

    data = df.sort_values("date").reset_index(drop=True)
    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data[TARGET_COL]
    valid = y.notna() & X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 50:
        return set(feature_cols)

    def _fit(dev: str):
        p = dict(n_estimators=200, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
        if dev == "gpu":
            p["device"] = "gpu"
        m = lgb.LGBMRegressor(**p)
        m.fit(X, y)
        return m

    try:
        model = _fit(device)
    except Exception as e:
        if device == "gpu" and ("opencl" in str(e).lower() or "cuda" in str(e).lower() or "device" in str(e).lower()):
            logger.warning("LightGBM GPU 실패, CPU fallback: %s", str(e)[:50])
            model = _fit("cpu")
        else:
            raise

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    n_keep = max(30, int(len(feature_cols) * top_ratio))
    kept = set(imp.nlargest(n_keep).index.tolist())
    return kept


def stage2_elasticnet(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    config: Optional[dict] = None,
) -> Set[str]:
    """
    Elastic Net (또는 LightGBM fallback).
    use_lightgbm_fallback=true 또는 수렴 실패 시 LightGBM 사용 (GPU 가능).
    """
    cfg = config or load_config()
    stage2 = cfg.get("stage2_elasticnet", {})
    use_lgb = stage2.get("use_lightgbm_fallback", False)
    device = _get_device(config)

    if use_lgb:
        top_ratio = stage2.get("lightgbm_top_ratio", 0.5)
        kept = _stage2_lightgbm(df, feature_cols, top_ratio, device)
        logger.info("Stage 2: %d -> %d (LightGBM %s)", len(feature_cols), len(kept), device)
        return kept

    n_splits = stage2.get("cv_splits", n_splits)
    max_iter = stage2.get("max_iter", 50000)
    tol = stage2.get("tol", 0.001)
    alphas = stage2.get("alphas", [0.1, 1.0, 10.0, 100.0])

    data = df.sort_values("date").reset_index(drop=True)
    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data[TARGET_COL]
    valid = y.notna() & X.notna().all(axis=1)
    X = X[valid].values
    y = y[valid].values

    if len(X) < 50:
        return set(feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Objective did not converge")
        model = ElasticNetCV(
            cv=tscv,
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            random_state=RANDOM_STATE,
        )
        model.fit(X_scaled, y)

    nonzero = np.abs(model.coef_) > 1e-6
    kept = {feature_cols[i] for i in np.where(nonzero)[0]}

    if len(kept) < 10:
        logger.warning("Stage 2: Elastic Net non-zero 적음(%d), LightGBM fallback", len(kept))
        kept = _stage2_lightgbm(df, feature_cols, 0.5, device)

    logger.info("Stage 2: %d -> %d (Elastic Net)", len(feature_cols), len(kept))
    return kept
