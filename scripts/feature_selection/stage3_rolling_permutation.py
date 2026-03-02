#!/usr/bin/env python3
"""
Stage 3: LightGBM + Rolling Permutation Importance
μ_imp - λ*σ_imp > 0 인 feature만 유지
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


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
) -> np.ndarray:
    """Permutation importance (MAE 기준)."""
    pred = model.predict(X)
    baseline = np.mean(np.abs(pred - y))
    imp = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_shuf = X.copy()
        for _ in range(n_repeats):
            X_shuf.iloc[:, i] = np.random.permutation(X_shuf.iloc[:, i].values)
            pred_shuf = model.predict(X_shuf)
            imp[i] += np.mean(np.abs(pred_shuf - y)) - baseline
    imp /= n_repeats
    return imp


def stage3_rolling_permutation(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_windows: int = 5,
    min_window_size: int = 60,
    stability_lambda: float = 0.3,
    top_percentile: float = 50,
    min_kept: int = 50,
    config: Optional[dict] = None,
) -> Set[str]:
    """
    Rolling window별 permutation importance.
    μ - λ*σ > 0 OR 상위 top_percentile% OR 최소 min_kept개 유지.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm 필요: pip install lightgbm")
        return set(feature_cols)

    cfg = config or load_config()
    stage3 = cfg.get("stage3_rolling_permutation", {})
    n_windows = stage3.get("n_windows", n_windows)
    min_window_size = stage3.get("min_window_size", min_window_size)
    stability_lambda = stage3.get("stability_lambda", stability_lambda)
    top_percentile = stage3.get("top_percentile", top_percentile)
    min_kept = stage3.get("min_kept", min_kept)

    data = df.sort_values("date").reset_index(drop=True)
    data = data[data[TARGET_COL].notna()].copy()
    data = data.fillna(data.median(numeric_only=True))

    if len(data) < min_window_size * 2:
        logger.warning("Stage 3: 데이터 부족, 전체 feature 유지")
        return set(feature_cols)

    n = len(data)
    step = max(1, (n - min_window_size) // n_windows)
    windows = []
    for i in range(n_windows):
        end = n - i * step
        start = max(0, end - min_window_size)
        if end - start < min_window_size // 2:
            break
        windows.append((start, end))

    if not windows:
        return set(feature_cols)

    all_imp = []
    for start, end in windows:
        sub = data.iloc[start:end]
        X = sub[feature_cols]
        y = sub[TARGET_COL].values
        device = _get_device(config)
        params = dict(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbosity=-1)
        if device == "gpu":
            params["device"] = "gpu"
        model = lgb.LGBMRegressor(**params)
        model = fit_lgb_with_fallback(model, X, y, device)
        imp = permutation_importance(model, X, y, n_repeats=3)
        all_imp.append(imp)

    all_imp = np.array(all_imp)
    mu = np.mean(all_imp, axis=0)
    sigma = np.std(all_imp, axis=0) + 1e-10
    score = mu - stability_lambda * sigma

    # 기준 1: μ - λ*σ > 0
    by_stability = {feature_cols[i] for i in np.where(score > 0)[0]}

    # 기준 2: 상위 top_percentile% (mean importance 기준)
    n_keep_pct = max(min_kept, int(len(feature_cols) * top_percentile / 100))
    top_idx = np.argsort(mu)[-n_keep_pct:]
    by_percentile = {feature_cols[i] for i in top_idx}

    # Union: 둘 중 하나라도 통과하면 유지. 최소 min_kept개 보장
    kept = by_stability | by_percentile
    if len(kept) < min_kept:
        top_idx = np.argsort(mu)[-min_kept:]
        kept = {feature_cols[i] for i in top_idx}

    logger.info("Stage 3: %d -> %d (Rolling Perm, stability=%d, pct=%d)", len(feature_cols), len(kept), len(by_stability), len(by_percentile))
    return kept
