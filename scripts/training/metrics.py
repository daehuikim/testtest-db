#!/usr/bin/env python3
"""
다양한 오차 측정 지표.
"""

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)"""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1 - ss_res / ss_tot


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%) - 분모에 (|y|+|pred|)/2 사용"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def mase_naive(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """Mean Absolute Scaled Error (naive 기준)"""
    mae_pred = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    if mae_naive < 1e-10:
        return float("inf")
    return mae_pred / mae_naive


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray = None,
) -> dict:
    """모든 지표 계산."""
    out = {
        "mape": mape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if y_naive is not None and len(y_naive) == len(y_true):
        out["mase"] = mase_naive(y_true, y_pred, y_naive)
    return out
