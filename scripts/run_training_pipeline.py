#!/usr/bin/env python3
"""
학습 파이프라인 (단일 진입점)

1. 데이터 분할: Train 고정 + Test 계절 균형 랜덤
2. Expanding Walk-forward CV
3. Baseline (naive, 7-day seasonal)
4. Feature 재정제 (SHAP → 40~60개)
5. 모델 비교 (LGBM / CatBoost / ElasticNet)
6. Stacking 여부 판단 (OOF corr < 0.95)
7. 최종 학습 + Test 평가 (계절별 MAPE)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"
RANDOM_STATE = 42
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "best_model"

# Season for MAPE breakdown
SEASON_MONTHS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "fall": [9, 10, 11], "winter": [12, 1, 2]}


def _month_to_season(month: int) -> str:
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    return "unknown"


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


Y_TARGET_COL = "_y_target"  # 학습용 (log1p 또는 raw)


def preprocess_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    clip_pct: float = 1.0,
    use_log: bool = True,
) -> tuple:
    """
    Outlier clip + log 변환.
    train 기준으로 clip 구간 계산 후 train/test 동일 적용.
    Returns: (train_df, test_df) with _y_target 추가
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    valid = train_df[target_col].notna()
    if valid.sum() < 10:
        train_df[Y_TARGET_COL] = train_df[target_col]
        test_df[Y_TARGET_COL] = test_df[target_col]
        return train_df, test_df

    lo = np.percentile(train_df.loc[valid, target_col], clip_pct)
    hi = np.percentile(train_df.loc[valid, target_col], 100 - clip_pct)
    train_df[target_col] = train_df[target_col].clip(lo, hi)
    test_df[target_col] = test_df[target_col].clip(lo, hi)
    if use_log:
        train_df[Y_TARGET_COL] = np.log1p(train_df[target_col])
        test_df[Y_TARGET_COL] = np.log1p(test_df[target_col])
    else:
        train_df[Y_TARGET_COL] = train_df[target_col]
        test_df[Y_TARGET_COL] = test_df[target_col]
    return train_df, test_df


def _lgb_params(config: dict, use_gpu: bool) -> dict:
    cfg = config.get("lgb", {})
    p = dict(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=6,
        num_leaves=31,
        verbosity=-1,
        random_state=RANDOM_STATE,
    )
    if use_gpu:
        p["device"] = "gpu"
    return p


def _cb_params(config: dict, use_gpu: bool) -> dict:
    cfg = config.get("catboost", {})
    p = dict(
        iterations=cfg.get("iterations", 500),
        depth=6,
        verbose=0,
        random_seed=RANDOM_STATE,
    )
    if use_gpu:
        p["task_type"] = "GPU"
    return p


def load_data_and_features(
    data_path: Path = None,
    report_path: Path = None,
) -> tuple:
    """merged CSV + feature_selection report 로드."""
    data_path = data_path or PROJECT_ROOT / "temp" / "merged_stage0.csv"
    report_path = report_path or PROJECT_ROOT / "reports" / "feature_selection_pipeline_report.json"

    if not data_path.exists():
        logger.info("merged_stage0 없음 → Feature selection pipeline 실행 후 temp/merged_stage0.csv 생성")
        from feature_selection.config_loader import load_config
        from feature_selection.data_merger import DataMerger
        from feature_selection.stage5_common import get_variety_list

        config = load_config()
        merger = DataMerger(data_root=PROJECT_ROOT / "data" / "raw", config=config)
        df = merger.run()
        variety_cfg = config.get("variety_filter", {})
        if variety_cfg.get("min_pct"):
            cnt = df["품종"].value_counts()
            pct = cnt / len(df) * 100
            keep = pct[pct >= variety_cfg["min_pct"]].index.tolist()
            df = df[df["품종"].isin(keep)].copy()
    else:
        df = pd.read_csv(data_path, encoding="utf-8-sig")

    df["date"] = df["date"].astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)

    features = []
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        features = report.get("final_features_ranked", report.get("final_features", []))
    # report에 feature가 너무 적으면 (<20) 데이터에서 전체 numeric 사용
    if not features or len(features) < 20:
        exclude = {"date", "품종", TARGET_COL, "price_per_kg_median", "price_per_kg_std"}
        fallback = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64"]]
        if fallback and (not features or len(features) < 20):
            features = fallback
            if report_path.exists():
                logger.info("Report feature %d개 → 데이터 전체 numeric %d개 사용", len(report.get("final_features", [])), len(features))
    features = [f for f in features if f in df.columns]
    logger.info("Feature 수: %d", len(features))
    return df, features


def run_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> dict:
    """Naive (y_{t-1}), 7-day seasonal naive MAPE."""
    results = {}
    lag1 = "price_per_kg_mean_lag1"
    lag7 = "price_per_kg_mean_lag7" if "price_per_kg_mean_lag7" in train_df.columns else None

    # Naive: y_t = y_{t-1}
    if lag1 in train_df.columns and lag1 in test_df.columns:
        valid = test_df[TARGET_COL].notna() & test_df[lag1].notna()
        if valid.sum() > 0:
            m = mape(test_df.loc[valid, TARGET_COL].values, test_df.loc[valid, lag1].values)
            results["naive_lag1"] = m
            logger.info("Baseline Naive (y_{t-1}) MAPE: %.2f%%", m)

    # 7-day seasonal
    if lag7 and lag7 in test_df.columns:
        valid = test_df[TARGET_COL].notna() & test_df[lag7].notna()
        if valid.sum() > 0:
            m = mape(test_df.loc[valid, TARGET_COL].values, test_df.loc[valid, lag7].values)
            results["seasonal_7d"] = m
            logger.info("Baseline 7-day seasonal MAPE: %.2f%%", m)

    return results


def refine_features_shap(
    train_df: pd.DataFrame,
    features: list,
    config: dict,
) -> list:
    """CV fold별 SHAP → 평균 importance, rank corr > 0.4 → 40~60개."""
    from training.cv import expanding_walk_fold_indices

    try:
        import lightgbm as lgb
        import shap
    except ImportError:
        logger.warning("shap 미설치 → feature refinement 스킵")
        return features

    cfg = config.get("feature_refine", {})
    top_n = cfg.get("top_n_by_mean", 60)
    min_corr = cfg.get("min_rank_corr", 0.4)
    max_final = cfg.get("max_final", 60)
    min_final = cfg.get("min_final", 40)

    data = train_df.sort_values("date").reset_index(drop=True)
    data = data[data[TARGET_COL].notna()].fillna(data.median(numeric_only=True))
    X = data[features]
    y = data[Y_TARGET_COL] if Y_TARGET_COL in data.columns else data[TARGET_COL]

    ranks_per_fold = []
    for train_idx, valid_idx in expanding_walk_fold_indices(
        data, n_folds=5, valid_days=30
    ):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        try:
            from feature_selection.device_utils import fit_lgb_with_fallback
            use_gpu = config.get("use_gpu", False)
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1, random_state=RANDOM_STATE, device="gpu" if use_gpu else "cpu")
            m = fit_lgb_with_fallback(m, X_tr, y_tr, "gpu" if use_gpu else "cpu")
            expl = shap.TreeExplainer(m, X_tr)
            sh = expl.shap_values(X_tr)
            imp = np.abs(sh).mean(axis=0)
            rk = pd.Series(imp, index=features).rank(ascending=False)
            ranks_per_fold.append(rk)
        except Exception as e:
            logger.warning("SHAP fold 실패: %s", str(e)[:50])

    if len(ranks_per_fold) < 2:
        return features[:max_final]

    rank_df = pd.DataFrame(ranks_per_fold)
    mean_rank = rank_df.mean()
    corrs = rank_df.T.corr()
    np.fill_diagonal(corrs.values, np.nan)
    mean_corr = np.nanmean(corrs.values)
    if mean_corr < min_corr:
        logger.info("Fold rank corr %.2f < %.2f → 상위 %d개만 유지", mean_corr, min_corr, top_n)

    kept = mean_rank.nsmallest(min(top_n, len(features))).index.tolist()
    kept = kept[:max_final]
    if len(kept) < min_final:
        kept = mean_rank.nsmallest(min_final).index.tolist()
    logger.info("Feature refine: %d -> %d", len(features), len(kept))
    return kept


def expanding_cv_mape(
    train_df: pd.DataFrame,
    features: list,
    model_name: str,
    config: dict,
    use_log: bool = False,
    cv_method: Optional[str] = None,
) -> tuple:
    """CV로 MAPE 평균, std, worst fold 반환. OOF 예측도 반환 (stacking용)."""
    from training.cv_splits import get_cv_folds

    data = train_df.sort_values("date").reset_index(drop=True)
    data = data[data[TARGET_COL].notna()].copy()
    data = data.fillna(data[features].median())
    X = data[features]
    y_train = data[Y_TARGET_COL].values if Y_TARGET_COL in data.columns else data[TARGET_COL].values
    y_orig = data[TARGET_COL].values  # metrics용

    oof_pred = np.full(len(data), np.nan)
    fold_mapes = []
    use_gpu = config.get("use_gpu", False)
    cv_cfg = config.get("cv", {})
    _cv_method = cv_method or cv_cfg.get("method", "expanding")
    config["cv"] = {**cv_cfg, "method": _cv_method}

    fold_gen = get_cv_folds(
        data,
        method=config["cv"].get("method", "expanding"),
        n_folds=cv_cfg.get("n_folds", 5),
        n_splits=cv_cfg.get("n_folds", 5),
        valid_days=cv_cfg.get("valid_days", 30),
        purge_days=cv_cfg.get("purge_days", 7),
        embargo_days=cv_cfg.get("embargo_days", 3),
    )

    for train_idx, valid_idx in fold_gen:
        X_tr, y_tr = X.iloc[train_idx], y_train[train_idx]
        X_val = X.iloc[valid_idx]
        y_val_orig = y_orig[valid_idx]

        try:
            if model_name == "lgb":
                import lightgbm as lgb
                from feature_selection.device_utils import fit_lgb_with_fallback
                m = lgb.LGBMRegressor(**_lgb_params(config, use_gpu))
                m = fit_lgb_with_fallback(m, X_tr, y_tr, "gpu" if use_gpu else "cpu")
            elif model_name == "catboost":
                import catboost as cb
                try:
                    m = cb.CatBoostRegressor(**_cb_params(config, use_gpu))
                    m.fit(X_tr, y_tr)
                except Exception as e:
                    if use_gpu and ("gpu" in str(e).lower() or "cuda" in str(e).lower() or "device" in str(e).lower()):
                        logger.warning("CatBoost GPU 실패, CPU fallback: %s", str(e)[:40])
                        m = cb.CatBoostRegressor(**_cb_params(config, False))
                        m.fit(X_tr, y_tr)
                    else:
                        raise
            elif model_name == "elasticnet":
                from sklearn.linear_model import ElasticNet
                m = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
                m.fit(X_tr, y_tr)
            elif model_name == "lstm":
                from training.deep_models import LSTMWrapper, _get_lag_sequence_cols, build_sequence_array
                seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("lstm", {}).get("seq_len", 365))
                if len(seq_cols) < 2:
                    continue
                cfg = config.get("lstm", {})
                X_seq_tr = build_sequence_array(data.iloc[train_idx], seq_cols)
                X_seq_val = build_sequence_array(data.iloc[valid_idx], seq_cols)
                m = LSTMWrapper(
                    seq_len=len(seq_cols), hidden=cfg.get("hidden", 64), layers=cfg.get("layers", 2),
                    epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 10),
                    dropout=cfg.get("dropout", 0.2),
                )
                m.fit(X_seq_tr, y_tr, X_val=X_seq_val, y_val=y_train[valid_idx])
                pred_val = m.predict(X_seq_val)
            elif model_name == "transformer":
                from training.deep_models import TransformerWrapper, _get_lag_sequence_cols, build_sequence_array
                seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("transformer", {}).get("seq_len", 365))
                if len(seq_cols) < 2:
                    continue
                cfg = config.get("transformer", {})
                X_seq_tr = build_sequence_array(data.iloc[train_idx], seq_cols)
                X_seq_val = build_sequence_array(data.iloc[valid_idx], seq_cols)
                m = TransformerWrapper(
                    seq_len=len(seq_cols), d_model=cfg.get("d_model", 32), nhead=cfg.get("nhead", 4),
                    num_layers=cfg.get("num_layers", 2), epochs=cfg.get("epochs", 100),
                    patience=cfg.get("patience", 10), dropout=cfg.get("dropout", 0.2),
                )
                m.fit(X_seq_tr, y_tr, X_val=X_seq_val, y_val=y_train[valid_idx])
                pred_val = m.predict(X_seq_val)
            else:
                continue

            if model_name not in ("lstm", "transformer"):
                pred_val = m.predict(X_val)
            oof_pred[valid_idx] = pred_val  # stacking용 (log scale 유지)
            pred_for_mape = np.expm1(pred_val) if use_log else pred_val
            fold_mapes.append(mape(y_val_orig, pred_for_mape))
        except Exception as e:
            logger.warning("Fold 실패 (%s): %s", model_name, str(e)[:50])

    if not fold_mapes:
        return float("inf"), float("inf"), float("inf"), oof_pred
    return np.mean(fold_mapes), np.std(fold_mapes), max(fold_mapes), oof_pred


def run_pipeline(
    data_path: Path = None,
    config_path: Path = None,
    seed: Optional[int] = None,
    skip_feature_refine: bool = False,
    skip_shap: bool = True,
    models: Optional[list] = None,
    cv_method: Optional[str] = None,
    no_deep: bool = False,
) -> dict:
    """전체 파이프라인 실행."""
    config_path = config_path or PROJECT_ROOT / "config" / "training_config.yaml"
    config = {}
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass

    use_gpu = config.get("use_gpu", False)
    if use_gpu:
        logger.info("GPU 모드 활성화 (LightGBM, CatBoost)")

    df, features = load_data_and_features(data_path=data_path)
    if len(features) < 5:
        logger.warning("Feature 수 부족. feature selection pipeline 먼저 실행 권장.")

    # 품종: 대표 3품종 또는 최다 품종
    variety_cfg = config.get("variety", {})
    top_variety = None
    variety_info = {}
    rep_varieties = variety_cfg.get("representative_varieties") or []
    if not rep_varieties:
        try:
            import yaml
            fs_config_path = PROJECT_ROOT / "config" / "feature_selection_config.yaml"
            if fs_config_path.exists():
                with open(fs_config_path, encoding="utf-8") as f:
                    fs_cfg = yaml.safe_load(f) or {}
                rep_varieties = fs_cfg.get("representative_varieties") or []
        except Exception:
            pass

    if "품종" in df.columns:
        if rep_varieties:
            df = df[df["품종"].isin(rep_varieties)].copy()
            df["date"] = df["date"].astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)
            # 유효 날짜: 대표 품종 모두 데이터 있는 날짜만
            dates_per_var = df.groupby("품종")["date"].apply(lambda s: set(s.dropna().unique()))
            sets_to_intersect = [dates_per_var.get(v, set()) for v in rep_varieties if dates_per_var.get(v)]
            valid_dates = set.intersection(*sets_to_intersect) if sets_to_intersect else set()
            valid_dates = sorted(valid_dates)
            df = df[df["date"].isin(valid_dates)].copy()
            variety_info = df["품종"].value_counts().to_dict()
            top_variety = ", ".join(rep_varieties)
            logger.info("대표 품종 %s: 유효 %d일, 샘플수: %s", rep_varieties, len(valid_dates), variety_info)
        elif variety_cfg.get("use_top_only", True):
            cnt = df["품종"].value_counts()
            top_n = variety_cfg.get("top_n", 1)
            top_varieties = cnt.head(top_n).index.tolist()
            df = df[df["품종"].isin(top_varieties)].copy()
            variety_info = {v: int(cnt[v]) for v in top_varieties}
            top_variety = top_varieties[0] if len(top_varieties) == 1 else ", ".join(top_varieties)
            logger.info("예측 품종: %s (최다 %d개, 샘플수: %s)", top_varieties, top_n, variety_info)

    results = {"variety": top_variety, "variety_counts": variety_info}

    from training.split import train_test_split
    split_cfg = config.get("split", {})
    split_seed = seed if seed is not None else split_cfg.get("seed", RANDOM_STATE)
    train_df, test_df, test_dates = train_test_split(
        df,
        test_days=split_cfg.get("test_days", 0),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_seed,
    )

    # Target 전처리: outlier clip + log
    target_cfg = config.get("target", {})
    use_log = target_cfg.get("use_log", True)
    clip_pct = target_cfg.get("outlier_clip_pct", 1.0)
    train_df, test_df = preprocess_target(
        train_df, test_df, TARGET_COL,
        clip_pct=clip_pct, use_log=use_log,
    )
    if use_log:
        logger.info("Target: log1p(price) 학습, expm1 복원 후 평가")
    if clip_pct > 0:
        logger.info("Outlier clip: 상/하위 %.1f%%", clip_pct)

    # Baseline
    baseline_results = run_baselines(train_df, test_df, features)
    results["baseline"] = baseline_results

    # Lag clustering: base별 대표 lag 1개 (94개 → ~40개 수준)
    use_gpu = config.get("use_gpu", False)
    cluster_cfg = config.get("feature_cluster", {})
    if cluster_cfg.get("enabled", True) and len(features) > 30:
        from training.feature_cluster import reduce_by_lag_representative
        features = reduce_by_lag_representative(
            train_df,
            features,
            target_col=Y_TARGET_COL if Y_TARGET_COL in train_df.columns else TARGET_COL,
            top_k_per_base=cluster_cfg.get("top_k_per_base", 1),
            max_final=cluster_cfg.get("max_final", 50),
            seasonal_lags=cluster_cfg.get("seasonal_lags", [364, 365, 366]),
            use_gpu=use_gpu,
            random_state=RANDOM_STATE,
        )
        results["n_after_cluster"] = len(features)

    # Feature refine: SHAP 또는 LGBM importance로 40~60개
    if not skip_feature_refine and not skip_shap and len(features) > 60:
        features = refine_features_shap(train_df, features, config)
    elif len(features) > 60:
        # SHAP 스킵 시 LGBM importance로 상위 60개
        try:
            import lightgbm as lgb
            from feature_selection.device_utils import fit_lgb_with_fallback
            tr = train_df[train_df[TARGET_COL].notna()].fillna(train_df.median(numeric_only=True))
            y_tr = tr[Y_TARGET_COL] if Y_TARGET_COL in tr.columns else tr[TARGET_COL]
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1, random_state=RANDOM_STATE, device="gpu" if use_gpu else "cpu")
            m = fit_lgb_with_fallback(m, tr[features], y_tr, "gpu" if use_gpu else "cpu")
            imp = pd.Series(m.feature_importances_, index=features).nlargest(60)
            features = imp.index.tolist()
            logger.info("Feature 상위 60개 (LGBM importance) 사용")
        except Exception as e:
            logger.warning("Feature refine 실패: %s → 상위 60개", str(e)[:40])
            features = features[:60]

    # Model comparison (다양한 모델: LGB, CatBoost, ElasticNet, LSTM, Transformer)
    cfg_models = config.get("models")
    if models:
        model_list = list(models)
    elif cfg_models:
        model_list = cfg_models if isinstance(cfg_models, list) else [cfg_models]
    else:
        model_list = ["lgb", "catboost", "elasticnet"]
        if not no_deep:
            if config.get("lstm", {}).get("enabled", True):
                model_list.append("lstm")
            if config.get("transformer", {}).get("enabled", True):
                model_list.append("transformer")
    logger.info("실행 모델: %s", model_list)

    model_scores = {}
    oof_dict = {}
    for name in model_list:
        try:
            mean_mape, std_mape, worst_mape, oof = expanding_cv_mape(
                train_df, features, name, config, use_log=use_log, cv_method=cv_method,
            )
            model_scores[name] = {"mean": mean_mape, "std": std_mape, "worst": worst_mape}
            if oof is not None and np.isfinite(oof).sum() > 10:
                oof_dict[name] = oof
            logger.info("%s CV MAPE: %.2f%% ± %.2f (worst %.2f%%)", name, mean_mape, std_mape, worst_mape)
        except ImportError as e:
            logger.warning("%s 스킵: %s", name, str(e)[:50])
        except Exception as e:
            logger.warning("%s 실패: %s", name, str(e)[:80])

    results["cv_scores"] = model_scores

    # 앙상블 탐색: 2~5 model 조합 중 OOF MAPE 최소인 것 선택
    from itertools import combinations
    from sklearn.linear_model import Ridge

    data = train_df.sort_values("date").reset_index(drop=True)
    data = data[data[TARGET_COL].notna()].copy()
    y_orig = data[TARGET_COL].values

    best_combo = None
    best_combo_mape = float("inf")
    best_combo_name = "lgb"

    for r in range(2, min(6, len(oof_dict) + 1)):
        for combo in combinations(oof_dict.keys(), r):
            try:
                oofs = np.column_stack([oof_dict[m] for m in combo])
                valid = np.all(np.isfinite(oofs), axis=1)
                if valid.sum() < 20:
                    continue
                meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
                y_log = data[Y_TARGET_COL].values[valid] if Y_TARGET_COL in data.columns else data[TARGET_COL].values[valid]
                meta.fit(oofs[valid], y_log)
                oof_pred = np.full(len(oofs), np.nan)
                oof_pred[valid] = meta.predict(oofs[valid])
                oof_pred_orig = np.expm1(oof_pred) if use_log else oof_pred
                valid2 = np.isfinite(oof_pred_orig) & np.isfinite(y_orig)
                if valid2.sum() < 20:
                    continue
                mape_val = mape(y_orig[valid2], oof_pred_orig[valid2])
                combo_name = "+".join(combo)
                if mape_val < best_combo_mape:
                    best_combo_mape = mape_val
                    best_combo = list(combo)
                    best_combo_name = f"stacking({combo_name})"
                logger.info("  앙상블 %s OOF MAPE: %.2f%%", combo_name, mape_val)
            except Exception as e:
                logger.warning("  앙상블 %s 스킵: %s", "+".join(combo), str(e)[:60])

    use_stacking = best_combo is not None and len(best_combo) >= 2
    stacking_oofs = [oof_dict[m] for m in best_combo] if best_combo else []
    if use_stacking:
        logger.info("최적 앙상블: %s (OOF MAPE %.2f%%)", best_combo_name, best_combo_mape)

    # 단일 모델 best (앙상블보다 나쁠 수 있음)
    best_single = min(
        (k for k, v in model_scores.items() if v["mean"] < 1e9),
        key=lambda k: (model_scores[k]["mean"], model_scores[k]["std"]),
        default="lgb",
    )
    best_name = best_combo_name if (use_stacking and best_combo_mape < model_scores.get(best_single, {}).get("mean", 1e9)) else best_single

    # Final train on full train set
    train_full = train_df[train_df[TARGET_COL].notna()].fillna(train_df.median(numeric_only=True))
    X_tr = train_full[features]
    y_tr = train_full[Y_TARGET_COL] if Y_TARGET_COL in train_full.columns else train_full[TARGET_COL]
    X_te = test_df[features].fillna(train_full[features].median())
    y_te = test_df[TARGET_COL].values  # metrics용 원본
    valid_te = np.isfinite(y_te)

    final_model = None
    pred = None

    base_models_for_checkpoint: List[Any] = []
    meta_for_checkpoint = None
    seq_cols_for_checkpoint: Optional[List[str]] = None

    if use_stacking and best_combo and len(best_combo) >= 2:
        # Stacking: best_combo 모델들 → Ridge meta
        import lightgbm as lgb
        import catboost as cb
        from sklearn.linear_model import Ridge
        from feature_selection.device_utils import fit_lgb_with_fallback

        preds_tr, preds_te = [], []
        for mname in best_combo:
            if mname == "lgb":
                m = lgb.LGBMRegressor(**_lgb_params(config, use_gpu))
                m = fit_lgb_with_fallback(m, X_tr, y_tr, "gpu" if use_gpu else "cpu")
                base_models_for_checkpoint.append(m)
                preds_tr.append(m.predict(X_tr))
                preds_te.append(m.predict(X_te))
            elif mname == "catboost":
                try:
                    m = cb.CatBoostRegressor(**_cb_params(config, use_gpu))
                    m.fit(X_tr, y_tr)
                except Exception as e:
                    if use_gpu and ("gpu" in str(e).lower() or "cuda" in str(e).lower() or "device" in str(e).lower()):
                        m = cb.CatBoostRegressor(**_cb_params(config, False))
                        m.fit(X_tr, y_tr)
                    else:
                        raise
                base_models_for_checkpoint.append(m)
                preds_tr.append(m.predict(X_tr))
                preds_te.append(m.predict(X_te))
            elif mname == "elasticnet":
                from sklearn.linear_model import ElasticNet
                m = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
                m.fit(X_tr, y_tr)
                base_models_for_checkpoint.append(m)
                preds_tr.append(m.predict(X_tr))
                preds_te.append(m.predict(X_te))
            elif mname == "lstm":
                from training.deep_models import LSTMWrapper, _get_lag_sequence_cols, build_sequence_array
                seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("lstm", {}).get("seq_len", 365))
                if len(seq_cols) >= 2:
                    cfg = config.get("lstm", {})
                    X_seq_tr = build_sequence_array(train_full, seq_cols)
                    X_seq_te = build_sequence_array(test_df, seq_cols)
                    m = LSTMWrapper(seq_len=len(seq_cols), hidden=cfg.get("hidden", 64), layers=cfg.get("layers", 2), epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 10), dropout=cfg.get("dropout", 0.2))
                    m.fit(X_seq_tr, y_tr.values)
                    base_models_for_checkpoint.append(m)
                    seq_cols_for_checkpoint = seq_cols
                    preds_tr.append(m.predict(X_seq_tr))
                    preds_te.append(m.predict(X_seq_te))
            elif mname == "transformer":
                from training.deep_models import TransformerWrapper, _get_lag_sequence_cols, build_sequence_array
                seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("transformer", {}).get("seq_len", 365))
                if len(seq_cols) >= 2:
                    cfg = config.get("transformer", {})
                    X_seq_tr = build_sequence_array(train_full, seq_cols)
                    X_seq_te = build_sequence_array(test_df, seq_cols)
                    m = TransformerWrapper(seq_len=len(seq_cols), d_model=cfg.get("d_model", 32), nhead=cfg.get("nhead", 4), num_layers=cfg.get("num_layers", 2), epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 10), dropout=cfg.get("dropout", 0.2))
                    m.fit(X_seq_tr, y_tr.values)
                    base_models_for_checkpoint.append(m)
                    seq_cols_for_checkpoint = seq_cols
                    preds_tr.append(m.predict(X_seq_tr))
                    preds_te.append(m.predict(X_seq_te))
        if len(preds_tr) >= 2:
            try:
                oof_tr = np.column_stack(preds_tr)
                oof_te = np.column_stack(preds_te)
                valid_tr = np.all(np.isfinite(oof_tr), axis=1)
                if valid_tr.sum() < 20:
                    raise ValueError("Stacking: 유효한 train 샘플 부족")
                meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
                meta.fit(oof_tr[valid_tr], y_tr.values[valid_tr])
                meta_for_checkpoint = meta
                valid_te_stk = np.all(np.isfinite(oof_te), axis=1)
                pred = np.full(len(oof_te), np.nan)
                if valid_te_stk.sum() > 0:
                    pred[valid_te_stk] = meta.predict(oof_te[valid_te_stk])
                best_name = best_combo_name
                logger.info("Stacking %s (%d models→Ridge) 적용", best_combo_name, len(preds_tr))
            except Exception as e:
                logger.warning("Stacking 실패, 단일 모델 사용: %s", str(e)[:80])
                use_stacking = False
                best_name = best_single
                pred = None
                base_models_for_checkpoint = []
                meta_for_checkpoint = None
        else:
            use_stacking = False

    if pred is None and best_name:
        # Stacking 실패 시 또는 단일 모델 경로
        if best_name == "lgb":
            import lightgbm as lgb
            from feature_selection.device_utils import fit_lgb_with_fallback
            final_model = lgb.LGBMRegressor(**_lgb_params(config, use_gpu))
            final_model = fit_lgb_with_fallback(final_model, X_tr, y_tr, "gpu" if use_gpu else "cpu")
            base_models_for_checkpoint.append(final_model)
            pred = final_model.predict(X_te) if valid_te.sum() > 0 else None
        elif best_name == "catboost":
            import catboost as cb
            try:
                final_model = cb.CatBoostRegressor(**_cb_params(config, use_gpu))
                final_model.fit(X_tr, y_tr)
            except Exception as e:
                if use_gpu and ("gpu" in str(e).lower() or "cuda" in str(e).lower() or "device" in str(e).lower()):
                    logger.warning("CatBoost GPU 실패, CPU fallback")
                    final_model = cb.CatBoostRegressor(**_cb_params(config, False))
                    final_model.fit(X_tr, y_tr)
                else:
                    raise
            base_models_for_checkpoint.append(final_model)
            pred = final_model.predict(X_te) if valid_te.sum() > 0 else None
        elif best_name == "elasticnet":
            from sklearn.linear_model import ElasticNet
            final_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
            final_model.fit(X_tr, y_tr)
            base_models_for_checkpoint.append(final_model)
            pred = final_model.predict(X_te) if valid_te.sum() > 0 else None
        elif best_name == "lstm":
            from training.deep_models import LSTMWrapper, _get_lag_sequence_cols, build_sequence_array
            seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("lstm", {}).get("seq_len", 365))
            if len(seq_cols) >= 2:
                cfg = config.get("lstm", {})
                X_seq_tr = build_sequence_array(train_full, seq_cols)
                X_seq_te = build_sequence_array(test_df, seq_cols)
                n = len(X_seq_tr)
                if n > 100:
                    split = int(n * 0.8)
                    X_tr_sub, X_val_sub = X_seq_tr[:split], X_seq_tr[split:]
                    y_tr_sub, y_val_sub = y_tr.values[:split], y_tr.values[split:]
                    final_model = LSTMWrapper(seq_len=len(seq_cols), hidden=cfg.get("hidden", 64), layers=cfg.get("layers", 2), epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 10), dropout=cfg.get("dropout", 0.2))
                    final_model.fit(X_tr_sub, y_tr_sub, X_val=X_val_sub, y_val=y_val_sub)
                else:
                    final_model = LSTMWrapper(seq_len=len(seq_cols), hidden=cfg.get("hidden", 64), layers=cfg.get("layers", 2), epochs=cfg.get("epochs", 100), dropout=cfg.get("dropout", 0.2))
                    final_model.fit(X_seq_tr, y_tr.values)
                base_models_for_checkpoint.append(final_model)
                seq_cols_for_checkpoint = seq_cols
                pred = final_model.predict(X_seq_te) if valid_te.sum() > 0 else None
            else:
                pred = None
        elif best_name == "transformer":
            from training.deep_models import TransformerWrapper, _get_lag_sequence_cols, build_sequence_array
            seq_cols = _get_lag_sequence_cols(features, max_lag=config.get("transformer", {}).get("seq_len", 365))
            if len(seq_cols) >= 2:
                cfg = config.get("transformer", {})
                X_seq_tr = build_sequence_array(train_full, seq_cols)
                X_seq_te = build_sequence_array(test_df, seq_cols)
                n = len(X_seq_tr)
                if n > 100:
                    split = int(n * 0.8)
                    X_tr_sub, X_val_sub = X_seq_tr[:split], X_seq_tr[split:]
                    y_tr_sub, y_val_sub = y_tr.values[:split], y_tr.values[split:]
                    final_model = TransformerWrapper(seq_len=len(seq_cols), d_model=cfg.get("d_model", 32), nhead=cfg.get("nhead", 4), num_layers=cfg.get("num_layers", 2), epochs=cfg.get("epochs", 100), patience=cfg.get("patience", 10), dropout=cfg.get("dropout", 0.2))
                    final_model.fit(X_tr_sub, y_tr_sub, X_val=X_val_sub, y_val=y_val_sub)
                else:
                    final_model = TransformerWrapper(seq_len=len(seq_cols), d_model=cfg.get("d_model", 32), nhead=cfg.get("nhead", 4), num_layers=cfg.get("num_layers", 2), epochs=cfg.get("epochs", 100), dropout=cfg.get("dropout", 0.2))
                    final_model.fit(X_seq_tr, y_tr.values)
                base_models_for_checkpoint.append(final_model)
                seq_cols_for_checkpoint = seq_cols
                pred = final_model.predict(X_seq_te) if valid_te.sum() > 0 else None
            else:
                pred = None

    if pred is not None and valid_te.sum() > 0:
        pred_orig = np.expm1(pred) if use_log else pred
        valid_pred = np.isfinite(pred_orig)
        valid_final = valid_te & valid_pred
        if valid_final.sum() < 5:
            logger.warning("유효한 예측 샘플 부족 (%d), 메트릭 스킵", valid_final.sum())
        else:
            from training.metrics import compute_all_metrics
            lag1_col = "price_per_kg_mean_lag1"
            y_naive = None
            if lag1_col in test_df.columns:
                nv = test_df[lag1_col].values[valid_final]
                if np.isfinite(nv).all():
                    y_naive = nv
            metrics = compute_all_metrics(y_te[valid_final], pred_orig[valid_final], y_naive=y_naive)
            results["test_metrics"] = metrics
            results["test_mape"] = metrics["mape"]
            for k, v in metrics.items():
                logger.info("Test %s: %.4f", k, v)
            if "mape_log" in metrics:
                logger.info("Test MAPE (log scale): %.4f%%", metrics["mape_log"])

            # Per-season MAPE
            test_df = test_df.copy()
            test_df["_pred"] = pred_orig
            test_df["_month"] = pd.to_datetime(test_df["date"].astype(str), format="%Y%m%d").dt.month
            test_df["_season"] = test_df["_month"].apply(_month_to_season)
            season_mape = {}
            for s in SEASON_MONTHS:
                mask = (test_df["_season"] == s) & test_df[TARGET_COL].notna() & np.isfinite(test_df["_pred"])
                if mask.sum() > 0:
                    season_mape[s] = mape(
                        test_df.loc[mask, TARGET_COL].values,
                        test_df.loc[mask, "_pred"].values,
                    )
            results["season_mape"] = season_mape
            for s, m in season_mape.items():
                logger.info("  %s MAPE: %.2f%%", s, m)

            # 계절별 예측 사례 ~10개 (날짜, 실제가격, 예측가격)
            examples = []
            n_per_season = max(2, 10 // len(SEASON_MONTHS))
            for s in SEASON_MONTHS:
                mask = (test_df["_season"] == s) & test_df[TARGET_COL].notna() & np.isfinite(test_df["_pred"])
                sub = test_df.loc[mask, ["date", TARGET_COL, "_pred"]].drop_duplicates("date")
                if len(sub) == 0:
                    continue
                n_take = min(n_per_season, len(sub))
                sampled = sub.sample(n=n_take, random_state=RANDOM_STATE)
                for _, row in sampled.iterrows():
                    d = str(int(row["date"]))
                    date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 else d
                    examples.append({
                        "date": date_str,
                        "season": s,
                        "actual": float(row[TARGET_COL]),
                        "predicted": float(row["_pred"]),
                    })
            examples = sorted(examples, key=lambda x: (list(SEASON_MONTHS).index(x["season"]), x["date"]))[:10]
            results["prediction_examples"] = examples
            logger.info("Test prediction examples (by season):")
            for ex in examples:
                err = abs(ex["actual"] - ex["predicted"]) / (ex["actual"] + 1e-8) * 100
                logger.info("  %s %s | actual %.0f KRW/kg -> predicted %.0f KRW/kg (error %.1f%%)", ex["date"], ex["season"], ex["actual"], ex["predicted"], err)

    results["best_model"] = best_name
    results["use_stacking"] = use_stacking
    results["n_features"] = len(features)
    results["test_dates"] = len(test_dates)

    # Save best model checkpoint for serving/inference
    if pred is not None and (base_models_for_checkpoint or meta_for_checkpoint):
        try:
            import joblib
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            base_names = list(best_combo) if use_stacking and best_combo else [best_name]
            rep_varieties = config.get("variety", {}).get("representative_varieties") or []
            if not rep_varieties:
                try:
                    import yaml
                    fc = PROJECT_ROOT / "config" / "feature_selection_config.yaml"
                    if fc.exists():
                        with open(fc, encoding="utf-8") as f:
                            rep_varieties = (yaml.safe_load(f) or {}).get("representative_varieties") or []
                except Exception:
                    pass
            ckpt = {
                "use_stacking": use_stacking,
                "best_name": best_name,
                "meta": meta_for_checkpoint,
                "base_models": base_models_for_checkpoint,
                "base_names": base_names,
                "features": features,
                "use_log": use_log,
                "seq_cols": seq_cols_for_checkpoint,
                "train_dates": sorted(train_df["date"].dropna().unique().tolist()),
                "test_dates": test_dates,
                "variety_filter": rep_varieties,
            }
            ckpt_path = CHECKPOINT_DIR / "checkpoint.joblib"
            joblib.dump(ckpt, ckpt_path)
            logger.info("Best model checkpoint saved: %s", ckpt_path)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", str(e)[:80])

    # Report
    report_path = PROJECT_ROOT / "reports" / "training_pipeline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Pipeline Report",
        "",
        "## Target Variety",
        f"**{results.get('variety', '-')}**",
        "",
        "### Sample Count by Variety",
        "| Variety | Count |",
        "|---------|-------|",
    ]
    for v, c in results.get("variety_counts", {}).items():
        lines.append(f"| {v} | {c} |")
    if not results.get("variety_counts"):
        lines.append("| (all varieties) | - |")
    lines.extend([
        "",
        "## Baseline",
        "| Model | MAPE (%) |",
        "|-------|----------|",
    ])
    for k, v in baseline_results.items():
        lines.append(f"| {k} | {v:.2f} |")
    lines.extend([
        "",
        "## CV Scores",
        "| Model | Mean | Std | Worst |",
        "|-------|------|-----|-------|",
    ])
    for k, v in model_scores.items():
        lines.append(f"| {k} | {v['mean']:.2f} | {v['std']:.2f} | {v['worst']:.2f} |")
    lines.extend([
        "",
        f"## CV Method: {config.get('cv', {}).get('method', 'expanding')}",
        "",
        f"## Best Model: {best_name}",
        "",
        "## Test Metrics",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    for k, v in results.get("test_metrics", {}).items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append("*(mape: raw price scale, mape_log: MAPE on log1p-transformed)*")
    if "test_metrics" not in results and isinstance(results.get("test_mape"), (int, float)):
        lines.append(f"| mape | {results['test_mape']:.4f} |")
    lines.extend([
        "",
        "## Season MAPE",
        "| Season | MAPE (%) |",
        "|--------|----------|",
    ])
    for s, m in results.get("season_mape", {}).items():
        lines.append(f"| {s} | {m:.2f} |")
    ex_list = results.get("prediction_examples", [])
    lines.extend([
        "",
        "## Test Prediction Examples (by Season)",
        "| Date | Season | Actual (KRW/kg) | Predicted (KRW/kg) | Error (%) |",
        "|------|--------|-----------------|--------------------|-----------|",
    ])
    for ex in ex_list:
        err = abs(ex["actual"] - ex["predicted"]) / (ex["actual"] + 1e-8) * 100
        lines.append(f"| {ex['date']} | {ex['season']} | {ex['actual']:,.0f} | {ex['predicted']:,.0f} | {err:.1f}% |")
    if not ex_list:
        lines.append("| *(no valid predictions)* | | | | |")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", report_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Split 고정용 seed (기본: config)")
    parser.add_argument("--skip-feature-refine", action="store_true")
    parser.add_argument("--skip-shap", action="store_true", help="SHAP feature refinement 스킵 (기본)")
    parser.add_argument("--model", type=str, nargs="+", default=None, help="실행할 모델만 (lgb, catboost, elasticnet, lstm, transformer)")
    parser.add_argument("--cv", type=str, default=None, help="CV 방법: expanding, timeseries, purged")
    parser.add_argument("--no-deep", action="store_true", help="LSTM/Transformer 스킵")
    parser.add_argument("--plot", action="store_true", help="학습 후 inference + 시계열 플롯 생성")
    args = parser.parse_args()

    results = run_pipeline(
        data_path=Path(args.data_path) if args.data_path else None,
        config_path=Path(args.config) if args.config else None,
        seed=args.seed,
        skip_feature_refine=args.skip_feature_refine,
        skip_shap=args.skip_shap,
        models=args.model,
        cv_method=args.cv,
        no_deep=args.no_deep,
    )
    if args.plot and results.get("best_model"):
        try:
            from run_inference_and_plot import run_inference_and_plot
            run_inference_and_plot()
        except Exception as e:
            logger.warning("Inference plot failed: %s", str(e)[:80])


if __name__ == "__main__":
    main()
