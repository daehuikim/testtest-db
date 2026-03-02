#!/usr/bin/env python3
"""
Feature selection: Gini(RF), Mutual Information, Correlation 기반 중요도 산출.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET_COL = "price_per_kg_mean"
RANDOM_STATE = 42


class FeatureSelector:
    """다중 기법 기반 feature importance 산출."""

    def __init__(
        self,
        target_col: str = TARGET_COL,
        random_state: int = RANDOM_STATE,
    ):
        self.target_col = target_col
        self.random_state = random_state

    def prepare_Xy(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        impute: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """X, y 분리. impute=True면 결측을 컬럼 중앙값으로 대체."""
        use_cols = [c for c in feature_cols if c in df.columns]
        X = df[use_cols].copy()
        y = df[self.target_col]

        # target 결측 행 제거
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        if impute:
            for c in X.select_dtypes(include=[np.number]).columns:
                if X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())
            # 남은 결측 (비수치형 등)은 0
            X = X.fillna(0)

        return X, y

    def gini_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        n_bins: int = 5,
    ) -> pd.Series:
        """
        Gini 기반 feature importance.
        연속형 target을 구간화(quantile binning) 후 RandomForestClassifier(criterion='gini') 사용.
        Regressor는 MSE 기반이므로, 진정한 지니계수는 Classifier에서만 산출됨.
        """
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        if y_binned.nunique() < 2:
            logger.warning("구간화 후 클래스 수 부족, Gini 산출 불가")
            return pd.Series(dtype=float)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            criterion="gini",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y_binned)
        imp = pd.Series(rf.feature_importances_, index=X.columns)
        return imp.sort_values(ascending=False)

    def mutual_info_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Mutual Information 기반 중요도."""
        # 수치형만 (MI는 수치형 가정)
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty:
            return pd.Series(dtype=float)
        mi = mutual_info_regression(X_num, y, random_state=self.random_state)
        imp = pd.Series(mi, index=X_num.columns)
        return imp.sort_values(ascending=False)

    def correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Target과의 절대 상관계수."""
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty:
            return pd.Series(dtype=float)
        corr = X_num.corrwith(y).abs()
        return corr.sort_values(ascending=False)

    def run_all(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        variety: Optional[str] = None,
    ) -> Dict[str, pd.Series]:
        """품종 필터 적용 후 모든 기법 실행."""
        if variety:
            sub = df[df["품종"] == variety].copy()
            logger.info("품종 '%s' 필터: %d행", variety, len(sub))
        else:
            sub = df.copy()

        X, y = self.prepare_Xy(sub, feature_cols)
        if len(X) < 10:
            logger.warning("샘플 수 부족: %d", len(X))
            return {}

        results = {}
        results["gini"] = self.gini_importance(X, y)
        results["mutual_info"] = self.mutual_info_importance(X, y)
        results["correlation"] = self.correlation_importance(X, y)
        return results

    def rank_aggregate(
        self,
        results: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """기법별 순위를 가중 평균하여 통합 순위."""
        if not results:
            return pd.Series(dtype=float)
        weights = weights or {"gini": 0.4, "mutual_info": 0.35, "correlation": 0.25}

        all_features = set()
        for s in results.values():
            all_features.update(s.index)
        all_features = list(all_features)

        rank_scores = {}
        for feat in all_features:
            score = 0.0
            for method, w in weights.items():
                if method in results and feat in results[method].index:
                    # 순위 기반 (1등=1, 2등=2, ...) → 역순으로 점수 (높을수록 좋음)
                    rank = (results[method].index.get_loc(feat) + 1)
                    score += w * (1.0 / rank)  # 1/rank: 1등이 가장 높은 점수
            rank_scores[feat] = score
        agg = pd.Series(rank_scores).sort_values(ascending=False)
        return agg
