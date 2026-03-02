#!/usr/bin/env python3
"""
컬럼 프로파일링: 컬럼별 5개 샘플, 타입, cardinality, 의미없는 컬럼 필터링.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "auction_price_mean"  # merged 데이터에서 target 이름
EXCLUDE_FROM_FEATURES = {
    "date",
    "품종",
    "price_per_kg_mean",  # target
    "price_per_kg_median",
    "price_per_kg_std",
    "weather_TM",  # date와 중복 (YYYYMMDD)
}


class ColumnProfiler:
    """컬럼별 프로파일 및 후보 선정."""

    def __init__(
        self,
        n_samples: int = 5,
        high_cardinality_threshold: int = 100,
        low_variance_threshold: float = 1e-6,
    ):
        self.n_samples = n_samples
        self.high_cardinality_threshold = high_cardinality_threshold
        self.low_variance_threshold = low_variance_threshold

    def get_samples(self, series: pd.Series, n: int = 5) -> List[Any]:
        """컬럼별 대표 샘플 n개 (고유값 우선, 결측 제외)."""
        valid = series.dropna()
        if valid.empty:
            return [None] * n
        unique = valid.unique()
        if len(unique) <= n:
            samples = list(unique)
        else:
            # 다양성: 앞, 뒤, 중간에서 골고루
            idx = np.linspace(0, len(unique) - 1, n, dtype=int)
            samples = [unique[i] for i in idx]
        return samples[:n]

    def profile_column(
        self,
        name: str,
        series: pd.Series,
    ) -> Dict[str, Any]:
        """단일 컬럼 프로파일."""
        dtype = str(series.dtype)
        n_total = len(series)
        n_null = series.isna().sum()
        n_unique = series.nunique()

        profile = {
            "column": name,
            "dtype": dtype,
            "n_total": n_total,
            "n_null": int(n_null),
            "null_pct": round(n_null / n_total * 100, 2) if n_total > 0 else 0,
            "n_unique": n_unique,
            "samples": self.get_samples(series, self.n_samples),
        }

        if pd.api.types.is_numeric_dtype(series):
            profile["min"] = float(series.min()) if series.notna().any() else None
            profile["max"] = float(series.max()) if series.notna().any() else None
            profile["mean"] = float(series.mean()) if series.notna().any() else None
            profile["std"] = float(series.std()) if series.notna().any() else 0.0
            profile["is_numeric"] = True
        else:
            profile["is_numeric"] = False

        return profile

    def classify_column(self, profile: Dict[str, Any]) -> str:
        """
        컬럼 분류: numeric_ok | categorical_encodable | high_cardinality | constant | meaningless
        """
        if profile["n_unique"] <= 1:
            return "constant"
        if profile["null_pct"] >= 99:
            return "meaningless"
        if profile["is_numeric"]:
            if profile.get("std", 0) is not None and profile.get("std", 0) < self.low_variance_threshold:
                return "constant"
            return "numeric_ok"
        if profile["n_unique"] <= self.high_cardinality_threshold:
            return "categorical_encodable"
        return "high_cardinality"

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """
        전체 DataFrame 프로파일.
        Returns: (profiles, candidate_columns, drop_columns)
        """
        exclude = set(exclude_cols or []) | EXCLUDE_FROM_FEATURES
        profiles = []
        candidates = []
        drops = []

        for col in df.columns:
            if col in exclude:
                continue
            profile = self.profile_column(col, df[col])
            profile["classification"] = self.classify_column(profile)
            profiles.append(profile)

            if profile["classification"] in ("constant", "meaningless", "high_cardinality"):
                drops.append(col)
            else:
                candidates.append(col)

        return profiles, candidates, drops

    def to_markdown_table(self, profiles: List[Dict]) -> str:
        """프로파일을 마크다운 테이블로 출력."""
        lines = []
        lines.append("| 컬럼 | dtype | n_unique | null% | 분류 | 샘플(5개) |")
        lines.append("|------|-------|----------|-------|------|-----------|")

        for p in profiles:
            samples_str = ", ".join(str(s)[:20] for s in p["samples"])
            if len(samples_str) > 60:
                samples_str = samples_str[:57] + "..."
            lines.append(
                f"| {p['column']} | {p['dtype']} | {p['n_unique']} | {p['null_pct']}% | "
                f"{p['classification']} | {samples_str} |"
            )
        return "\n".join(lines)
