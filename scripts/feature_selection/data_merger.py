#!/usr/bin/env python3
"""
Stage 0: Maximal lag feature generation.

Target: 1kg당 경락가 (단량 파싱 후 단량당 경락가/kg → 일별 품종별 평균)
- target price lag: 1~60일 (autoregressive)
- 도매/소매 lag: 1~30일 (당일 경매 후 수집)
- 날씨 lag: 1~21일
- rolling mean: 7, 14, 30일

⚠️ 모든 품종에 동일한 lag/feature 적용 (품종별 차이 없음)
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config_loader import load_config

logger = logging.getLogger(__name__)

TARGET_RAW = "단량당 경락가(원)"
VARIETY_COL_AUCTION = "품종"
VARIETY_COL_DOMAE_SOMAE = "SPCIES_NM"

DEFAULT_LAG = {
    "target_price": 60,
    "domae_somae": 30,
    "weather": 21,
}
ROLLING_WINDOWS = [7, 14, 30]


def parse_kg_from_danryang(s: str) -> Optional[float]:
    """단량 문자열에서 kg 추출."""
    if pd.isna(s) or not str(s).strip():
        return None
    m = re.search(r"(\d*\.?\d+)\s*kg", str(s).strip(), re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            return val if val > 0 else None
        except ValueError:
            return None
    return None


class DataMerger:
    """Maximal lag feature generation. 품종 무관 동일 스키마."""

    def __init__(
        self,
        data_root: Path,
        start: str = None,
        end: str = None,
        config: Optional[dict] = None,
    ):
        self.data_root = Path(data_root)
        self._config = config or load_config()
        dr = self._config.get("data_range", {})
        self.start = start or dr.get("start", "20200101")
        self.end = end or dr.get("end", "20251231")
        lag = self._config.get("lag_depths", DEFAULT_LAG)
        self.target_lag = lag.get("target_price", 60)
        self.domae_somae_lag = lag.get("domae_somae", 30)
        self.weather_lag = lag.get("weather", 21)
        self.rolling_windows = self._config.get("rolling_windows", ROLLING_WINDOWS)

    def _normalize_date(self, s: pd.Series) -> pd.Series:
        return s.astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)

    def load_auction(self) -> pd.DataFrame:
        path = self.data_root / "auction" / f"auction_{self.start}_{self.end}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = self._normalize_date(df["경락일시"])
        valid = df["품종"].notna() & df[TARGET_RAW].notna()
        df = df[valid].copy()
        df[TARGET_RAW] = pd.to_numeric(df[TARGET_RAW], errors="coerce")
        df["수량"] = pd.to_numeric(df["수량"], errors="coerce")
        df = df[df[TARGET_RAW].notna()].copy()
        df["kg"] = df["단량"].apply(parse_kg_from_danryang)
        df["price_per_kg"] = np.where(
            df["kg"].notna() & (df["kg"] > 0),
            df[TARGET_RAW] / df["kg"],
            np.nan,
        )
        df = df[df["price_per_kg"].notna()].copy()
        logger.info("auction: %d행 (1kg당 가격 환산)", len(df))
        return df

    def load_domae(self) -> pd.DataFrame:
        path = self.data_root / "domae" / f"domae_{self.start}_{self.end}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = self._normalize_date(df["EXAMIN_DE"])
        df["AMT"] = pd.to_numeric(df["AMT"], errors="coerce")
        df = df[df["AMT"].notna()].copy()
        return df

    def load_somae(self) -> pd.DataFrame:
        path = self.data_root / "somae" / f"somae_{self.start}_{self.end}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = self._normalize_date(df["EXAMIN_DE"])
        df["AMT"] = pd.to_numeric(df["AMT"], errors="coerce")
        df = df[df["AMT"].notna()].copy()
        return df

    def load_weather(self) -> pd.DataFrame:
        path = self.data_root / "weather" / f"weather_{self.start}_{self.end}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = self._normalize_date(df["TM"])
        return df

    def _fill_low_tx_with_week_median(self, df: pd.DataFrame, min_tx: int) -> pd.DataFrame:
        """거래건수 부족 일자: 앞뒤 ±7일 동일 품종 median으로 채움 (merge_config.low_tx=fill_week)."""
        df = df.copy()
        df["_date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
        if (df["auction_transaction_count"] < min_tx).sum() == 0:
            return df.drop(columns=["_date_dt"], errors="ignore")
        filled = []
        for variety, group in df.groupby("품종"):
            g = group.sort_values("_date_dt").reset_index(drop=True)
            for i in range(len(g)):
                row = g.iloc[i]
                if row["auction_transaction_count"] >= min_tx:
                    filled.append(row.to_dict())
                    continue
                lo = row["_date_dt"] - pd.Timedelta(days=7)
                hi = row["_date_dt"] + pd.Timedelta(days=7)
                window = g[(g["_date_dt"] >= lo) & (g["_date_dt"] <= hi) & (g["auction_transaction_count"] >= min_tx)]
                if len(window) > 0:
                    r = row.to_dict()
                    r["price_per_kg_mean"] = window["price_per_kg_mean"].median()
                    filled.append(r)
                else:
                    filled.append(row.to_dict())
        return pd.DataFrame(filled).drop(columns=["_date_dt"], errors="ignore")

    def aggregate_auction(self, df: pd.DataFrame) -> pd.DataFrame:
        merge_cfg = self._config.get("merge_config", {})
        agg_method = merge_cfg.get("auction_agg", "mean")
        agg_map = {"mean": "mean", "median": "median"}
        agg_func = agg_map.get(agg_method, "mean")

        agg = (
            df.groupby(["date", VARIETY_COL_AUCTION], as_index=False)
            .agg(
                price_per_kg_mean=("price_per_kg", agg_func),
                price_per_kg_median=("price_per_kg", "median"),
                price_per_kg_std=("price_per_kg", "std"),
                auction_quantity_sum=("수량", "sum"),
                auction_quantity_mean=("수량", "mean"),
                auction_transaction_count=("price_per_kg", "count"),
            )
            .rename(columns={VARIETY_COL_AUCTION: "품종"})
        )

        min_tx = merge_cfg.get("min_tx_count", 5)
        low_tx = merge_cfg.get("low_tx", "exclude")
        if low_tx == "exclude":
            before = len(agg)
            agg = agg[agg["auction_transaction_count"] >= min_tx].copy()
            logger.info("Low tx exclude: %d -> %d rows (min_tx=%d)", before, len(agg), min_tx)
        elif low_tx == "fill_week":
            agg = self._fill_low_tx_with_week_median(agg, min_tx)
            logger.info("Low tx fill_week 적용 (min_tx=%d)", min_tx)

        return agg

    def _aggregate_domae_somae(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        agg = (
            df.groupby(["date", VARIETY_COL_DOMAE_SOMAE], as_index=False)
            .agg(
                amt_mean=("AMT", "mean"),
                amt_median=("AMT", "median"),
                amt_std=("AMT", "std"),
                market_count=("AMT", "count"),
            )
            .rename(columns={VARIETY_COL_DOMAE_SOMAE: "품종"})
        )
        agg.columns = [f"{prefix}_{c}" if c not in ("date", "품종") else c for c in agg.columns]
        return agg

    def aggregate_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "STN"]
        agg = df.groupby("date", as_index=False)[numeric_cols].mean()
        agg = agg.rename(columns={c: f"weather_{c}" for c in numeric_cols})
        return agg

    def _add_lag_merge(
        self,
        base: pd.DataFrame,
        src: pd.DataFrame,
        lag_max: int,
        join_on: Tuple[str, ...],
    ) -> pd.DataFrame:
        """merge 방식으로 lag 추가 (date+품종)."""
        value_cols = [c for c in src.columns if c not in join_on]
        merged = base.copy()

        for lag in range(1, lag_max + 1):
            lag_src = src.copy()
            lag_src["_join_date"] = (
                pd.to_datetime(lag_src["date"], format="%Y%m%d") + pd.Timedelta(days=lag)
            ).dt.strftime("%Y%m%d")
            rename_map = {c: f"{c}_lag{lag}" for c in value_cols}
            lag_src = lag_src.rename(columns=rename_map)
            merge_cols = ["_join_date", "품종"] + [f"{c}_lag{lag}" for c in value_cols]
            merged = merged.merge(
                lag_src[merge_cols],
                left_on=["date", "품종"],
                right_on=["_join_date", "품종"],
                how="left",
            ).drop(columns=["_join_date"], errors="ignore")
        return merged

    def _add_weather_lag(self, base: pd.DataFrame, weather: pd.DataFrame, lag_max: int) -> pd.DataFrame:
        """날씨는 date만 키 (품종 무관)."""
        value_cols = [c for c in weather.columns if c != "date"]
        merged = base.copy()

        for lag in range(1, lag_max + 1):
            lag_w = weather.copy()
            lag_w["_join_date"] = (
                pd.to_datetime(lag_w["date"], format="%Y%m%d") + pd.Timedelta(days=lag)
            ).dt.strftime("%Y%m%d")
            rename_map = {c: f"{c}_lag{lag}" for c in value_cols}
            lag_w = lag_w.rename(columns=rename_map)
            merge_cols = ["_join_date"] + [f"{c}_lag{lag}" for c in value_cols]
            merged = merged.merge(
                lag_w[merge_cols],
                left_on="date",
                right_on="_join_date",
                how="left",
            ).drop(columns=["_join_date"], errors="ignore")
        return merged

    def run(self) -> pd.DataFrame:
        auction = self.load_auction()
        if auction.empty:
            raise ValueError("auction 데이터 없음")

        auction_agg = self.aggregate_auction(auction)

        # Target lag (autoregressive): price_per_kg_mean lag 1~60
        target_src = auction_agg[["date", "품종", "price_per_kg_mean"]].copy()
        merged = self._add_lag_merge(
            auction_agg,
            target_src,
            min(self.target_lag, 60),
            ("date", "품종"),
        )

        # Seasonal lags (Year-over-Year): 364, 365, 366 - 별도 관리
        seasonal_cfg = self._config.get("seasonal_lags", {})
        if seasonal_cfg.get("enabled", True):
            seasonal_list = seasonal_cfg.get("lags", [364, 365, 366])
            for lag_d in seasonal_list:
                if lag_d <= 60:
                    continue  # 이미 target_lag에 포함
                lag_src = target_src.copy()
                lag_src["_join_date"] = (
                    pd.to_datetime(lag_src["date"], format="%Y%m%d") + pd.Timedelta(days=lag_d)
                ).dt.strftime("%Y%m%d")
                lag_src = lag_src.rename(columns={"price_per_kg_mean": f"price_per_kg_mean_lag{lag_d}"})
                merged = merged.merge(
                    lag_src[["_join_date", "품종", f"price_per_kg_mean_lag{lag_d}"]],
                    left_on=["date", "품종"],
                    right_on=["_join_date", "품종"],
                    how="left",
                ).drop(columns=["_join_date"], errors="ignore")
            logger.info("Seasonal lags 추가: %s", seasonal_list)

        domae = self.load_domae()
        if not domae.empty:
            merge_cfg = self._config.get("merge_config", {})
            if merge_cfg.get("domae_filter") == "garak" and "MRKT_NM" in domae.columns:
                domae = domae[domae["MRKT_NM"] == "가락도매"].copy()
                logger.info("Domae filtered: 가락도매 only (%d rows)", len(domae))
        domae_agg = self._aggregate_domae_somae(domae, "domae") if not domae.empty else None
        if domae_agg is not None:
            merged = self._add_lag_merge(merged, domae_agg, self.domae_somae_lag, ("date", "품종"))

        somae = self.load_somae()
        somae_agg = self._aggregate_domae_somae(somae, "somae") if not somae.empty else None
        if somae_agg is not None:
            merged = self._add_lag_merge(merged, somae_agg, self.domae_somae_lag, ("date", "품종"))

        weather = self.load_weather()
        weather_agg = self.aggregate_weather(weather) if not weather.empty else None
        if weather_agg is not None:
            merged = self._add_weather_lag(merged, weather_agg, self.weather_lag)

        # Rolling means (key variables)
        for base_col, max_lag in [
            ("domae_amt_mean", min(30, self.domae_somae_lag)),
            ("somae_amt_mean", min(30, self.domae_somae_lag)),
        ]:
            if f"{base_col}_lag1" in merged.columns:
                for w in self.rolling_windows:
                    if w <= max_lag:
                        lag_cols = [f"{base_col}_lag{k}" for k in range(1, w + 1)]
                        if all(c in merged.columns for c in lag_cols):
                            merged[f"{base_col}_rolling{w}"] = merged[lag_cols].mean(axis=1)

        # Weather rolling (TA_AVG 등)
        if "weather_TA_AVG_lag1" in merged.columns:
            for w in self.rolling_windows:
                if w <= self.weather_lag:
                    lag_cols = [f"weather_TA_AVG_lag{k}" for k in range(1, w + 1)]
                    if all(c in merged.columns for c in lag_cols):
                        merged[f"weather_TA_AVG_rolling{w}"] = merged[lag_cols].mean(axis=1)

        # Date features
        merged["date_parsed"] = pd.to_datetime(merged["date"], format="%Y%m%d", errors="coerce")
        merged["date_year"] = merged["date_parsed"].dt.year
        merged["date_month"] = merged["date_parsed"].dt.month
        merged["date_day"] = merged["date_parsed"].dt.day
        merged["date_dayofweek"] = merged["date_parsed"].dt.dayofweek
        merged = merged.drop(columns=["date_parsed"])

        logger.info("Stage 0 완료: %d행, %d컬럼", len(merged), len(merged.columns))
        return merged
