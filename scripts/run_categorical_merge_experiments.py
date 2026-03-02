#!/usr/bin/env python3
"""
카테고리컬 보정 실험 스크립트

실험 차원:
1. auction target: mean vs median (학습 target, feature selection/inference sweet spot)
2. domae filter: all vs 가락도매 only (경매 가락시장과 동일 시장 정렬)
3. low transaction: exclude vs fill_week (거래건수 부족 일자 처리)
4. min_tx_count: 최소 거래건수 (5 등)

출력: 보고서, 플롯, 실험별 비교
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

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "temp"
REPORT_DIR = PROJECT_ROOT / "reports"
EXP_OUTPUT = TEMP_DIR / "categorical_experiments"


def analyze_categorical_columns() -> Dict[str, Any]:
    """auction, domae, somae에서 카테고리컬 컬럼 분석."""
    result = {"auction": {}, "domae": {}, "somae": {}}

    # Auction
    auction_path = DATA_ROOT / "auction"
    auction_files = list(auction_path.glob("auction_*.csv")) if auction_path.exists() else []
    if auction_files:
        df = pd.read_csv(auction_files[0], encoding="utf-8-sig", nrows=5000)
        for col in df.select_dtypes(include=["object"]).columns:
            uniq = df[col].dropna().unique()
            samples = (uniq[:10].tolist() if hasattr(uniq, "tolist") else list(uniq)[:10])
            result["auction"][col] = {
                "dtype": "object",
                "n_unique": int(len(uniq)),
                "sample_values": [str(v) for v in samples],
            }

    # Domae
    domae_path = DATA_ROOT / "domae"
    domae_files = list(domae_path.glob("domae_*.csv")) if domae_path.exists() else []
    if domae_files:
        df = pd.read_csv(domae_files[0], encoding="utf-8-sig", nrows=10000)
        for col in df.select_dtypes(include=["object"]).columns:
            uniq = df[col].dropna().unique()
            samples = (uniq[:10].tolist() if hasattr(uniq, "tolist") else list(uniq)[:10])
            result["domae"][col] = {
                "dtype": "object",
                "n_unique": int(len(uniq)),
                "sample_values": [str(v) for v in samples],
            }
        if "MRKT_NM" in df.columns:
            mrkt_counts = df["MRKT_NM"].value_counts()
            result["domae"]["MRKT_NM_counts"] = {str(k): int(v) for k, v in mrkt_counts.head(15).items()}

    # Somae
    somae_path = DATA_ROOT / "somae"
    somae_files = list(somae_path.glob("somae_*.csv")) if somae_path.exists() else []
    if somae_files:
        df = pd.read_csv(somae_files[0], encoding="utf-8-sig", nrows=5000)
        for col in df.select_dtypes(include=["object"]).columns:
            uniq = df[col].dropna().unique()
            samples = (uniq[:10].tolist() if hasattr(uniq, "tolist") else list(uniq)[:10])
            result["somae"][col] = {
                "dtype": "object",
                "n_unique": int(len(uniq)),
                "sample_values": [str(v) for v in samples],
            }

    return result


def run_merge_with_config(
    merge_config: dict,
    data_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    """merge_config에 따라 데이터 병합. DataMerger 확장 로직."""
    from feature_selection.config_loader import load_config
    from feature_selection.data_merger import (
        DataMerger,
        parse_kg_from_danryang,
        DEFAULT_LAG,
        ROLLING_WINDOWS,
        TARGET_RAW,
        VARIETY_COL_AUCTION,
        VARIETY_COL_DOMAE_SOMAE,
    )

    config = load_config()
    merger = DataMerger(data_root=data_root, config=config)

    # Load raw
    auction = merger.load_auction()
    if auction.empty:
        raise ValueError("auction 데이터 없음")

    # Auction aggregation: mean vs median as target
    agg_method = merge_config.get("auction_agg", "mean")
    agg_map = {
        "mean": ("price_per_kg", "mean"),
        "median": ("price_per_kg", "median"),
    }
    agg_func = agg_map.get(agg_method, agg_map["mean"])

    auction_agg = (
        auction.groupby(["date", VARIETY_COL_AUCTION], as_index=False)
        .agg(
            price_per_kg_mean=(agg_func[0], agg_func[1]),  # target (항상 price_per_kg_mean 이름)
            price_per_kg_median=("price_per_kg", "median"),
            price_per_kg_std=("price_per_kg", "std"),
            auction_quantity_sum=("수량", "sum"),
            auction_quantity_mean=("수량", "mean"),
            auction_transaction_count=("price_per_kg", "count"),
        )
        .rename(columns={VARIETY_COL_AUCTION: "품종"})
    )

    # Low transaction handling
    min_tx = merge_config.get("min_tx_count", 5)
    low_tx_mode = merge_config.get("low_tx", "exclude")

    if low_tx_mode == "exclude":
        before = len(auction_agg)
        auction_agg = auction_agg[auction_agg["auction_transaction_count"] >= min_tx].copy()
        logger.info("Low tx exclude: %d -> %d rows (min_tx=%d)", before, len(auction_agg), min_tx)
    elif low_tx_mode == "fill_week":
        # Fill low-tx days with median from ±7 days (same variety)
        auction_agg = _fill_low_tx_with_week_median(auction_agg, min_tx)

    # Build merged (reuse merger's merge logic via run, but we need custom auction_agg)
    # We'll call merger.run() but replace auction_agg - actually merger doesn't support that.
    # So we need to duplicate the merge logic or extend DataMerger.
    # Simpler: create a custom run that uses our auction_agg.
    merged = _run_merge_with_custom_auction(
        merger, auction_agg, merge_config, config
    )

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved: %s (%d rows)", output_path, len(merged))
    return merged


def _fill_low_tx_with_week_median(df: pd.DataFrame, min_tx: int) -> pd.DataFrame:
    """거래건수 부족 일자: 앞뒤 ±7일 동일 품종 median으로 채움."""
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    low_mask = df["auction_transaction_count"] < min_tx

    if not low_mask.any():
        return df.drop(columns=["date_dt"], errors="ignore")

    filled = []
    for variety, group in df.groupby("품종"):
        g = group.sort_values("date_dt").reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            if row["auction_transaction_count"] >= min_tx:
                filled.append(row.to_dict())
                continue
            lo = row["date_dt"] - pd.Timedelta(days=7)
            hi = row["date_dt"] + pd.Timedelta(days=7)
            window = g[(g["date_dt"] >= lo) & (g["date_dt"] <= hi) & (g["auction_transaction_count"] >= min_tx)]
            if len(window) > 0:
                med = window["price_per_kg_mean"].median()
                r = row.to_dict()
                r["price_per_kg_mean"] = med
                filled.append(r)
            else:
                filled.append(row.to_dict())

    out = pd.DataFrame(filled).drop(columns=["date_dt"], errors="ignore")
    return out


def _run_merge_with_custom_auction(merger, auction_agg: pd.DataFrame, merge_config: dict, config: dict) -> pd.DataFrame:
    """Custom auction_agg로 merge 수행."""
    from feature_selection.data_merger import VARIETY_COL_DOMAE_SOMAE

    target_src = auction_agg[["date", "품종", "price_per_kg_mean"]].copy()
    merged = merger._add_lag_merge(
        auction_agg,
        target_src,
        min(merger.target_lag, 60),
        ("date", "품종"),
    )

    # Seasonal lags
    seasonal_cfg = config.get("seasonal_lags", {})
    if seasonal_cfg.get("enabled", True):
        seasonal_list = seasonal_cfg.get("lags", [364, 365, 366])
        for lag_d in seasonal_list:
            if lag_d <= 60:
                continue
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

    # Domae (filter: all vs 가락도매)
    domae = merger.load_domae()
    if not domae.empty:
        domae_filter = merge_config.get("domae_filter", "all")
        if domae_filter == "garak" and "MRKT_NM" in domae.columns:
            domae = domae[domae["MRKT_NM"] == "가락도매"].copy()
            logger.info("Domae filtered: 가락도매 only (%d rows)", len(domae))
        domae_agg = merger._aggregate_domae_somae(domae, "domae")
        merged = merger._add_lag_merge(merged, domae_agg, merger.domae_somae_lag, ("date", "품종"))

    # Somae
    somae = merger.load_somae()
    if not somae.empty:
        somae_agg = merger._aggregate_domae_somae(somae, "somae")
        merged = merger._add_lag_merge(merged, somae_agg, merger.domae_somae_lag, ("date", "품종"))

    # Weather
    weather = merger.load_weather()
    if not weather.empty:
        weather_agg = merger.aggregate_weather(weather)
        merged = merger._add_weather_lag(merged, weather_agg, merger.weather_lag)

    # Rolling
    for base_col, max_lag in [
        ("domae_amt_mean", min(30, merger.domae_somae_lag)),
        ("somae_amt_mean", min(30, merger.domae_somae_lag)),
    ]:
        if f"{base_col}_lag1" in merged.columns:
            for w in merger.rolling_windows:
                if w <= max_lag:
                    lag_cols = [f"{base_col}_lag{k}" for k in range(1, w + 1)]
                    if all(c in merged.columns for c in lag_cols):
                        merged[f"{base_col}_rolling{w}"] = merged[lag_cols].mean(axis=1)

    if "weather_TA_AVG_lag1" in merged.columns:
        for w in merger.rolling_windows:
            if w <= merger.weather_lag:
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

    return merged


def run_training_for_merged(merged_path: Path, exp_id: str) -> Dict[str, Any]:
    """merged 데이터로 학습 파이프라인 실행, 결과 반환."""
    from run_training_pipeline import run_pipeline

    results = run_pipeline(
        data_path=merged_path,
        skip_feature_refine=True,
        skip_shap=True,
        no_deep=True,  # LSTM 스킵 (실험 속도)
    )
    return {
        "exp_id": exp_id,
        "test_mape": results.get("test_mape"),
        "test_metrics": results.get("test_metrics", {}),
        "best_model": results.get("best_model"),
        "month_mape": results.get("month_mape", {}),
    }


def build_experiment_matrix() -> List[dict]:
    """실험 조합 생성."""
    matrix = []
    for auction_agg in ["mean", "median"]:
        for domae_filter in ["all", "garak"]:
            for low_tx in ["exclude", "fill_week"]:
                for min_tx in [5]:
                    matrix.append({
                        "auction_agg": auction_agg,
                        "domae_filter": domae_filter,
                        "low_tx": low_tx,
                        "min_tx_count": min_tx,
                    })
    return matrix


def main():
    parser = argparse.ArgumentParser(description="Categorical merge experiments")
    parser.add_argument("--analyze-only", action="store_true", help="카테고리컬 분석만 수행")
    parser.add_argument("--skip-training", action="store_true", help="merge만 수행, 학습 스킵")
    parser.add_argument("--exp", type=str, nargs="+", default=None, help="실험 ID만 실행 (예: exp_0 exp_1)")
    args = parser.parse_args()

    EXP_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. Categorical analysis
    logger.info("=" * 60)
    logger.info("1. Categorical columns analysis")
    cat_analysis = analyze_categorical_columns()
    cat_path = EXP_OUTPUT / "categorical_analysis.json"
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(cat_analysis, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", cat_path)

    # Report: categorical
    report_lines = [
        "# Categorical Columns Analysis",
        "",
        "## Auction",
        "| Column | n_unique | Sample values |",
        "|--------|----------|---------------|",
    ]
    for col, info in cat_analysis.get("auction", {}).items():
        if col == "sample_values":
            continue
        samples = info.get("sample_values", [])[:5]
        report_lines.append(f"| {col} | {info.get('n_unique', '')} | {samples} |")
    report_lines.extend(["", "## Domae", "| Column | n_unique | Sample values |", "|--------|----------|---------------|"])
    for col, info in cat_analysis.get("domae", {}).items():
        if col in ("MRKT_NM_counts",) or not isinstance(info, dict):
            continue
        samples = info.get("sample_values", [])[:5]
        report_lines.append(f"| {col} | {info.get('n_unique', '')} | {samples} |")
    if "MRKT_NM_counts" in cat_analysis.get("domae", {}):
        report_lines.extend(["", "### Domae MRKT_NM counts", "| Market | Count |", "|--------|-------|"])
        for k, v in cat_analysis["domae"]["MRKT_NM_counts"].items():
            report_lines.append(f"| {k} | {v} |")

    report_lines.extend([
        "",
        "## 적용 가능한 Feature (실험에서 사용)",
        "",
        "| 소스 | 컬럼 | 용도 |",
        "|------|------|------|",
        "| domae | MRKT_NM | 가락도매 필터 (경매 가락시장과 동일 시장 정렬) |",
        "| domae | GRAD_NM | 등급별 분리 (상품/중품) - 추후 확장 |",
        "| auction | 출하지 | 산지별 분리 - 추후 확장 |",
        "| auction | auction_transaction_count | 저거래건수 필터/보정 |",
    ])

    report_path = REPORT_DIR / "categorical_analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Report: %s", report_path)

    if args.analyze_only:
        return

    # 2. Run experiments
    matrix = build_experiment_matrix()
    results_list = []

    for i, cfg in enumerate(matrix):
        exp_id = f"exp_{i}"
        if args.exp and exp_id not in args.exp:
            continue

        cfg_str = f"agg={cfg['auction_agg']}_domae={cfg['domae_filter']}_lowtx={cfg['low_tx']}_mintx={cfg['min_tx_count']}"
        logger.info("=" * 60)
        logger.info("Experiment %s: %s", exp_id, cfg_str)

        merged_path = EXP_OUTPUT / f"merged_{exp_id}.csv"
        try:
            run_merge_with_config(cfg, DATA_ROOT, merged_path)
        except Exception as e:
            logger.warning("Merge failed %s: %s", exp_id, str(e)[:80])
            results_list.append({"exp_id": exp_id, "config": cfg, "error": str(e), "test_mape": None})
            continue

        if not args.skip_training:
            try:
                res = run_training_for_merged(merged_path, exp_id)
                res["config"] = cfg
                res["config_str"] = cfg_str
                results_list.append(res)
            except Exception as e:
                logger.warning("Training failed %s: %s", exp_id, str(e)[:80])
                results_list.append({"exp_id": exp_id, "config": cfg, "config_str": cfg_str, "error": str(e), "test_mape": None})

    # 3. Report & plots
    if results_list:
        _write_experiment_report(results_list)
        _plot_experiment_comparison(results_list)


def _write_experiment_report(results_list: List[dict]) -> None:
    """실험 결과 보고서 작성."""
    lines = [
        "# Categorical Merge Experiments Report",
        "",
        "## Experiment Matrix",
        "| auction_agg | domae_filter | low_tx | min_tx_count |",
        "|-------------|--------------|--------|--------------|",
    ]
    for r in results_list:
        c = r.get("config", {})
        lines.append(f"| {c.get('auction_agg')} | {c.get('domae_filter')} | {c.get('low_tx')} | {c.get('min_tx_count')} |")

    lines.extend([
        "",
        "## Results",
        "| Exp ID | Config | Test MAPE | Best Model |",
        "|--------|--------|-----------|------------|",
    ])
    for r in results_list:
        mape = r.get("test_mape")
        mape_str = f"{mape:.2f}%" if mape is not None else "N/A"
        err = r.get("error", "")
        if err:
            mape_str = f"ERROR: {err[:40]}"
        lines.append(f"| {r.get('exp_id')} | {r.get('config_str', '')} | {mape_str} | {r.get('best_model', '')} |")

    # Best
    valid = [r for r in results_list if r.get("test_mape") is not None]
    if valid:
        best = min(valid, key=lambda x: x["test_mape"])
        lines.extend([
            "",
            "## Best Configuration",
            f"- **Exp ID**: {best['exp_id']}",
            f"- **Config**: {best.get('config_str')}",
            f"- **Test MAPE**: {best['test_mape']:.2f}%",
            f"- **Best Model**: {best.get('best_model')}",
        ])

    report_path = REPORT_DIR / "categorical_experiments_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", report_path)


def _plot_experiment_comparison(results_list: List[dict]) -> None:
    """실험 비교 플롯."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not found, skip plot")
        return

    valid = [r for r in results_list if r.get("test_mape") is not None]
    if not valid:
        return

    labels = [r.get("config_str", r["exp_id"])[:40] for r in valid]
    mapes = [r["test_mape"] for r in valid]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(valid)))

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid) * 0.4)))
    y_pos = np.arange(len(valid))
    bars = ax.barh(y_pos, mapes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Test MAPE (%)")
    ax.set_title("Categorical Merge Experiments: Test MAPE Comparison")
    ax.invert_yaxis()
    plt.tight_layout()
    out_path = REPORT_DIR / "categorical_experiments_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot: %s", out_path)


if __name__ == "__main__":
    main()
