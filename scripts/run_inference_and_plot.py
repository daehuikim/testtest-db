#!/usr/bin/env python3
"""
Load best model checkpoint, run inference on test data, and plot time series.
- Blue line: actual prices (train period, not predicted)
- Red line: predicted prices (test period)
All labels in English to avoid font issues.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "best_model"


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load model checkpoint."""
    import joblib
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    size = ckpt_path.stat().st_size
    if size < 100:
        raise ValueError(f"Checkpoint file too small ({size} bytes), likely corrupted. Re-run training.")
    try:
        ckpt = joblib.load(ckpt_path)
    except (EOFError, ValueError) as e:
        raise ValueError(
            f"Checkpoint corrupted or incomplete (EOFError). "
            f"Delete {ckpt_path} and re-run: python scripts/run_training_pipeline.py --skip-shap"
        ) from e
    return ckpt


def load_data(data_path: Path, variety_filter: list) -> pd.DataFrame:
    """Load merged data with optional variety filter."""
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df["date"] = df["date"].astype(str).str.replace(r"\D", "", regex=True).str[:8].str.zfill(8)
    if variety_filter and "품종" in df.columns:
        df = df[df["품종"].isin(variety_filter)].copy()
    return df


SEASON_MONTHS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "fall": [9, 10, 11], "winter": [12, 1, 2]}


def _month_to_season(month: int) -> str:
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    return "unknown"


def predict_batch(ckpt: dict, df: pd.DataFrame, dates: list) -> tuple:
    """Run inference for given dates. Returns (predictions array, ordered dates)."""
    features = ckpt["features"]
    use_log = ckpt["use_log"]
    use_stacking = ckpt["use_stacking"]
    use_seasonal_routing = ckpt.get("use_seasonal_routing", False)
    seasonal_models = ckpt.get("seasonal_models")
    base_models = ckpt.get("base_models", [])
    base_names = ckpt.get("base_names", [])
    meta = ckpt.get("meta")
    seq_cols = ckpt.get("seq_cols")

    sub = df[df["date"].isin(dates)].copy()
    if sub.empty:
        return np.array([]), []
    sub = sub.sort_values("date").drop_duplicates("date", keep="first")
    sub = sub[sub["date"].isin(dates)].sort_values("date")
    ordered_dates = sub["date"].tolist()
    med = sub.median(numeric_only=True)
    sub = sub.fillna(med)

    if use_seasonal_routing and seasonal_models:
        sub["_month"] = pd.to_datetime(sub["date"].astype(str), format="%Y%m%d").dt.month
        sub["_season"] = sub["_month"].apply(_month_to_season)
        pred_list = []
        fallback_s = list(seasonal_models)[0]
        for idx in range(len(sub)):
            s = sub.iloc[idx]["_season"]
            m = seasonal_models.get(s, seasonal_models[fallback_s])
            pred_list.append(m.predict(sub[features].iloc[[idx]])[0])
        pred = np.array(pred_list)
    else:
        preds_list = []
        for i, name in enumerate(base_names):
            m = base_models[i]
            if name in ("lstm", "transformer") and seq_cols:
                from training.deep_models import build_sequence_array
                X_seq = build_sequence_array(sub, seq_cols)
                p = m.predict(X_seq)
            else:
                X = sub[features]
                p = m.predict(X)
            preds_list.append(p)

        if use_stacking and meta is not None and len(preds_list) >= 2:
            oof = np.column_stack(preds_list)
            valid = np.all(np.isfinite(oof), axis=1)
            pred = np.full(len(oof), np.nan)
            pred[valid] = meta.predict(oof[valid])
        else:
            pred = preds_list[0] if preds_list else np.array([])

    if use_log:
        pred = np.expm1(pred)
    return pred, ordered_dates


def run_inference_and_plot(
    checkpoint_path: Path = None,
    data_path: Path = None,
    output_path: Path = None,
) -> None:
    """Load checkpoint, infer on test, plot time series."""
    ckpt_path = checkpoint_path or CHECKPOINT_DIR / "checkpoint.joblib"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s. Run training pipeline first.", ckpt_path)
        return

    data_path = data_path or PROJECT_ROOT / "temp" / "merged_stage0.csv"
    if not data_path.exists():
        logger.error("Data not found: %s", data_path)
        return

    ckpt = load_checkpoint(ckpt_path)
    variety_filter = ckpt.get("variety_filter", [])
    df = load_data(data_path, variety_filter)
    features = ckpt.get("features", [])
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning("Missing features in data: %s", missing[:5])

    train_dates = ckpt.get("train_dates", [])
    test_dates = ckpt.get("test_dates", [])

    # Aggregate by date (in case multiple rows per date)
    def agg_by_date(dates):
        sub = df[df["date"].isin(dates)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["date", "price"])
        g = sub.groupby("date")[TARGET_COL].mean().reset_index()
        g.columns = ["date", "price"]
        return g.sort_values("date")

    train_df = agg_by_date(train_dates)
    test_actual_df = agg_by_date(test_dates)

    if train_df.empty:
        logger.warning("No train data found")
        return

    # Run inference on test dates
    test_pred, test_ordered_dates = predict_batch(ckpt, df, test_dates)
    test_plot_df = pd.DataFrame({"date": test_ordered_dates, "plot_price": test_pred[: len(test_ordered_dates)]})
    test_plot_df["source"] = "test"

    # Build plot data: all dates sorted
    train_df["source"] = "train"
    train_df["plot_price"] = train_df["price"]
    all_dates = sorted(set(train_df["date"].tolist()) | set(test_plot_df["date"].tolist()))
    plot_data = []
    for d in all_dates:
        if d in train_df["date"].values:
            row = train_df[train_df["date"] == d].iloc[0]
            plot_data.append({"date": d, "price": row["plot_price"], "source": "train"})
        else:
            row = test_plot_df[test_plot_df["date"] == d].iloc[0]
            plot_data.append({"date": d, "price": row["plot_price"], "source": "test"})
    plot_df = pd.DataFrame(plot_data).sort_values("date")
    plot_df["date_parsed"] = pd.to_datetime(plot_df["date"].astype(str), format="%Y%m%d")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    train_mask = plot_df["source"] == "train"
    test_mask = plot_df["source"] == "test"

    ax.plot(
        plot_df.loc[train_mask, "date_parsed"],
        plot_df.loc[train_mask, "price"],
        color="blue",
        linewidth=1.5,
        label="Actual (train)",
    )
    ax.plot(
        plot_df.loc[test_mask, "date_parsed"],
        plot_df.loc[test_mask, "price"],
        color="red",
        linewidth=1.5,
        label="Predicted (test)",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (KRW/kg)", fontsize=12)
    ax.set_title("Price Time Series: Actual (Train) vs Predicted (Test)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out_path = output_path or PROJECT_ROOT / "reports" / "inference_timeseries.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Inference and time series plot")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.joblib")
    parser.add_argument("--data", type=str, default=None, help="Path to merged_stage0.csv")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    run_inference_and_plot(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        data_path=Path(args.data) if args.data else None,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
