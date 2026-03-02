#!/usr/bin/env python3
"""
Research-grade Feature Selection Pipeline

Stage 0: Maximal lag generation (동일 스키마, 품종 무관)
Stage 1: CCF + MI pre-filter
Stage 2: Elastic Net (TimeSeriesSplit)
Stage 3: LightGBM + Rolling Permutation Importance
Stage 4: Temporal stability
Stage 5: 공통 feature (품종 리스트업 → 모든 품종에서 중요한 feature만)

⚠️ 품종별 다른 lag/feature 사용 금지. 최종 feature set은 모든 품종에 동일 적용.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_selection.config_loader import load_config
from feature_selection.data_merger import DataMerger
from feature_selection.stage1_prefilter import stage1_prefilter
from feature_selection.stage2_elasticnet import stage2_elasticnet
from feature_selection.stage3_rolling_permutation import stage3_rolling_permutation
from feature_selection.stage4_stability import stage4_stability
from feature_selection.stage5_common import stage5_common, get_variety_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"


def run_pipeline(
    data_dir: Path,
    save_merged: bool = False,
    output_dir: Path = None,
) -> list:
    """전체 파이프라인 실행. 최종 feature list 반환."""
    output_dir = output_dir or REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    if config.get("use_gpu"):
        logger.info("GPU 모드 활성화 (LightGBM)")

    # Stage 0
    logger.info("=" * 60)
    logger.info("Stage 0: Maximal lag generation")
    merger = DataMerger(data_root=data_dir, config=config)
    df = merger.run()

    if save_merged:
        temp_path = PROJECT_ROOT / "temp" / "merged_stage0.csv"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        logger.info("저장: %s", temp_path)

    # 품종 필터 (popular만 사용)
    variety_cfg = config.get("variety_filter", {})
    min_pct = variety_cfg.get("min_pct")
    whitelist = variety_cfg.get("whitelist")
    max_varieties = variety_cfg.get("max_varieties")

    if whitelist:
        df = df[df["품종"].isin(whitelist)].copy()
        logger.info("품종 whitelist 적용: %s", whitelist)
    elif min_pct is not None and min_pct > 0:
        cnt = df["품종"].value_counts()
        pct = cnt / len(df) * 100
        keep = pct[pct >= min_pct].index.tolist()
        df = df[df["품종"].isin(keep)].copy()
        logger.info("품종 min_pct>=%.1f 적용: %d개", min_pct, len(keep))
    elif max_varieties is not None:
        top = df["품종"].value_counts().head(max_varieties).index.tolist()
        df = df[df["품종"].isin(top)].copy()
        logger.info("품종 상위 %d개 적용: %s", max_varieties, top)

    varieties = get_variety_list(df, min_samples=50)
    logger.info("품종 %d개: %s", len(varieties), varieties[:10])

    # Feature 후보 (target 제외)
    from feature_selection.stage1_prefilter import get_feature_cols
    all_features = get_feature_cols(df)
    logger.info("전체 feature 후보: %d개", len(all_features))

    # Stage 1
    logger.info("=" * 60)
    logger.info("Stage 1: CCF + MI pre-filter")
    s1 = stage1_prefilter(df, config=config)
    features = list(s1) if s1 else all_features

    # Stage 2
    logger.info("=" * 60)
    logger.info("Stage 2: Elastic Net")
    s2 = stage2_elasticnet(df, features, config=config)
    features = list(s2) if s2 else features

    # Stage 3
    logger.info("=" * 60)
    logger.info("Stage 3: Rolling Permutation Importance")
    s3 = stage3_rolling_permutation(df, features, config=config)
    features = list(s3) if s3 else features

    # Stage 4
    logger.info("=" * 60)
    logger.info("Stage 4: Temporal stability")
    s4 = stage4_stability(df, features, config=config)
    features = list(s4) if s4 else features

    # Stage 5: 공통 feature (품종별 중요도 → 교집합)
    logger.info("=" * 60)
    logger.info("Stage 5: Common feature across varieties")
    s5 = stage5_common(df, features, config=config)
    features = list(s5) if s5 else features

    # Seasonal lags 항상 포함 (Stage 1~5에서 제거되어도 복원)
    seasonal_cfg = config.get("seasonal_lags", {})
    seasonal_list = seasonal_cfg.get("lags", [364, 365, 366]) if isinstance(seasonal_cfg, dict) else [364, 365, 366]
    if seasonal_cfg.get("enabled", True):
        for lag in seasonal_list:
            col = f"price_per_kg_mean_lag{lag}"
            if col in df.columns and col not in features:
                features.append(col)
                logger.info("Seasonal lag 복원: %s", col)

    # Stage 6: Lag clustering (base별 대표 lag 1개)
    s5_count = len(features)
    cluster_cfg = config.get("feature_cluster", {})
    if cluster_cfg.get("enabled", True) and len(features) > 30:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            from training.feature_cluster import reduce_by_lag_representative
            seasonal_cfg = config.get("seasonal_lags", {})
            seasonal_list = seasonal_cfg.get("lags", [364, 365, 366]) if isinstance(seasonal_cfg, dict) else [364, 365, 366]
            features = reduce_by_lag_representative(
                df, features,
                target_col="price_per_kg_mean",
                top_k_per_base=cluster_cfg.get("top_k_per_base", 1),
                max_final=cluster_cfg.get("max_final", 50),
                seasonal_lags=seasonal_list,
                use_gpu=config.get("use_gpu", False),
                random_state=42,
            )
            logger.info("Stage 6: Lag clustering 완료")
        except Exception as e:
            logger.warning("Stage 6 (Lag clustering) 스킵: %s", str(e)[:50])
    final_features = sorted(features)

    # 최종 feature importance (LGBM) - 순위/점수/설명용
    feature_importance = {}
    try:
        import lightgbm as lgb
        from feature_selection.device_utils import fit_lgb_with_fallback
        data = df[df["price_per_kg_mean"].notna()].fillna(df.median(numeric_only=True))
        X = data[[f for f in final_features if f in data.columns]]
        y = data["price_per_kg_mean"]
        if len(X) > 50 and len(X.columns) > 0:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1, random_state=42, device="gpu" if config.get("use_gpu") else "cpu")
            m = fit_lgb_with_fallback(m, X, y, "gpu" if config.get("use_gpu") else "cpu")
            imp = pd.Series(m.feature_importances_, index=X.columns)
            imp = (imp / imp.sum() * 100).round(2)  # 비율 %
            feature_importance = imp.to_dict()
    except Exception as e:
        logger.warning("Feature importance 계산 스킵: %s", str(e)[:50])

    # 품종별 비율
    variety_counts = df["품종"].value_counts()
    variety_ratios = (variety_counts / len(df) * 100).round(1).to_dict()
    representative = config.get("representative_varieties") or []
    if not representative:
        whitelist = (config.get("variety_filter") or {}).get("whitelist")
        if whitelist:
            representative = whitelist if isinstance(whitelist, list) else [whitelist]

    # 중요도순 정렬 (리포트용)
    sorted_features = sorted(
        final_features,
        key=lambda x: (feature_importance.get(x) or 0),
        reverse=True,
    )

    # 리포트 저장
    report = {
        "varieties": varieties,
        "variety_ratios": variety_ratios,
        "representative_varieties": representative,
        "n_varieties": len(varieties),
        "stage0_features": len(all_features),
        "stage1_kept": len(s1) if s1 else 0,
        "stage2_kept": len(s2) if s2 else 0,
        "stage3_kept": len(s3) if s3 else 0,
        "stage4_kept": len(s4) if s4 else 0,
        "stage5_kept": s5_count,
        "stage6_kept": len(final_features),
        "final_features": final_features,
        "final_features_ranked": sorted_features,
        "feature_importance": feature_importance,
        "n_final": len(final_features),
    }
    report_path = output_dir / "feature_selection_pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("리포트: %s", report_path)

    # feature_metadata 로드 (description)
    meta_path = PROJECT_ROOT / "config" / "feature_metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    def get_description(fname: str) -> str:
        if fname in meta:
            return meta[fname]
        m = re.match(r"^(.+)_lag(\d+)$", fname)
        if m:
            base, lag = m.group(1), m.group(2)
            base_key = f"{base}_lag1"
            if base_key in meta:
                return meta[base_key].replace("1일", f"{lag}일")
            if base in meta:
                return f"{lag}일 전 " + meta[base]
            if base == "price_per_kg_mean":
                return f"{lag}일 전 경매 1kg당 가격"
            if "domae" in base:
                return f"{lag}일 전 도매 관련"
            if "somae" in base:
                return f"{lag}일 전 소매 관련"
            if "weather" in base:
                return f"{lag}일 전 기상"
            return f"{lag}일 전 {base}"
        m = re.match(r"^(.+)_rolling(\d+)$", fname)
        if m:
            base, w = m.group(1), m.group(2)
            base_desc = meta.get(base, base)
            return f"{base_desc} ({w}일 rolling)"
        return meta.get(fname, fname)

    # Markdown 요약
    stage_descriptions = {
        0: "Maximal lag 생성 (target/도매/소매/날씨 lag + rolling)",
        1: "CCF + MI pre-filter (상관·상호정보 상위만)",
        2: "Elastic Net (TimeSeriesSplit, L1/L2 정규화)",
        3: "Rolling Permutation Importance (시간창별 안정성)",
        4: "Temporal stability (연도별 rank correlation)",
        5: "Common (품종별 중요도 → 모든 품종에서 중요한 feature만)",
        6: "Lag + Auction cluster (base별 대표 lag 1개, auction_* 상위 2개)",
    }
    md_lines = [
        "# Feature Selection Pipeline 결과",
        "",
        "## 품종 (공통 feature 적용)",
        ", ".join(varieties[:20]) + (" ..." if len(varieties) > 20 else ""),
        "",
        "## 품종별 비율 (%)",
        "| 품종 | 비율 |",
        "|------|------|",
    ]
    for v, pct in list(variety_ratios.items())[:15]:
        md_lines.append(f"| {v} | {pct}% |")
    if len(variety_ratios) > 15:
        md_lines.append(f"| ... | ({len(variety_ratios)}개 품종) |")
    md_lines.extend([
        "",
        "## 대표 품종 (학습용)",
        ", ".join(representative) if representative else "(미설정)",
        "",
        "## Stage별 감소",
        "| Stage | 유지 feature 수 | 설명 |",
        "|-------|----------------|------|",
    ])
    md_lines.append(f"| 0 (Maximal lag) | {len(all_features)} | {stage_descriptions[0]} |")
    md_lines.append(f"| 1 (CCF+MI) | {len(s1) if s1 else '-'} | {stage_descriptions[1]} |")
    md_lines.append(f"| 2 (Elastic Net) | {len(s2) if s2 else '-'} | {stage_descriptions[2]} |")
    md_lines.append(f"| 3 (Rolling Perm) | {len(s3) if s3 else '-'} | {stage_descriptions[3]} |")
    md_lines.append(f"| 4 (Stability) | {len(s4) if s4 else '-'} | {stage_descriptions[4]} |")
    md_lines.append(f"| 5 (Common) | {s5_count} | {stage_descriptions[5]} |")
    md_lines.append(f"| 6 (Lag cluster) | {len(final_features)} | {stage_descriptions[6]} |")
    md_lines.extend([
        "",
        "## 최종 Feature Set",
        "",
        "순위는 Stage 6 Lag clustering 후 LGBM feature importance 기준 (높을수록 중요).",
        "",
        "| 순위 | Feature | 설명 | Importance (%) |",
        "|------|---------|------|----------------|",
    ])
    for i, f in enumerate(sorted_features, 1):
        desc = get_description(f)
        imp_val = feature_importance.get(f, "-")
        imp_str = f"{imp_val}%" if isinstance(imp_val, (int, float)) else str(imp_val)
        md_lines.append(f"| {i} | `{f}` | {desc} | {imp_str} |")
    md_path = output_dir / "feature_selection_pipeline_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Markdown: %s", md_path)

    return final_features


def main():
    parser = argparse.ArgumentParser(description="Feature selection pipeline")
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    final = run_pipeline(
        data_dir=Path(args.data_dir),
        save_merged=args.save_merged,
        output_dir=Path(args.output_dir) if args.output_dir else REPORT_DIR,
    )
    print("\n" + "=" * 60)
    print("Pipeline 완료. 최종 feature 수:", len(final))
    print("=" * 60)


if __name__ == "__main__":
    main()
