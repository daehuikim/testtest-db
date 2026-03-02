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
import sys
from pathlib import Path

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
    final_features = sorted(s5) if s5 else features

    # 리포트 저장
    report = {
        "varieties": varieties,
        "n_varieties": len(varieties),
        "stage0_features": len(all_features),
        "stage1_kept": len(s1) if s1 else 0,
        "stage2_kept": len(s2) if s2 else 0,
        "stage3_kept": len(s3) if s3 else 0,
        "stage4_kept": len(s4) if s4 else 0,
        "final_features": final_features,
        "n_final": len(final_features),
    }
    report_path = output_dir / "feature_selection_pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("리포트: %s", report_path)

    # Markdown 요약
    md_lines = [
        "# Feature Selection Pipeline 결과",
        "",
        "## 품종 (공통 feature 적용)",
        ", ".join(varieties[:20]) + (" ..." if len(varieties) > 20 else ""),
        "",
        "## Stage별 감소",
        "| Stage | 유지 feature 수 |",
        "|-------|----------------|",
        f"| 0 (Maximal lag) | {len(all_features)} |",
        f"| 1 (CCF+MI) | {len(s1) if s1 else '-'} |",
        f"| 2 (Elastic Net) | {len(s2) if s2 else '-'} |",
        f"| 3 (Rolling Perm) | {len(s3) if s3 else '-'} |",
        f"| 4 (Stability) | {len(s4) if s4 else '-'} |",
        f"| 5 (Common) | {len(final_features)} |",
        "",
        "## 최종 Feature Set",
        "",
    ]
    for i, f in enumerate(final_features, 1):
        md_lines.append(f"{i}. `{f}`")
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
