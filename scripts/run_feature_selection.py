#!/usr/bin/env python3
"""
Feature selection 파이프라인 실행.

1. auction + domae + somae + weather 통합 (1kg당 가격 target, 도매/소매 lag 1~7일)
2. 컬럼 프로파일 (5개 샘플, 분류)
3. Gini / Mutual Info / Correlation 기반 중요도
4. Markdown 리포트 + 메타데이터 config 기반 전체 순위 출력
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from feature_selection.column_profiler import ColumnProfiler, EXCLUDE_FROM_FEATURES
from feature_selection.data_merger import DataMerger
from feature_selection.feature_selector import FeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "temp"
REPORT_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"


def load_metadata() -> dict:
    """config/feature_metadata.json 로드."""
    path = CONFIG_DIR / "feature_metadata.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_report(
    profiles: list,
    candidates: list,
    drops: list,
    results: dict,
    agg_rank: object,
    variety: str,
    metadata: dict,
) -> str:
    """Markdown 리포트 생성. 모든 순위에 설명 포함."""
    lines = []
    desc = lambda c: metadata.get(c, c)

    lines.append("# Feature Selection 리포트")
    lines.append("")
    lines.append("## 1. 목표")
    lines.append("- **Target**: `price_per_kg_mean` (일별 품종별 **1kg당 경락가** 평균)")
    lines.append("  - 단량(20kg, 10kg 등) 파싱 후 단량당 경락가/kg로 환산한 값의 평균")
    lines.append("- **도매/소매가**: 당일 경매 후 수집되므로 **lag 1~7일**만 사용")
    lines.append("- **품종별 모델**: 사용자가 품종을 선택하면 해당 품종 전용 ML 모델 학습")
    lines.append("- **데이터 소스**: auction, domae, somae, weather")
    lines.append("")

    lines.append("## 2. 컬럼 프로파일 (컬럼별 5개 샘플)")
    lines.append("")
    lines.append("### 2.1 후보 컬럼 (candidate)")
    cand_profiles = [p for p in profiles if p["column"] in candidates]
    lines.append(ColumnProfiler().to_markdown_table(cand_profiles))
    lines.append("")

    lines.append("### 2.2 제외 컬럼 (drop)")
    drop_profiles = [p for p in profiles if p["column"] in drops]
    if drop_profiles:
        lines.append(ColumnProfiler().to_markdown_table(drop_profiles))
    else:
        lines.append("없음")
    lines.append("")

    lines.append("## 3. Feature Importance (품종: " + variety + ")")
    lines.append("")

    def table_with_desc(method: str, title: str, val_fmt: str = ".4f"):
        if method not in results:
            return
        lines.append(f"### {title}")
        lines.append("| 순위 | 컬럼 | 설명 | 수치 |")
        lines.append("|------|------|------|------|")
        for i, (col, val) in enumerate(results[method].items(), 1):
            lines.append(f"| {i} | {col} | {desc(col)} | {val:{val_fmt}} |")
        lines.append("")

    table_with_desc("gini", "3.1 Gini (RandomForestClassifier, target 5-quantile 구간화)")
    table_with_desc("mutual_info", "3.2 Mutual Information")
    table_with_desc("correlation", "3.3 Correlation (|r|)")

    lines.append("## 4. 통합 순위 (가중 평균)")
    lines.append("가중치: Gini 40%, Mutual Info 35%, Correlation 25%")
    lines.append("")
    if agg_rank is not None and len(agg_rank) > 0:
        lines.append("| 순위 | 컬럼 | 설명 | 통합점수 |")
        lines.append("|------|------|------|----------|")
        for i, (col, val) in enumerate(agg_rank.items(), 1):
            lines.append(f"| {i} | {col} | {desc(col)} | {val:.4f} |")
    lines.append("")

    lines.append("## 5. 권장 Feature 후보")
    lines.append("")
    lines.append("위 통합 순위 전체를 확인하여 초기 feature set 선정. 품종별 importance 재실행 권장.")
    lines.append("")
    return "\n".join(lines)


def print_full_ranked_table(agg_rank, results: dict, metadata: dict):
    """콘솔에 순위, 컬럼명, 설명, 수치 전체 출력."""
    desc = lambda c: metadata.get(c, c)

    print("\n" + "=" * 80)
    print("전체 Feature 순위 (통합)")
    print("=" * 80)
    print(f"{'순위':<6} {'컬럼':<35} {'설명':<45} {'수치':<10}")
    print("-" * 80)
    for i, (col, val) in enumerate(agg_rank.items(), 1):
        d = desc(col)
        if len(d) > 42:
            d = d[:39] + "..."
        print(f"{i:<6} {col:<35} {d:<45} {val:.4f}")

    for method, title in [
        ("gini", "Gini"),
        ("mutual_info", "Mutual Information"),
        ("correlation", "Correlation"),
    ]:
        if method not in results:
            continue
        print("\n" + "=" * 80)
        print(f"전체 Feature 순위 ({title})")
        print("=" * 80)
        print(f"{'순위':<6} {'컬럼':<35} {'설명':<45} {'수치':<10}")
        print("-" * 80)
        for i, (col, val) in enumerate(results[method].items(), 1):
            d = desc(col)
            if len(d) > 42:
                d = d[:39] + "..."
            print(f"{i:<6} {col:<35} {d:<45} {val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Feature selection 파이프라인")
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--variety", type=str, default="후지", help="분석 품종")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-merged", action="store_true", help="통합 데이터 CSV 저장")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else REPORT_DIR / "feature_selection_report.md"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    # 1. 데이터 통합
    logger.info("1. 데이터 통합 중...")
    merger = DataMerger(data_root=data_dir)
    merged = merger.run()

    if args.save_merged:
        merged_path = TEMP_DIR / "merged_for_feature_selection.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        logger.info("통합 데이터 저장: %s", merged_path)

    # 2. 컬럼 프로파일
    logger.info("2. 컬럼 프로파일링...")
    profiler = ColumnProfiler(n_samples=5)
    profiles, candidates, drops = profiler.profile_dataframe(merged)
    logger.info("후보 컬럼: %d개, 제외: %d개", len(candidates), len(drops))

    # 3. Feature selection (품종별)
    logger.info("3. Feature importance 계산 (품종: %s)...", args.variety)
    selector = FeatureSelector()
    results = selector.run_all(merged, candidates, variety=args.variety)

    if not results:
        logger.error("Feature selection 결과 없음 (데이터 부족?)")
        sys.exit(1)

    agg_rank = selector.rank_aggregate(results)

    # 4. 리포트 생성
    report = build_report(profiles, candidates, drops, results, agg_rank, args.variety, metadata)
    output_path.write_text(report, encoding="utf-8")
    logger.info("리포트 저장: %s", output_path)

    # 5. 콘솔 전체 출력 (순위, 컬럼, 설명, 수치)
    print_full_ranked_table(agg_rank, results, metadata)

    print("\n" + "=" * 80)
    print("Feature Selection 완료")
    print("=" * 80)
    print(f"리포트: {output_path}")
    print(f"메타데이터: {CONFIG_DIR / 'feature_metadata.json'}")
    print(f"후보 컬럼 수: {len(candidates)}")


if __name__ == "__main__":
    main()
