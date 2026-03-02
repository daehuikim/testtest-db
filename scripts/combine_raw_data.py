#!/usr/bin/env python3
"""
수집된 raw 데이터를 날짜 기준 시계열로 통합

- auction, somae, domae, weather CSV를 읽어
- 날짜(date) 컬럼을 통일하고
- 하나의 통합 raw 파일로 저장
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = DATA_ROOT / "combined"


def normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """날짜 컬럼을 YYYYMMDD 문자열로 통일"""
    df = df.copy()
    if col not in df.columns:
        logger.warning("컬럼 없음: %s", col)
        return df
    s = df[col].astype(str).str.replace(r"\D", "", regex=True)
    df["date"] = s.str[:8].str.zfill(8)
    return df


def load_and_tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """소스 태그 추가"""
    df = df.copy()
    df["source"] = source
    return df


def load_auction(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    """경매 데이터 로드"""
    path = data_dir / "auction" / f"auction_{start}_{end}.csv"
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = normalize_date_col(df, "경락일시")
    return load_and_tag(df, "auction")


def load_somae(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    """소매 데이터 로드"""
    path = data_dir / "somae" / f"somae_{start}_{end}.csv"
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = normalize_date_col(df, "EXAMIN_DE")
    return load_and_tag(df, "somae")


def load_domae(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    """도매 데이터 로드"""
    path = data_dir / "domae" / f"domae_{start}_{end}.csv"
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = normalize_date_col(df, "EXAMIN_DE")
    return load_and_tag(df, "domae")


def load_weather(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    """기상 데이터 로드"""
    path = data_dir / "weather" / f"weather_{start}_{end}.csv"
    if not path.exists():
        logger.warning("파일 없음: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = normalize_date_col(df, "TM")
    return load_and_tag(df, "weather")


def combine(
    data_dir: Path,
    start: str,
    end: str,
    output_path: Path,
) -> pd.DataFrame:
    """모든 raw 데이터를 날짜 기준으로 통합"""
    dfs = []

    for name, loader in [
        ("auction", load_auction),
        ("somae", load_somae),
        ("domae", load_domae),
        ("weather", load_weather),
    ]:
        df = loader(data_dir, start, end)
        if not df.empty:
            dfs.append(df)
            logger.info("%s: %d행 로드", name, len(df))

    if not dfs:
        logger.error("로드된 데이터 없음")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["date", "source"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("통합 저장: %s (%d행)", output_path, len(combined))
    return combined


def main():
    parser = argparse.ArgumentParser(description="raw 데이터 시계열 통합")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_ROOT),
        help="raw 데이터 루트 디렉터리",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="20240101",
        help="시작일 (YYYYMMDD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="20251231",
        help="종료일 (YYYYMMDD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (기본: data/raw/combined/raw_combined.csv)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "raw_combined.csv"

    combine(data_dir, args.start, args.end, output_path)


if __name__ == "__main__":
    main()
