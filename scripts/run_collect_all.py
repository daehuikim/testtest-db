#!/usr/bin/env python3
"""
2020-01-01 ~ 2025-12-31 전체 데이터 수집 통합 스크립트

- auction, somae, domae, weather를 병렬로 수집
- 각 소스별 병렬 처리 적용 (auction: 시장별, somae/domae: 날짜별, weather: 월별)
- 수집된 raw 데이터를 개별 CSV로 저장
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return _FakeTqdm(**kwargs)
        return iterable

    class _FakeTqdm:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(SCRIPT_DIR))

from collectors import AuctionCollector, DomaeCollector, SomaeCollector, WeatherCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date(2025, 12, 31)
DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def run_auction(start: date, end: date, output_dir: Path) -> Path:
    """경매 데이터 수집"""
    c = AuctionCollector(start, end, str(output_dir / "auction"))
    df = c.collect()
    if df.empty:
        logger.warning("auction: 수집된 데이터 없음")
        return Path()
    return c.save(df, f"auction_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")


def run_somae(start: date, end: date, output_dir: Path) -> Path:
    """소매 데이터 수집"""
    c = SomaeCollector(start, end, str(output_dir / "somae"))
    df = c.collect()
    if df.empty:
        logger.warning("somae: 수집된 데이터 없음")
        return Path()
    return c.save(df, f"somae_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")


def run_domae(start: date, end: date, output_dir: Path) -> Path:
    """도매 데이터 수집"""
    c = DomaeCollector(start, end, str(output_dir / "domae"))
    df = c.collect()
    if df.empty:
        logger.warning("domae: 수집된 데이터 없음")
        return Path()
    return c.save(df, f"domae_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")


def run_weather(start: date, end: date, output_dir: Path) -> Path:
    """기상 데이터 수집 (단일 파일로 저장)"""
    c = WeatherCollector(start, end, str(output_dir / "weather"))
    df = c.collect()
    if df.empty:
        logger.warning("weather: 수집된 데이터 없음")
        return Path()
    return c.save(
        df,
        f"weather_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="전체 데이터 수집 (2020-01-01 ~ 2025-12-31)")
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="시작일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="종료일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 루트 디렉터리 (기본: data/raw)",
    )
    parser.add_argument(
        "--parallel-sources",
        action="store_true",
        default=True,
        help="auction/somae/domae/weather를 병렬로 수집 (기본: True)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="소스별 순차 수집 (디버깅용)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=["auction", "somae", "domae", "weather"],
        default=None,
        help="수집할 소스만 지정 (미지정 시 전체)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="수집 후 combine_raw_data 실행하여 시계열 통합",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="수집 생략, 기존 raw 데이터만 combine (데이터 있을 때)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output) if args.output else DATA_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("수집 기간: %s ~ %s", start, end)
    logger.info("출력 디렉터리: %s", output_dir)

    if args.combine_only:
        logger.info("combine만 실행 (수집 생략)")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        output_path = output_dir / "combined" / "raw_combined.csv"
        try:
            from combine_raw_data import combine
            result = combine(output_dir, start_str, end_str, output_path)
            if result.empty:
                logger.warning("통합할 데이터 없음 (data/raw/*.csv 확인 필요)")
            else:
                logger.info("통합 완료: %s (%d행)", output_path, len(result))
        except Exception as e:
            logger.exception("통합 실패: %s", e)
        return

    all_tasks = [
        ("auction", lambda: run_auction(start, end, output_dir)),
        ("somae", lambda: run_somae(start, end, output_dir)),
        ("domae", lambda: run_domae(start, end, output_dir)),
        ("weather", lambda: run_weather(start, end, output_dir)),
    ]
    tasks = (
        [(n, fn) for n, fn in all_tasks if n in args.only]
        if args.only
        else all_tasks
    )

    # KMA 기상 API는 동시 요청 시 rate limit → weather는 마지막에 순차 실행
    parallel_tasks = [(n, fn) for n, fn in tasks if n != "weather"]
    weather_task = [(n, fn) for n, fn in tasks if n == "weather"]

    if args.sequential:
        for name, fn in tqdm(tasks, desc="전체 수집", unit="source"):
            t0 = time.perf_counter()
            logger.info("=== %s 수집 시작 ===", name)
            fn()
            elapsed = time.perf_counter() - t0
            logger.info("=== %s 수집 완료 (%.1f초) ===", name, elapsed)
    else:
        # auction/somae/domae 병렬 → weather 순차 (KMA rate limit 회피)
        if parallel_tasks:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(fn): name for name, fn in parallel_tasks}
                with tqdm(total=len(futures), desc="전체 수집", unit="source") as pbar:
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            path = future.result()
                            logger.info("%s 완료: %s", name, path)
                        except Exception as e:
                            logger.exception("%s 실패: %s", name, e)
                        pbar.update(1)
        for name, fn in weather_task:
            t0 = time.perf_counter()
            logger.info("=== %s 수집 시작 ===", name)
            try:
                path = fn()
                logger.info("%s 완료: %s", name, path)
            except Exception as e:
                logger.exception("%s 실패: %s", name, e)
            logger.info("=== %s 수집 완료 (%.1f초) ===", name, time.perf_counter() - t0)

    if args.combine:
        logger.info("시계열 통합 실행 중...")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        output_path = output_dir / "combined" / "raw_combined.csv"
        try:
            from combine_raw_data import combine
            result = combine(output_dir, start_str, end_str, output_path)
            if result.empty:
                logger.warning("통합할 데이터 없음 (수집된 파일 확인 필요)")
            else:
                logger.info("통합 완료: %s (%d행)", output_path, len(result))
        except Exception as e:
            logger.exception("통합 실패: %s", e)
    else:
        logger.info("전체 수집 완료. --combine 옵션으로 시계열 통합을 진행하세요.")


if __name__ == "__main__":
    main()
