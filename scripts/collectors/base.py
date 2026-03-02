"""
데이터 수집기 베이스 클래스
"""

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """데이터 수집기 추상 베이스 클래스"""

    def __init__(
        self,
        start_date: date,
        end_date: date,
        output_dir: str,
        seed: int = 42,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.seed = seed

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """데이터 수집 후 DataFrame 반환"""
        pass

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """DataFrame을 CSV로 저장"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info("저장 완료: %s (%d행)", filepath, len(df))
        return filepath

    @staticmethod
    def date_range(start: date, end: date):
        """날짜 범위 생성기"""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)
