#!/usr/bin/env python3
"""
각 API 개별 테스트
실행: python scripts/test_apis.py [somae|domae|weather|auction|all]
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("API_KEY", "").strip()


def test_somae():
    """소매 API (Grid_20141225000000000163_1)"""
    print("\n" + "=" * 60)
    print("1. SOMAE (소매가격) API")
    print("=" * 60)
    if not API_KEY:
        print("❌ API_KEY 없음")
        return False

    code = "Grid_20141225000000000163_1"
    url = f"http://211.237.50.150:7080/openapi/{API_KEY}/json/{code}/1/10?EXAMIN_DE=20150529&FRMPRD_CATGORY_CD=400&PRDLST_CD=411"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        grid = data.get(code, {})
        rows = grid.get("row", [])
        total = grid.get("totalCnt", 0)
        print(f"  20150529 (문서 sample): totalCnt={total}, row={len(rows)}")
        if rows:
            print(f"  ✅ 첫 행: {rows[0].get('PRDLST_NM')} {rows[0].get('AMT')}원")
            return True
        print("  ⚠️ row 비어있음")
    except Exception as e:
        print(f"  ❌ {e}")
    return False


def test_domae():
    """도매 API (Grid_20150406000000000217_1)"""
    print("\n" + "=" * 60)
    print("2. DOMAE (도매가격) API")
    print("=" * 60)
    if not API_KEY:
        print("❌ API_KEY 없음")
        return False

    code = "Grid_20150406000000000217_1"
    url = f"http://211.237.50.150:7080/openapi/{API_KEY}/json/{code}/1/10?EXAMIN_DE=20150504&FRMPRD_CATGORY_CD=400&PRDLST_CD=411"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        grid = data.get(code, {})
        rows = grid.get("row", [])
        total = grid.get("totalCnt", 0)
        print(f"  20150504 (문서 sample): totalCnt={total}, row={len(rows)}")
        if rows:
            print(f"  ✅ 첫 행: {rows[0].get('PRDLST_NM')} {rows[0].get('AMT')}원")
            return True
        print("  ⚠️ row 비어있음")
    except Exception as e:
        print(f"  ❌ {e}")
    return False


def test_weather(use_sample_key: bool = False):
    """기상청 API - 문서 1.3 일자료(kma_sfcdd) / 1.4 일자료 기간 조회(kma_sfcdd3)"""
    print("\n" + "=" * 60)
    print("3. WEATHER (기상청) API")
    print("=" * 60)
    auth = "b1IAnoyWT-aSAJ6Mll_muw" if use_sample_key else \
           (os.getenv("WEATHER_API_KEY") or os.getenv("WEATHER_AUTH_KEY"))
    if not auth:
        print("  ⚠️ WEATHER_API_KEY / WEATHER_AUTH_KEY 없음")
        return False
    if use_sample_key:
        print("  (문서 샘플 authKey 사용)")

    # 1.4 일자료(기간 조회) - tm1, tm2, stn (문서 그대로)
    base = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
    url = f"{base}?tm1=20151211&tm2=20151214&stn=108&help=0&authKey={auth}"
    print(f"  [1.4 일자료 기간] tm1=20151211, tm2=20151214, stn=108")
    try:
        r = requests.get(url, timeout=30)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            lines = [l for l in r.text.strip().split("\n") if not l.startswith("#")]
            print(f"  데이터 라인: {len(lines)}개")
            if lines:
                print(f"  ✅ 첫 줄: {lines[0][:80]}...")
                return True
            print(f"  응답: {r.text[:300]}")
        else:
            try:
                err = r.json()
                msg = err.get("result", {}).get("message", r.text[:150])
            except Exception:
                msg = r.text[:150]
            print(f"  ❌ {msg}")
            if "403" in str(r.status_code) or "용량" in str(msg):
                print("  → 403: 일일 호출 한도 초과. 내일 재시도 또는 apihub.kma.go.kr에서 한도 확인")
    except Exception as e:
        print(f"  ❌ {e}")
        return False

    # 1.3 일자료 (단일일) - tm, stn
    if not use_sample_key:
        print(f"\n  [1.3 일자료 단일] tm=20150715, stn=0")
    url2 = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php?tm=20150715&stn=0&help=0&authKey={auth}"
    try:
        r2 = requests.get(url2, timeout=30)
        print(f"  Status: {r2.status_code}")
        if r2.status_code == 200:
            lines2 = [l for l in r2.text.strip().split("\n") if not l.startswith("#")]
            if lines2:
                print(f"  ✅ 1.3 일자료 {len(lines2)}행")
                return True
    except Exception:
        pass
    return False


def test_auction():
    """경매(가락시장) - API 키 불필요"""
    print("\n" + "=" * 60)
    print("4. AUCTION (가락시장 경매)")
    print("=" * 60)
    url = "https://at.agromarket.kr/domeinfo/sanRealtime.do?pageNo=1&saledate=2024-01-10&whsalCd=110001&largeCd=06&midCd=01&pageSize=100"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        import io
        import pandas as pd
        dfs = pd.read_html(io.StringIO(r.text), thousands=",", encoding="utf-8")
        if dfs and not dfs[0].empty and "경매가" not in str(dfs[0].iloc[0, 0]):
            print(f"  ✅ 테이블 {len(dfs[0])}행 수신")
            return True
        print("  ⚠️ 데이터 테이블 없음")
    except Exception as e:
        print(f"  ❌ {e}")
    return False


def main():
    args = [a for a in sys.argv[1:] if a not in ("--sample", "-s")]
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    which = (args[0] if args else "all").lower()

    print(f"\nAPI_KEY: {API_KEY[:20]}...{API_KEY[-8:]}" if API_KEY else "API_KEY: (없음)")
    if use_sample:
        print("(weather: 문서 샘플 authKey로 API 형식 검증)")

    results = {}
    if which in ("somae", "all"):
        results["somae"] = test_somae()
    if which in ("domae", "all"):
        results["domae"] = test_domae()
    if which in ("weather", "all"):
        results["weather"] = test_weather(use_sample_key=use_sample)
    if which in ("auction", "all"):
        results["auction"] = test_auction()

    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'✅ OK' if ok else '❌ 실패'}")
    print()


if __name__ == "__main__":
    main()
