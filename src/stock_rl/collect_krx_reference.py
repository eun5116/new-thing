from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from stock_rl.config import project_path
from stock_rl.krx_openapi import (
    KrxOpenApiClient,
    fetch_index_history,
    normalize_issue_base,
)


DEFAULT_INDEX_NAMES = {
    "KOSPI": ["코스피", "코스피 200"],
    "KOSDAQ": ["코스닥", "코스닥 150"],
}


def _latest_market_day(client: KrxOpenApiClient, market: str, end: str | None = None, lookback_days: int = 14) -> str:
    end_date = pd.Timestamp(end).date() if end else dt.date.today()
    for offset in range(lookback_days + 1):
        day = end_date - dt.timedelta(days=offset)
        raw = client.fetch_issue_base(market, day)
        if not raw.empty:
            return day.strftime("%Y-%m-%d")
    raise ValueError(f"No {market} issue base data found in the last {lookback_days} days.")


def collect_issue_base(
    client: KrxOpenApiClient,
    data_dir: str,
    market: str,
    bas_dd: str | None = None,
    config_path: str | Path = "configs/krx_kospi.yaml",
) -> Path:
    day = bas_dd or _latest_market_day(client, market)
    raw = client.fetch_issue_base(market, day)
    frame = normalize_issue_base(raw, market)
    out_dir = project_path(config_path, data_dir, "raw", "reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{market.lower()}_issue_base.parquet"
    frame.to_parquet(path, index=False)
    print(f"{market} issue base {day}: {path} rows={len(frame)}")
    return path


def collect_index_series(
    client: KrxOpenApiClient,
    data_dir: str,
    market: str,
    start: str,
    end: str | None = None,
    allow_empty: bool = False,
    empty_cache_ttl_minutes: int = 60,
    config_path: str | Path = "configs/krx_kospi.yaml",
) -> Path:
    cache_dir = project_path(config_path, data_dir, "raw", "krx_daily_cache")
    state_path = project_path(config_path, data_dir, "raw", "collection_state.json")
    out_dir = project_path(config_path, data_dir, "raw", "indices")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{market.lower()}_indices.parquet"
    try:
        frame = fetch_index_history(
            client,
            market,
            start,
            end,
            index_names=DEFAULT_INDEX_NAMES[market],
            cache_dir=cache_dir,
            state_path=state_path,
            empty_cache_ttl_minutes=empty_cache_ttl_minutes,
        )
    except ValueError as exc:
        if allow_empty and "No KRX index data returned" in str(exc) and path.exists():
            print(f"{market} indices {start}..{end or 'today'}: no new rows, keeping {path}")
            return path
        raise
    frame = _merge_index_history(path, frame)
    print(f"{market} indices {start}..{end or 'today'}: {path} rows={len(frame)}")
    return path


def _merge_index_history(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.drop_duplicates(["market", "index_name", "date"], keep="last")
    frame = frame.sort_values(["market", "index_name", "date"]).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return frame


def latest_index_date_by_market(data_dir: str, config_path: str | Path = "configs/krx_kospi.yaml") -> dict[str, str]:
    out_dir = project_path(config_path, data_dir, "raw", "indices")
    latest = {}
    for market in DEFAULT_INDEX_NAMES:
        path = out_dir / f"{market.lower()}_indices.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            continue
        latest[market] = pd.to_datetime(frame["date"]).max().date().isoformat()
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect KRX reference and market index data.")
    parser.add_argument("--data-dir", default="data_krx")
    parser.add_argument("--start", default="2025-05-12")
    parser.add_argument("--end", default=None)
    parser.add_argument("--markets", nargs="+", default=["KOSPI", "KOSDAQ"], choices=["KOSPI", "KOSDAQ"])
    args = parser.parse_args()

    client = KrxOpenApiClient.from_env()
    for market in args.markets:
        collect_issue_base(client, args.data_dir, market, args.end)
        collect_index_series(client, args.data_dir, market, args.start, args.end)


if __name__ == "__main__":
    main()
