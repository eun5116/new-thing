from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_rl.config import load_config, project_path
from stock_rl.krx_openapi import KrxOpenApiClient, fetch_stock_prices
from stock_rl.trading_env import normalize_ticker


PRICE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def fetch_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if frame.empty:
        raise ValueError(f"no price data returned for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=PRICE_COLUMNS).reset_index()
    frame = frame.rename(columns={"Date": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["ticker"] = ticker
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return frame[keep].sort_values(["ticker", "date"])


def collect_prices(
    config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    start_by_market: dict[str, str] | None = None,
    empty_cache_ttl_minutes: int = 60,
) -> list[Path]:
    config = load_config(config_path)
    market = config["market"]
    start_date = start or market["start"]
    end_date = end if end is not None else market.get("end")
    out_dir = project_path(config_path, config["project"]["data_dir"], "raw", "prices")
    out_dir.mkdir(parents=True, exist_ok=True)

    if market.get("price_source") == "krx_openapi":
        client = KrxOpenApiClient.from_env()
        ticker_markets = {
            _normalize_krx_ticker(ticker): str(market_name).upper()
            for ticker, market_name in market.get("ticker_markets", {}).items()
        }
        default_market = str(market.get("krx_market", "KOSPI")).upper()
        grouped_tickers: dict[str, list[str]] = {}
        for ticker in market["tickers"]:
            normalized_ticker = _normalize_krx_ticker(ticker)
            market_name = ticker_markets.get(normalized_ticker, default_market)
            grouped_tickers.setdefault(market_name, []).append(normalized_ticker)

        cache_dir = project_path(config_path, config["project"]["data_dir"], "raw", "krx_daily_cache")
        state_path = project_path(config_path, config["project"]["data_dir"], "raw", "collection_state.json")
        price_frames = []
        for market_name, tickers in grouped_tickers.items():
            market_start = (start_by_market or {}).get(market_name, start_date)
            try:
                price_frames.append(
                    fetch_stock_prices(
                        client,
                        market_name,
                        tickers,
                        market_start,
                        end_date,
                        cache_dir=cache_dir,
                        state_path=state_path,
                        empty_cache_ttl_minutes=empty_cache_ttl_minutes,
                    )
                )
            except ValueError as exc:
                incremental = start is not None or start_by_market is not None
                if not incremental or "No KRX price data returned" not in str(exc):
                    raise
                print(f"KRX {market_name}: no new rows for {market_start}..{end_date or 'today'}", flush=True)
        if not price_frames:
            existing_paths = [out_dir / f"{_normalize_krx_ticker(ticker)}.parquet" for ticker in market["tickers"]]
            if all(path.exists() for path in existing_paths):
                return existing_paths
            raise ValueError(f"No KRX price data returned for {start_date}..{end_date or 'today'}.")
        prices = pd.concat(price_frames, ignore_index=True)
        written = []
        for ticker, group in prices.groupby("ticker"):
            out_path = out_dir / f"{ticker}.parquet"
            _write_prices(group, out_path)
            written.append(out_path)
        return written

    written: list[Path] = []
    for ticker in market["tickers"]:
        prices = fetch_yfinance(ticker, start_date, end_date)
        out_path = out_dir / f"{ticker}.csv"
        _write_prices(prices, out_path)
        written.append(out_path)
    return written


def _normalize_krx_ticker(ticker: str) -> str:
    return str(ticker).replace(".KS", "").replace(".KQ", "").zfill(6)


def _read_existing_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def latest_price_date_by_market(config_path: str | Path) -> dict[str, str]:
    config = load_config(config_path)
    market = config["market"]
    out_dir = project_path(config_path, config["project"]["data_dir"], "raw", "prices")
    ticker_markets = {
        _normalize_krx_ticker(ticker): str(market_name).upper()
        for ticker, market_name in market.get("ticker_markets", {}).items()
    }
    default_market = str(market.get("krx_market", "KOSPI")).upper()
    dates: dict[str, list[pd.Timestamp]] = {}
    for ticker in market["tickers"]:
        normalized_ticker = _normalize_krx_ticker(ticker)
        market_name = ticker_markets.get(normalized_ticker, default_market)
        path = out_dir / f"{normalized_ticker}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            continue
        dates.setdefault(market_name, []).append(pd.to_datetime(frame["date"]).max())
    return {market_name: min(values).date().isoformat() for market_name, values in dates.items() if values}


def _write_prices(prices: pd.DataFrame, out_path: Path) -> None:
    existing = _read_existing_prices(out_path)
    if existing.empty:
        merged = prices.copy()
    else:
        merged = pd.concat([existing, prices], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged["ticker"] = merged["ticker"].astype(str).map(normalize_ticker)
    merged = merged.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
    if out_path.suffix == ".parquet":
        merged.to_parquet(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start", default=None, help="Override config market.start for incremental collection")
    parser.add_argument("--end", default=None, help="Override config market.end")
    args = parser.parse_args()
    for path in collect_prices(args.config, start=args.start, end=args.end):
        print(path)


if __name__ == "__main__":
    main()
