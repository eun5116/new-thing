from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_rl.config import load_config, project_path
from stock_rl.krx_openapi import KrxOpenApiClient, fetch_stock_prices


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


def collect_prices(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    market = config["market"]
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
        price_frames = []
        for market_name, tickers in grouped_tickers.items():
            price_frames.append(
                fetch_stock_prices(
                    client,
                    market_name,
                    tickers,
                    market["start"],
                    market.get("end"),
                    cache_dir=cache_dir,
                )
            )
        prices = pd.concat(price_frames, ignore_index=True)
        written = []
        for ticker, group in prices.groupby("ticker"):
            out_path = out_dir / f"{ticker}.parquet"
            group.to_parquet(out_path, index=False)
            written.append(out_path)
        return written

    written: list[Path] = []
    for ticker in market["tickers"]:
        prices = fetch_yfinance(ticker, market["start"], market.get("end"))
        out_path = out_dir / f"{ticker}.csv"
        prices.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def _normalize_krx_ticker(ticker: str) -> str:
    return str(ticker).replace(".KS", "").replace(".KQ", "").zfill(6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    for path in collect_prices(args.config):
        print(path)


if __name__ == "__main__":
    main()
