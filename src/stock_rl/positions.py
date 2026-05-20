from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_rl.build_trading_sheet import _load_reference
from stock_rl.config import project_path
from stock_rl.trading_env import normalize_ticker


NUMERIC_COLUMNS = ["quantity", "avg_price", "current_price", "market_value"]


def _latest_close_from_parquet(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty or "close" not in frame.columns:
        return None
    frame = frame.sort_values("date") if "date" in frame.columns else frame
    close = pd.to_numeric(frame.iloc[-1]["close"], errors="coerce")
    return float(close) if pd.notna(close) else None


def _latest_close_from_position_files(config_path: str, ticker: str) -> float | None:
    data_dir = project_path(config_path, "data_krx", "raw")
    paths = [
        data_dir / "position_prices" / f"{ticker}.parquet",
        data_dir / "position_prices_us" / f"{ticker}.parquet",
        data_dir / "prices" / f"{ticker}.parquet",
    ]
    for path in paths:
        close = _latest_close_from_parquet(path)
        if close is not None:
            return close
    return None


def _latest_close_from_krx_cache(config_path: str, ticker: str) -> float | None:
    cache_dir = project_path(config_path, "data_krx", "raw", "krx_daily_cache")
    if not cache_dir.exists():
        return None
    cache_paths = sorted(
        [
            *cache_dir.glob("KOSPI_*.parquet"),
            *cache_dir.glob("KOSDAQ_*.parquet"),
            *cache_dir.glob("ETF_*.parquet"),
            *cache_dir.glob("ETN_*.parquet"),
        ],
        reverse=True,
    )
    for path in cache_paths[:20]:
        frame = pd.read_parquet(path)
        if frame.empty or "ticker" not in frame.columns or "close" not in frame.columns:
            continue
        tickers = frame["ticker"].astype(str).map(normalize_ticker)
        matched = frame[tickers == ticker]
        if matched.empty:
            continue
        close = pd.to_numeric(matched.iloc[-1]["close"], errors="coerce")
        if pd.notna(close):
            return float(close)
    return None


def _latest_close(config_path: str, ticker: str) -> float | None:
    close = _latest_close_from_position_files(config_path, ticker)
    if close is not None:
        return close
    if str(ticker).isdigit():
        return _latest_close_from_krx_cache(config_path, ticker)
    return None


def _reference_names(config_path: str) -> pd.Series:
    reference = _load_reference(config_path)
    if reference.empty:
        return pd.Series(dtype="object")
    reference["ticker"] = reference["ticker"].astype(str).map(normalize_ticker)
    return reference.drop_duplicates("ticker").set_index("ticker")["name"]


def load_positions(path: str | Path, config_path: str) -> pd.DataFrame:
    positions = pd.read_csv(path, dtype={"ticker": str})
    if "ticker" not in positions.columns:
        raise ValueError("positions CSV missing columns: ['ticker']")
    positions["ticker"] = positions["ticker"].map(normalize_ticker)
    if "quantity" not in positions.columns:
        positions["quantity"] = 0.0
    if "name" not in positions.columns:
        positions["name"] = ""
    if "avg_price" not in positions.columns:
        positions["avg_price"] = 0.0
    if "current_price" not in positions.columns:
        positions["current_price"] = 0.0
    if "market_value" not in positions.columns:
        positions["market_value"] = 0.0

    for column in NUMERIC_COLUMNS:
        positions[column] = pd.to_numeric(positions[column], errors="coerce").fillna(0.0)

    names = _reference_names(config_path)
    if not names.empty:
        missing_name = positions["name"].isna() | (positions["name"].astype(str).str.strip() == "")
        positions.loc[missing_name, "name"] = positions.loc[missing_name, "ticker"].map(names).fillna("")

    missing_price = positions["current_price"] <= 0
    if missing_price.any():
        latest_prices = {
            ticker: _latest_close(config_path, str(ticker))
            for ticker in sorted(set(positions.loc[missing_price, "ticker"].astype(str)))
        }
        positions.loc[missing_price, "current_price"] = (
            positions.loc[missing_price, "ticker"].map(latest_prices).fillna(0.0)
        )

    calculated_market_value = positions["quantity"] * positions["current_price"]
    positions["input_market_value"] = positions["market_value"]
    positions["market_value"] = calculated_market_value.where(calculated_market_value > 0, positions["market_value"])
    return positions
