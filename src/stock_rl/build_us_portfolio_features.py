from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.build_features import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_DEFAULTS,
    add_event_features,
    add_price_features,
    clean_numeric_features,
    read_events,
    read_price_files,
)
from stock_rl.collect_prices import fetch_yfinance
from stock_rl.config import load_config, project_path
from stock_rl.trading_env import normalize_ticker


US_ZERO_FILL_COLUMNS = [
    "trading_value_change",
    "trading_value_zscore_20d",
    "market_cap_change",
    "market_cap_ma60_gap",
    "turnover_value_ratio",
]

US_FUNDAMENTAL_COLUMNS = [
    "sec_revenue_to_assets",
    "sec_net_margin",
    "sec_operating_margin",
    "sec_liabilities_to_assets",
    "sec_cash_to_assets",
    "sec_ocf_to_assets",
    "sec_roe",
    "sec_profitable",
    "sec_positive_ocf",
    "sec_data_age_days",
]

US_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *US_FUNDAMENTAL_COLUMNS]


def collect_us_prices(config_path: str | Path, start: str | None = None, end: str | None = None) -> list[Path]:
    config = load_config(config_path)
    market = config["market"]
    tickers = [*market["tickers"], *market.get("benchmark_tickers", [])]
    start_date = start or market["start"]
    end_date = end if end is not None else market.get("end")
    out_dir = project_path(config_path, config["project"]["data_dir"], "raw", "prices")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker in dict.fromkeys(tickers):
        out_path = out_dir / f"{ticker}.parquet"
        try:
            prices = fetch_yfinance(str(ticker), start_date, end_date)
        except ValueError as exc:
            if out_path.exists():
                print(f"US {ticker}: no new rows for {start_date}..{end_date or 'today'}, keeping existing", flush=True)
                written.append(out_path)
                continue
            raise exc
        _write_prices(prices, out_path)
        written.append(out_path)
    return written


def _write_prices(prices: pd.DataFrame, out_path: Path) -> None:
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        prices = pd.concat([existing, prices], ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["ticker"] = prices["ticker"].astype(str).map(normalize_ticker)
    prices = prices.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
    prices.to_parquet(out_path, index=False)


def build_us_features(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    market = config["market"]
    tickers = [normalize_ticker(ticker) for ticker in market["tickers"]]
    benchmark = normalize_ticker(market.get("primary_benchmark", "SPY"))
    price_dir = project_path(config_path, data_dir, "raw", "prices")
    events_path = project_path(config_path, data_dir, "raw", "events", "events.csv")
    out_dir = project_path(config_path, data_dir, "processed")

    prices = read_price_files(price_dir)
    prices["ticker"] = prices["ticker"].astype(str).map(normalize_ticker)
    asset_prices = prices[prices["ticker"].isin(tickers)].copy()
    benchmark_prices = prices[prices["ticker"].eq(benchmark)].copy()
    if asset_prices.empty:
        raise ValueError("no US asset prices found for configured tickers")
    if benchmark_prices.empty:
        raise ValueError(f"benchmark price data not found: {benchmark}")

    features = add_price_features(asset_prices, config["features"].get("adjusted_price_column", "adj_close"))
    benchmark_features = _benchmark_features(benchmark_prices)
    features["date"] = pd.to_datetime(features["date"])
    drop_columns = [column for column in MARKET_FEATURE_DEFAULTS if column in features.columns]
    drop_columns.extend(
        column
        for column in ["excess_return_1d", "excess_return_5d", "relative_strength_20d", "drawdown_vs_market_60d"]
        if column in features.columns
    )
    features = features.drop(columns=drop_columns)
    features = features.merge(benchmark_features, how="left", on="date")
    for column, value in MARKET_FEATURE_DEFAULTS.items():
        features[column] = features[column].fillna(value)
    features["market_drawdown_60d"] = features["market_drawdown_60d"].fillna(0.0)
    features["excess_return_1d"] = features["return_1d"] - features["market_return_1d"]
    features["excess_return_5d"] = features["return_5d"] - features["market_return_5d"]
    features["relative_strength_20d"] = features["return_20d"] - features["market_return_20d"]
    features["drawdown_vs_market_60d"] = features["drawdown_60d"] - features["market_drawdown_60d"]
    features["relative_strength_regime"] = np.sign(features["relative_strength_20d"]).fillna(0.0)
    features = features.drop(columns=["market_return_20d", "market_drawdown_60d"])
    for column in US_ZERO_FILL_COLUMNS:
        features[column] = pd.to_numeric(features.get(column, 0.0), errors="coerce").fillna(0.0)
    features = add_sec_fundamental_features(features, config_path)
    features = add_event_features(features, read_events(events_path))
    features = clean_numeric_features(features)
    return split_and_write_us(features, config, out_dir)


def add_sec_fundamental_features(features: pd.DataFrame, config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    enriched = features.copy()
    point_in_time_path = project_path(config_path, config["project"]["data_dir"], "processed", "sec_point_in_time_features.csv")
    if point_in_time_path.exists():
        return add_sec_point_in_time_features(enriched, point_in_time_path)

    snapshot_path = project_path(config_path, config["project"]["data_dir"], "processed", "sec_companyfacts_snapshot.csv")
    if not snapshot_path.exists():
        for column in US_FUNDAMENTAL_COLUMNS:
            enriched[column] = 0.0
        return enriched
    snapshot = pd.read_csv(snapshot_path, dtype={"ticker": str})
    if snapshot.empty:
        for column in US_FUNDAMENTAL_COLUMNS:
            enriched[column] = 0.0
        return enriched

    frame = snapshot[["ticker"]].copy()
    for column in [
        "revenue_value",
        "net_income_value",
        "operating_income_value",
        "assets_value",
        "liabilities_value",
        "equity_value",
        "cash_value",
        "operating_cash_flow_value",
    ]:
        frame[column] = pd.to_numeric(snapshot.get(column), errors="coerce")
    assets = frame["assets_value"].replace(0, np.nan)
    revenue = frame["revenue_value"].replace(0, np.nan)
    equity = frame["equity_value"].replace(0, np.nan)
    frame["sec_revenue_to_assets"] = frame["revenue_value"] / assets
    frame["sec_net_margin"] = frame["net_income_value"] / revenue
    frame["sec_operating_margin"] = frame["operating_income_value"] / revenue
    frame["sec_liabilities_to_assets"] = frame["liabilities_value"] / assets
    frame["sec_cash_to_assets"] = frame["cash_value"] / assets
    frame["sec_ocf_to_assets"] = frame["operating_cash_flow_value"] / assets
    frame["sec_roe"] = frame["net_income_value"] / equity
    frame["sec_profitable"] = (frame["net_income_value"] > 0).astype(float)
    frame["sec_positive_ocf"] = (frame["operating_cash_flow_value"] > 0).astype(float)
    filed_columns = [column for column in snapshot.columns if column.endswith("_filed")]
    filed_dates = snapshot[filed_columns].apply(pd.to_datetime, errors="coerce") if filed_columns else pd.DataFrame()
    latest_filed = filed_dates.max(axis=1) if not filed_dates.empty else pd.Series(pd.NaT, index=snapshot.index)
    feature_date = pd.to_datetime(enriched["date"], errors="coerce")
    frame["sec_latest_filed"] = latest_filed
    frame = frame[["ticker", "sec_latest_filed", *US_FUNDAMENTAL_COLUMNS[:-1]]]
    enriched = enriched.merge(frame, how="left", on="ticker")
    enriched["sec_data_age_days"] = (
        feature_date - pd.to_datetime(enriched["sec_latest_filed"], errors="coerce")
    ).dt.days.clip(lower=0)
    enriched = enriched.drop(columns=["sec_latest_filed"])
    for column in US_FUNDAMENTAL_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    return enriched


def add_sec_point_in_time_features(features: pd.DataFrame, point_in_time_path: Path) -> pd.DataFrame:
    facts = pd.read_csv(point_in_time_path, dtype={"ticker": str})
    if facts.empty:
        enriched = features.copy()
        for column in US_FUNDAMENTAL_COLUMNS:
            enriched[column] = 0.0
        return enriched
    facts["filed_date"] = pd.to_datetime(facts["filed_date"], errors="coerce")
    facts = facts.dropna(subset=["filed_date"]).sort_values(["ticker", "filed_date"])
    frames = []
    for ticker, group in features.copy().groupby("ticker", sort=False):
        left = group.sort_values("date").copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        right = facts[facts["ticker"].eq(str(ticker))].copy()
        if right.empty:
            for column in US_FUNDAMENTAL_COLUMNS:
                left[column] = 0.0
            frames.append(left)
            continue
        merged = pd.merge_asof(
            left.sort_values("date"),
            right.sort_values("filed_date"),
            left_on="date",
            right_on="filed_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["sec_data_age_days"] = (merged["date"] - merged["filed_date"]).dt.days.clip(lower=0)
        drop_columns = [column for column in ["filed_date", "period_end", "form"] if column in merged.columns]
        merged = merged.drop(columns=drop_columns)
        for column in US_FUNDAMENTAL_COLUMNS:
            merged[column] = pd.to_numeric(merged.get(column, 0.0), errors="coerce").fillna(0.0)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def split_and_write_us(features: pd.DataFrame, config: dict, out_dir: Path) -> dict[str, Path]:
    train_end = pd.Timestamp(config["features"]["train_end"])
    valid_end = pd.Timestamp(config["features"]["valid_end"])
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = clean_numeric_features(features)
    event_columns = sorted(column for column in clean.columns if column.startswith("event_"))
    model_columns = US_FEATURE_COLUMNS + event_columns
    inference_clean = clean.dropna(subset=model_columns).reset_index(drop=True)
    trainable_clean = inference_clean.dropna(subset=["target_return_1d"]).reset_index(drop=True)
    splits = {
        "daily_features": inference_clean,
        "train": trainable_clean[trainable_clean["date"] <= train_end],
        "valid": trainable_clean[(trainable_clean["date"] > train_end) & (trainable_clean["date"] <= valid_end)],
        "test": trainable_clean[trainable_clean["date"] > valid_end],
    }
    written = {}
    for name, frame in splits.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written[name] = path
    return written


def _benchmark_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values("date").copy()
    close = pd.to_numeric(frame["adj_close"], errors="coerce")
    ret_1d = close.pct_change()
    market_drop = (ret_1d <= -0.02).astype(float)
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]),
            "market_return_1d": ret_1d,
            "market_return_5d": close.pct_change(5),
            "market_return_20d": close.pct_change(20),
            "market_return_60d": close.pct_change(60),
            "market_return_120d": close.pct_change(120),
            "market_ma20_gap": close / close.rolling(20).mean() - 1.0,
            "market_ma60_gap": close / close.rolling(60).mean() - 1.0,
            "market_ma120_gap": close / close.rolling(120).mean() - 1.0,
            "market_volatility_20d": ret_1d.rolling(20).std() * np.sqrt(252),
            "market_drawdown_60d": close / close.rolling(60).max() - 1.0,
            "market_drop_recent_5d": market_drop.shift(1).rolling(5, min_periods=1).max(),
            "market_drop_recent_20d": market_drop.shift(1).rolling(20, min_periods=1).max(),
            "market_trend_regime": np.sign(close / close.rolling(20).mean() - 1.0),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/US_PORTFOLIO_HELD.yaml")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    if args.collect:
        for path in collect_us_prices(args.config, args.start, args.end):
            print(f"price: {path}")
    for name, path in build_us_features(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
