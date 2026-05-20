from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config import load_config, project_path


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "ma5_gap",
    "ma20_gap",
    "ma60_gap",
    "ma120_gap",
    "volatility_20d",
    "volume_change",
    "rsi_14",
    "macd",
    "drawdown_20d",
    "drawdown_60d",
    "high_20d_gap",
    "low_20d_gap",
    "volume_zscore_20d",
    "volatility_60d",
    "trading_value_change",
    "trading_value_zscore_20d",
    "market_cap_change",
    "market_cap_ma60_gap",
    "turnover_value_ratio",
    "market_return_1d",
    "market_return_5d",
    "market_return_60d",
    "market_return_120d",
    "market_ma20_gap",
    "market_ma60_gap",
    "market_ma120_gap",
    "market_volatility_20d",
    "excess_return_1d",
    "excess_return_5d",
    "relative_strength_20d",
    "drawdown_vs_market_60d",
    "ma20_60_signal",
    "ma20_60_gap",
    "ma20_60_position",
    "market_drop_recent_5d",
    "market_drop_recent_20d",
    "relative_strength_regime",
    "market_trend_regime",
    "event_recent_5d",
    "event_recent_20d",
]

MARKET_FEATURE_DEFAULTS = {
    "market_return_1d": 0.0,
    "market_return_5d": 0.0,
    "market_return_60d": 0.0,
    "market_return_120d": 0.0,
    "market_ma20_gap": 0.0,
    "market_ma60_gap": 0.0,
    "market_ma120_gap": 0.0,
    "market_volatility_20d": 0.0,
    "market_drop_recent_5d": 0.0,
    "market_drop_recent_20d": 0.0,
    "market_trend_regime": 0.0,
}


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker)
    return value.zfill(6) if value.isdigit() else value


def clean_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    """Remove non-finite values from model-facing numeric columns."""
    clean = features.copy()
    numeric_cols = clean.select_dtypes(include=[np.number]).columns
    clean[numeric_cols] = clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return clean


def read_price_files(price_dir: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    files = sorted(price_dir.glob("*.parquet")) or sorted(price_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no price files found in {price_dir}")
    wanted = {_normalize_ticker(str(ticker).replace(".KS", "").replace(".KQ", "")) for ticker in tickers or []}
    frames = []
    for path in files:
        if wanted and _normalize_ticker(path.stem) not in wanted:
            continue
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path, parse_dates=["date"], dtype={"ticker": str}))
    if not frames:
        raise FileNotFoundError(f"no matching price files found in {price_dir}")
    prices = pd.concat(frames, ignore_index=True)
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"price data missing columns: {sorted(missing)}")
    prices["ticker"] = prices["ticker"].astype(str).map(_normalize_ticker)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_price_features(prices: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    frames = []
    for _, group in prices.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        px = g[price_col].astype(float)
        volume = g["volume"].astype(float)
        trading_value = pd.to_numeric(g.get("trading_value", pd.Series(0.0, index=g.index)), errors="coerce")
        market_cap = pd.to_numeric(g.get("market_cap", pd.Series(0.0, index=g.index)), errors="coerce")
        ret_1d = px.pct_change()

        g["return_1d"] = ret_1d
        g["return_5d"] = px.pct_change(5)
        g["return_20d"] = px.pct_change(20)
        g["return_60d"] = px.pct_change(60)
        g["return_120d"] = px.pct_change(120)
        g["ma5_gap"] = px / px.rolling(5).mean() - 1.0
        g["ma20_gap"] = px / px.rolling(20).mean() - 1.0
        g["ma60_gap"] = px / px.rolling(60).mean() - 1.0
        g["ma120_gap"] = px / px.rolling(120).mean() - 1.0
        g["volatility_20d"] = ret_1d.rolling(20).std() * np.sqrt(252)
        g["volatility_60d"] = ret_1d.rolling(60).std() * np.sqrt(252)
        g["volume_change"] = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        gain = ret_1d.clip(lower=0.0)
        loss = -ret_1d.clip(upper=0.0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        g["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        ema12 = px.ewm(span=12, adjust=False).mean()
        ema26 = px.ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 / ema26 - 1.0
        rolling_high_20 = px.rolling(20).max()
        rolling_low_20 = px.rolling(20).min()
        rolling_high_60 = px.rolling(60).max()
        g["drawdown_20d"] = px / rolling_high_20 - 1.0
        g["drawdown_60d"] = px / rolling_high_60 - 1.0
        g["high_20d_gap"] = px / rolling_high_20 - 1.0
        g["low_20d_gap"] = px / rolling_low_20 - 1.0
        volume_mean_20 = volume.rolling(20).mean()
        volume_std_20 = volume.rolling(20).std()
        g["volume_zscore_20d"] = (volume - volume_mean_20) / volume_std_20
        trading_value_mean_20 = trading_value.rolling(20).mean()
        trading_value_std_20 = trading_value.rolling(20).std()
        g["trading_value_change"] = trading_value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        g["trading_value_zscore_20d"] = (trading_value - trading_value_mean_20) / trading_value_std_20
        g["market_cap_change"] = market_cap.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        g["market_cap_ma60_gap"] = market_cap / market_cap.rolling(60).mean() - 1.0
        g["turnover_value_ratio"] = trading_value / market_cap.replace(0, np.nan)
        ma20 = px.rolling(20).mean()
        ma60 = px.rolling(60).mean()
        g["ma20_60_signal"] = (ma20 > ma60).astype(float)
        g["ma20_60_gap"] = ma20 / ma60 - 1.0
        g["ma20_60_position"] = g["ma20_60_signal"].shift(1).fillna(0.0)
        for column, value in MARKET_FEATURE_DEFAULTS.items():
            g[column] = value
        g["excess_return_1d"] = 0.0
        g["excess_return_5d"] = 0.0
        g["relative_strength_20d"] = 0.0
        g["drawdown_vs_market_60d"] = 0.0
        g["relative_strength_regime"] = 0.0
        g["event_recent_5d"] = 0.0
        g["event_recent_20d"] = 0.0
        g["target_return_1d"] = px.shift(-1) / px - 1.0
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def read_index_files(index_dir: Path) -> pd.DataFrame:
    files = sorted(index_dir.glob("*.parquet")) or sorted(index_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path, parse_dates=["date"]))
    indices = pd.concat(frames, ignore_index=True)
    if indices.empty:
        return indices
    indices["date"] = pd.to_datetime(indices["date"], errors="coerce")
    return indices.sort_values(["market", "index_name", "date"]).reset_index(drop=True)


def add_market_features(features: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    enriched = features.copy()
    defaults = MARKET_FEATURE_DEFAULTS
    relative_defaults = {
        "excess_return_1d": 0.0,
        "excess_return_5d": 0.0,
        "relative_strength_20d": 0.0,
        "drawdown_vs_market_60d": 0.0,
    }
    if indices.empty or "market" not in enriched.columns:
        for column, value in defaults.items():
            enriched[column] = value
        for column, value in relative_defaults.items():
            enriched[column] = value
        return enriched
    drop_columns = [column for column in [*defaults, *relative_defaults] if column in enriched.columns]
    enriched = enriched.drop(columns=drop_columns)

    preferred = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}
    frames = []
    for market, index_name in preferred.items():
        group = indices[(indices["market"] == market) & (indices["index_name"] == index_name)].copy()
        if group.empty:
            continue
        group = group.sort_values("date")
        close = pd.to_numeric(group["close"], errors="coerce")
        ret_1d = close.pct_change()
        group["market_return_1d"] = ret_1d
        group["market_return_5d"] = close.pct_change(5)
        group["market_return_60d"] = close.pct_change(60)
        group["market_return_120d"] = close.pct_change(120)
        group["market_ma20_gap"] = close / close.rolling(20).mean() - 1.0
        group["market_ma60_gap"] = close / close.rolling(60).mean() - 1.0
        group["market_ma120_gap"] = close / close.rolling(120).mean() - 1.0
        group["market_volatility_20d"] = ret_1d.rolling(20).std() * np.sqrt(252)
        group["market_drawdown_60d"] = close / close.rolling(60).max() - 1.0
        market_drop = (ret_1d <= -0.02).astype(float)
        group["market_drop_recent_5d"] = market_drop.shift(1).rolling(5, min_periods=1).max()
        group["market_drop_recent_20d"] = market_drop.shift(1).rolling(20, min_periods=1).max()
        group["market_trend_regime"] = np.sign(group["market_ma20_gap"]).fillna(0.0)
        frames.append(group[["date", "market", "market_drawdown_60d", *defaults.keys()]])

    if not frames:
        for column, value in defaults.items():
            enriched[column] = value
        return enriched

    market_features = pd.concat(frames, ignore_index=True)
    enriched["date"] = pd.to_datetime(enriched["date"], errors="coerce")
    enriched = enriched.merge(market_features, how="left", on=["date", "market"])
    for column, value in defaults.items():
        enriched[column] = enriched[column].fillna(value)
    enriched["market_drawdown_60d"] = enriched["market_drawdown_60d"].fillna(0.0)
    enriched["excess_return_1d"] = enriched["return_1d"] - enriched["market_return_1d"]
    enriched["excess_return_5d"] = enriched["return_5d"] - enriched["market_return_5d"]
    enriched["relative_strength_20d"] = enriched["return_20d"] - enriched["market_return_1d"].rolling(20).sum()
    enriched["drawdown_vs_market_60d"] = enriched["drawdown_60d"] - enriched["market_drawdown_60d"]
    enriched["relative_strength_regime"] = np.sign(enriched["relative_strength_20d"]).fillna(0.0)
    enriched = enriched.drop(columns=["market_drawdown_60d"])
    for column, value in relative_defaults.items():
        enriched[column] = enriched[column].fillna(value)
    enriched["relative_strength_regime"] = enriched["relative_strength_regime"].fillna(0.0)
    return enriched


def read_events(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame(columns=["effective_date", "ticker", "event_type", "event_score"])
    events = pd.read_csv(events_path)
    if events.empty:
        return pd.DataFrame(columns=["effective_date", "ticker", "event_type", "event_score"])
    events["effective_date"] = pd.to_datetime(events["effective_date"])
    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(1.0)
    return events


def add_event_features(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        features["event_any"] = 0.0
        return add_event_recent_features(features)

    event_scores = (
        events.assign(event_col=lambda x: "event_" + x["event_type"].astype(str))
        .groupby(["effective_date", "ticker", "event_col"], as_index=False)["event_score"]
        .sum()
    )
    wide = event_scores.pivot_table(
        index=["effective_date", "ticker"],
        columns="event_col",
        values="event_score",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    wide = wide.rename(columns={"effective_date": "date"})

    ticker_specific = features.merge(wide, how="left", on=["date", "ticker"])
    all_market = wide[wide["ticker"] == "ALL"].drop(columns=["ticker"])
    if not all_market.empty:
        ticker_specific = ticker_specific.merge(all_market, how="left", on="date", suffixes=("", "_all"))

    event_cols = [c for c in ticker_specific.columns if c.startswith("event_")]
    ticker_specific[event_cols] = ticker_specific[event_cols].fillna(0.0)
    ticker_specific["event_any"] = (ticker_specific[event_cols].abs().sum(axis=1) > 0).astype(float)
    ticker_specific = add_event_recent_features(ticker_specific)
    return ticker_specific


def add_event_recent_features(features: pd.DataFrame) -> pd.DataFrame:
    enriched = features.sort_values(["ticker", "date"]).copy()
    if "event_any" not in enriched.columns:
        enriched["event_any"] = 0.0
    frames = []
    for _, group in enriched.groupby("ticker", sort=False):
        g = group.copy()
        event_any = g["event_any"].astype(float)
        g["event_recent_5d"] = event_any.shift(1).rolling(5, min_periods=1).max().fillna(0.0)
        g["event_recent_20d"] = event_any.shift(1).rolling(20, min_periods=1).max().fillna(0.0)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def split_and_write(features: pd.DataFrame, config: dict, out_dir: Path) -> dict[str, Path]:
    train_end = pd.Timestamp(config["features"]["train_end"])
    valid_end = pd.Timestamp(config["features"]["valid_end"])
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = clean_numeric_features(features)
    event_columns = sorted(column for column in clean.columns if column.startswith("event_"))
    model_columns = FEATURE_COLUMNS + event_columns
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


def build_features(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    price_dir = project_path(config_path, data_dir, "raw", "prices")
    index_dir = project_path(config_path, data_dir, "raw", "indices")
    events_path = project_path(config_path, data_dir, "raw", "events", "events.csv")
    out_dir = project_path(config_path, data_dir, "processed")

    prices = read_price_files(price_dir, config["market"].get("tickers"))
    features = add_price_features(prices, config["features"].get("adjusted_price_column", "adj_close"))
    features = add_market_features(features, read_index_files(index_dir))
    features = add_event_features(features, read_events(events_path))
    return split_and_write(features, config, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    for name, path in build_features(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
