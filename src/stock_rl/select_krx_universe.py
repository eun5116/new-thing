from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stock_rl.config import project_path
from stock_rl.krx_openapi import KrxOpenApiClient, normalize_stock_daily


DEFAULT_EXCLUDE_KEYWORDS = (
    "스팩",
    "리츠",
    "우선주",
)


def latest_stock_daily(
    client: KrxOpenApiClient,
    market: str,
    end: str | None = None,
    lookback_days: int = 14,
) -> pd.DataFrame:
    end_date = pd.Timestamp(end).date() if end else dt.date.today()
    for offset in range(lookback_days + 1):
        day = end_date - dt.timedelta(days=offset)
        raw = client.fetch_stock_daily(market, day)
        normalized = normalize_stock_daily(raw, market=market, tickers=None)
        if not normalized.empty:
            return normalized
    raise ValueError(f"No {market} daily data found in the last {lookback_days} days.")


def read_issue_base(data_dir: str, market: str) -> pd.DataFrame:
    path = project_path("configs/krx_kospi.yaml", data_dir, "raw", "reference", f"{market.lower()}_issue_base.parquet")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    return frame


def filter_common_stock_candidates(daily: pd.DataFrame, issue_base: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    if not issue_base.empty:
        keep_cols = [column for column in ["ticker", "name", "security_group", "stock_type", "list_date"] if column in issue_base.columns]
        frame = frame.merge(issue_base[keep_cols], how="left", on="ticker")
        if "security_group" in frame.columns:
            frame = frame[frame["security_group"].fillna("").str.contains("주권", na=False)]
        if "name" in frame.columns:
            name = frame["name"].fillna("")
            for keyword in DEFAULT_EXCLUDE_KEYWORDS:
                name = name.mask(name.str.contains(keyword, regex=False), "")
            frame = frame[name != ""]
    frame = frame[pd.to_numeric(frame["close"], errors="coerce") > 0]
    frame = frame[pd.to_numeric(frame["trading_value"], errors="coerce").fillna(0) > 0]
    return frame


def select_universe(
    data_dir: str,
    kospi_count: int,
    kosdaq_count: int,
    end: str | None = None,
) -> pd.DataFrame:
    client = KrxOpenApiClient.from_env()
    selections = []
    for market, count in {"KOSPI": kospi_count, "KOSDAQ": kosdaq_count}.items():
        if count <= 0:
            continue
        daily = latest_stock_daily(client, market, end=end)
        issue_base = read_issue_base(data_dir, market)
        candidates = filter_common_stock_candidates(daily, issue_base)
        selected = candidates.sort_values(["trading_value", "market_cap"], ascending=False).head(count).copy()
        selected["rank_in_market"] = range(1, len(selected) + 1)
        selections.append(selected)
    if not selections:
        raise ValueError("No universe selected.")
    universe = pd.concat(selections, ignore_index=True)
    keep = [
        "ticker",
        "market",
        "rank_in_market",
        "name",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "date",
    ]
    return universe[[column for column in keep if column in universe.columns]].sort_values(["market", "rank_in_market"])


def write_universe_config(
    universe: pd.DataFrame,
    base_config_path: str | Path,
    output_config_path: str | Path,
    model_name: str,
    total_timesteps: int,
    action_mode: str,
    reward_mode: str,
    start: str | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
) -> Path:
    with Path(base_config_path).open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    tickers = universe["ticker"].astype(str).str.zfill(6).tolist()
    ticker_markets = dict(zip(tickers, universe["market"].astype(str), strict=True))
    config["market"]["tickers"] = tickers
    config["market"]["ticker_markets"] = ticker_markets
    config["market"]["krx_market"] = "MIXED"
    config["training"]["train_scope"] = "multi_ticker"
    config["training"]["model_name"] = model_name
    config["training"]["total_timesteps"] = int(total_timesteps)
    config["trading"]["action_mode"] = action_mode
    config["trading"]["reward_mode"] = reward_mode
    if start:
        config["market"]["start"] = start
    if train_end:
        config["features"]["train_end"] = train_end
    if valid_end:
        config["features"]["valid_end"] = valid_end

    path = Path(output_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a liquid KRX universe and write an experiment config.")
    parser.add_argument("--data-dir", default="data_krx")
    parser.add_argument("--base-config", default="configs/KRX_E024_target_hybrid_penalty_025.yaml")
    parser.add_argument("--out-config", default="configs/KRX_E026_liquid_universe_target_hybrid.yaml")
    parser.add_argument("--out-universe", default="data_krx/raw/reference/liquid_universe_E026.csv")
    parser.add_argument("--kospi-count", type=int, default=30)
    parser.add_argument("--kosdaq-count", type=int, default=20)
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-name", default="ppo_KRX_E026_liquid_universe_target_hybrid")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--action-mode", default="target_position")
    parser.add_argument("--reward-mode", default="ma20_60_drawdown_hybrid")
    parser.add_argument("--start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--valid-end", default=None)
    args = parser.parse_args()

    universe = select_universe(args.data_dir, args.kospi_count, args.kosdaq_count, args.end)
    universe_path = Path(args.out_universe)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(universe_path, index=False)
    config_path = write_universe_config(
        universe,
        args.base_config,
        args.out_config,
        args.model_name,
        args.total_timesteps,
        args.action_mode,
        args.reward_mode,
        args.start,
        args.train_end,
        args.valid_end,
    )
    print(f"universe: {universe_path} rows={len(universe)}")
    print(f"config: {config_path}")


if __name__ == "__main__":
    main()
