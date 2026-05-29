from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.build_features import build_features
from stock_rl.build_target_change_report import build_target_change_report
from stock_rl.build_trading_sheet import build_trading_sheet
from stock_rl.collect_krx_reference import collect_index_series, latest_index_date_by_market
from stock_rl.collect_prices import collect_prices, latest_price_date_by_market
from stock_rl.config import load_config, project_path
from stock_rl.generate_current_targets import generate_targets
from stock_rl.krx_openapi import KrxOpenApiClient


def infer_incremental_start(config_path: str | Path, features_name: str = "daily_features.parquet") -> str:
    config = load_config(config_path)
    features_path = project_path(config_path, config["project"]["data_dir"], "processed", features_name)
    if not features_path.exists():
        return str(config["market"]["start"])
    features = pd.read_parquet(features_path, columns=["date"])
    if features.empty:
        return str(config["market"]["start"])
    latest_date = pd.to_datetime(features["date"]).max().date()
    return (pd.Timestamp(latest_date) + pd.Timedelta(days=1)).date().isoformat()


def infer_market_collection_starts(config_path: str | Path) -> dict[str, str]:
    config = load_config(config_path)
    latest = latest_price_date_by_market(config_path)
    starts = {}
    markets = _config_markets(config)
    for market in markets:
        latest_date = latest.get(market)
        if latest_date:
            starts[market] = _next_day(latest_date)
        else:
            starts[market] = str(config["market"]["start"])
    return starts


def infer_index_collection_starts(config_path: str | Path) -> dict[str, str]:
    config = load_config(config_path)
    latest = latest_index_date_by_market(config["project"]["data_dir"], config_path=config_path)
    starts = {}
    for market in _config_index_markets(config):
        latest_date = latest.get(market)
        starts[market] = _next_day(latest_date) if latest_date else str(config["market"]["start"])
    return starts


def _config_index_markets(config: dict) -> list[str]:
    return [market for market in _config_markets(config) if market in {"KOSPI", "KOSDAQ"}]


def _config_markets(config: dict) -> list[str]:
    markets = sorted(set(str(market).upper() for market in config["market"].get("ticker_markets", {}).values()))
    if not markets:
        markets = [str(config["market"].get("krx_market", "KOSPI")).upper()]
    if "MIXED" in markets:
        return ["KOSPI", "KOSDAQ"]
    return markets


def _next_day(value: str) -> str:
    return (pd.Timestamp(value) + pd.Timedelta(days=1)).date().isoformat()


def _target_summary(path: Path) -> dict[str, object]:
    targets = pd.read_csv(path)
    as_of_dates = pd.to_datetime(targets["as_of_date"])
    feature_dates = pd.to_datetime(targets["feature_date"])
    feature_lag_days = (as_of_dates - feature_dates).dt.days
    return {
        "as_of_date": str(targets["as_of_date"].max()),
        "tickers": int(targets["ticker"].nunique()),
        "avg_target_pct": float((targets["target_ratio"].astype(float) * 100.0).mean()),
        "full_target_count": int((targets["target_ratio"].astype(float) >= 0.999).sum()),
        "capped_count": int((targets["cap_reason"] == "none").sum()),
        "stale_count": int((feature_lag_days > 0).sum()),
        "max_feature_lag_days": int(feature_lag_days.max()),
    }


def _index_summary(paths: list[Path]) -> dict[str, str]:
    summary = {}
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            continue
        summary[path.stem.replace("_indices", "").upper()] = pd.to_datetime(frame["date"]).max().date().isoformat()
    return summary


def update_daily_targets(
    config_path: str,
    rule: str = "strong_trend_full_else070",
    start: str | None = None,
    end: str | None = None,
    skip_collect: bool = False,
    skip_indices: bool = False,
    skip_features: bool = False,
) -> dict[str, object]:
    config = load_config(config_path)
    stock_starts = {market: start for market in _config_markets(config)} if start else infer_market_collection_starts(config_path)
    index_markets = _config_index_markets(config)
    index_starts = {market: start for market in index_markets} if start else infer_index_collection_starts(config_path)
    resolved_start = start or min([*stock_starts.values(), *index_starts.values()])
    written_prices = []
    if not skip_collect:
        written_prices = collect_prices(config_path, start=start, end=end, start_by_market=stock_starts)
    index_paths = []
    if not skip_indices:
        client = KrxOpenApiClient.from_env()
        for market in index_markets:
            index_paths.append(
                collect_index_series(
                    client,
                    config["project"]["data_dir"],
                    market,
                    index_starts[market],
                    end=end,
                    allow_empty=True,
                    config_path=config_path,
                )
            )
    feature_paths = {}
    if not skip_features:
        feature_paths = build_features(config_path)
    target_path = generate_targets(config_path, rule_name=rule)
    sheet_paths = build_trading_sheet(config_path, rule=rule, target_path=str(target_path))
    change_paths = build_target_change_report(config_path, rule=rule, current_target_path=str(target_path))
    summary = _target_summary(target_path)
    return {
        "collection_start": resolved_start,
        "stock_collection_starts": stock_starts,
        "index_collection_starts": index_starts,
        "collection_end": end or "today",
        "price_files": len(written_prices),
        "index_files": len(index_paths),
        "index_latest_dates": _index_summary(index_paths),
        "features": feature_paths,
        "target_path": target_path,
        "sheet_paths": sheet_paths,
        "change_paths": change_paths,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--start", default=None, help="Override inferred incremental collection start")
    parser.add_argument("--end", default=None, help="Override collection end date")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-indices", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    result = update_daily_targets(
        args.config,
        rule=args.rule,
        start=args.start,
        end=args.end,
        skip_collect=args.skip_collect,
        skip_indices=args.skip_indices,
        skip_features=args.skip_features,
    )
    summary = result["summary"]
    print(f"collection: {result['collection_start']}..{result['collection_end']}")
    stock_starts = " ".join(f"{market}={date}" for market, date in sorted(result["stock_collection_starts"].items()))
    index_starts = " ".join(f"{market}={date}" for market, date in sorted(result["index_collection_starts"].items()))
    print(f"stock_starts: {stock_starts}")
    print(f"index_starts: {index_starts}")
    print(f"price_files: {result['price_files']}")
    print(f"index_files: {result['index_files']}")
    if result["index_latest_dates"]:
        index_latest = " ".join(f"{market}={date}" for market, date in sorted(result["index_latest_dates"].items()))
        print(f"index_latest: {index_latest}")
    print(f"target: {result['target_path']}")
    print(f"sheet_csv: {result['sheet_paths']['csv']}")
    print(f"sheet_markdown: {result['sheet_paths']['markdown']}")
    if result["sheet_paths"].get("png"):
        print(f"sheet_png: {result['sheet_paths']['png']}")
    if result["change_paths"]:
        print(f"changes_csv: {result['change_paths']['csv']}")
        print(f"changes_markdown: {result['change_paths']['markdown']}")
        if result["change_paths"].get("png"):
            print(f"changes_png: {result['change_paths']['png']}")
    print(
        "summary: "
        f"as_of={summary['as_of_date']} "
        f"tickers={summary['tickers']} "
        f"avg_target={summary['avg_target_pct']:.1f}% "
        f"full={summary['full_target_count']} "
        f"capped={summary['capped_count']} "
        f"stale={summary['stale_count']} "
        f"max_lag={summary['max_feature_lag_days']}d"
    )


if __name__ == "__main__":
    main()
