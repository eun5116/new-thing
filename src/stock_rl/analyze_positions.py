from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.build_features import add_price_features
from stock_rl.build_trading_sheet import _load_reference, _markdown_table
from stock_rl.config import project_path
from stock_rl.collect_prices import fetch_yfinance
from stock_rl.krx_openapi import KrxOpenApiClient, fetch_stock_prices
from stock_rl.positions import load_positions
from stock_rl.report_png import render_position_analysis_png
from stock_rl.trading_env import normalize_ticker


def _load_positions(path: str | Path) -> pd.DataFrame:
    positions = pd.read_csv(path, dtype={"ticker": str})
    required = {"ticker", "name", "quantity", "avg_price", "current_price", "market_value"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"positions CSV missing columns: {sorted(missing)}")
    positions["ticker"] = positions["ticker"].map(normalize_ticker)
    for column in ["quantity", "avg_price", "current_price", "market_value"]:
        positions[column] = pd.to_numeric(positions[column], errors="coerce").fillna(0.0)
    calculated_market_value = positions["quantity"] * positions["current_price"]
    positions["input_market_value"] = positions["market_value"]
    positions["market_value"] = calculated_market_value.where(calculated_market_value > 0, positions["market_value"])
    return positions


def _latest_target_path(config_path: str, rule: str) -> Path | None:
    paths = sorted(project_path(config_path, "reports").glob(f"current_targets_*_{rule}.csv"))
    return paths[-1] if paths else None


def _load_targets(config_path: str, rule: str) -> pd.DataFrame:
    path = _latest_target_path(config_path, rule)
    if path is None:
        return pd.DataFrame(columns=["ticker"])
    targets = pd.read_csv(path, dtype={"ticker": str})
    targets["ticker"] = targets["ticker"].map(normalize_ticker)
    return targets


def _krx_reference_map(config_path: str) -> pd.DataFrame:
    reference = _load_reference(config_path)
    reference["ticker"] = reference["ticker"].map(normalize_ticker)
    return reference


def collect_held_krx_stock_prices(
    config_path: str,
    positions: pd.DataFrame,
    start: str = "2025-01-01",
    end: str | None = None,
) -> list[Path]:
    reference = _krx_reference_map(config_path)
    numeric_tickers = {ticker for ticker in positions["ticker"] if str(ticker).isdigit()}
    stock_reference = reference[reference["ticker"].isin(numeric_tickers)]

    client = KrxOpenApiClient.from_env()
    data_dir = "data_krx"
    out_dir = project_path(config_path, data_dir, "raw", "position_prices")
    cache_dir = project_path(config_path, data_dir, "raw", "krx_daily_cache")
    state_path = project_path(config_path, data_dir, "raw", "collection_state.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if not stock_reference.empty:
        for market, group in stock_reference.groupby("market"):
            tickers = sorted(group["ticker"].unique())
            prices = fetch_stock_prices(
                client,
                market,
                tickers,
                start,
                end,
                cache_dir=cache_dir,
                state_path=state_path,
                empty_cache_ttl_minutes=60,
            )
            for ticker, ticker_frame in prices.groupby("ticker"):
                path = out_dir / f"{ticker}.parquet"
                ticker_frame.to_parquet(path, index=False)
                written.append(path)

    unmapped_numeric_tickers = sorted(numeric_tickers.difference(set(stock_reference["ticker"])))
    for market in ["ETF", "ETN"]:
        if not unmapped_numeric_tickers:
            break
        try:
            prices = fetch_stock_prices(
                client,
                market,
                unmapped_numeric_tickers,
                start,
                end,
                cache_dir=cache_dir,
                state_path=state_path,
                empty_cache_ttl_minutes=60,
            )
        except Exception as exc:
            print(f"KRX {market}: skipped unmapped numeric tickers ({exc})", flush=True)
            continue
        found = set()
        for ticker, ticker_frame in prices.groupby("ticker"):
            path = out_dir / f"{ticker}.parquet"
            ticker_frame.to_parquet(path, index=False)
            written.append(path)
            found.add(str(ticker))
        unmapped_numeric_tickers = [ticker for ticker in unmapped_numeric_tickers if ticker not in found]
    written.extend(collect_held_krx_product_prices_yfinance(config_path, unmapped_numeric_tickers, start, end))
    return written


def collect_held_krx_product_prices_yfinance(
    config_path: str,
    tickers: list[str],
    start: str = "2025-01-01",
    end: str | None = None,
) -> list[Path]:
    if not tickers:
        return []
    out_dir = project_path(config_path, "data_krx", "raw", "position_prices")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker in sorted(tickers):
        prices = pd.DataFrame()
        for suffix in [".KS", ".KQ"]:
            try:
                prices = fetch_yfinance(f"{ticker}{suffix}", start, end)
            except Exception:
                continue
            if not prices.empty:
                break
        if prices.empty:
            continue
        prices["ticker"] = ticker
        path = out_dir / f"{ticker}.parquet"
        prices.to_parquet(path, index=False)
        written.append(path)
    return written


def collect_held_us_prices(
    config_path: str,
    positions: pd.DataFrame,
    start: str = "2025-01-01",
    end: str | None = None,
) -> list[Path]:
    tickers = sorted(str(ticker) for ticker in positions["ticker"] if not str(ticker).isdigit())
    if not tickers:
        return []
    out_dir = project_path(config_path, "data_krx", "raw", "position_prices_us")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker in tickers:
        prices = fetch_yfinance(ticker, start, end)
        prices["ticker"] = ticker
        path = out_dir / f"{ticker}.parquet"
        prices.to_parquet(path, index=False)
        written.append(path)
    return written


def _load_position_price_features(config_path: str) -> pd.DataFrame:
    price_dirs = [
        project_path(config_path, "data_krx", "raw", "position_prices"),
        project_path(config_path, "data_krx", "raw", "position_prices_us"),
    ]
    frames = []
    for price_dir in price_dirs:
        for path in sorted(price_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["ticker"])
    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].map(normalize_ticker)
    features = add_price_features(prices).sort_values(["ticker", "date"])
    return features.groupby("ticker", as_index=False).tail(1)


def _asset_scope(ticker: str, model_universe: set[str], reference_tickers: set[str]) -> str:
    if ticker in model_universe:
        return "model_universe"
    if ticker in reference_tickers:
        return "krx_stock_outside_model"
    if str(ticker).isdigit():
        return "krx_etf_or_unmapped"
    return "us_or_global"


def _trend_status(row: pd.Series) -> str:
    if pd.isna(row.get("ma20_60_position")):
        return "unknown"
    if float(row.get("ma20_60_position", 0.0)) > 0:
        return "uptrend"
    return "downtrend"


def analyze_positions(
    config_path: str,
    positions_path: str = "data_krx/raw/positions/current_positions.csv",
    rule: str = "strong_trend_full_else070",
    collect_krx: bool = True,
    collect_us: bool = True,
    krx_start: str = "2025-01-01",
    us_start: str = "2025-01-01",
    out_dir: str | None = None,
) -> dict[str, Path]:
    positions = load_positions(positions_path, config_path)
    if collect_krx:
        collect_held_krx_stock_prices(config_path, positions, start=krx_start)
    if collect_us:
        collect_held_us_prices(config_path, positions, start=us_start)
    positions = load_positions(positions_path, config_path)
    targets = _load_targets(config_path, rule)
    reference = _krx_reference_map(config_path)
    krx_features = _load_position_price_features(config_path)

    target_map = targets.set_index("ticker") if not targets.empty else pd.DataFrame()
    reference_map = reference.set_index("ticker") if not reference.empty else pd.DataFrame()
    feature_map = krx_features.set_index("ticker") if not krx_features.empty else pd.DataFrame()
    total_value = float(positions["market_value"].sum())
    model_universe = set(targets["ticker"]) if not targets.empty else set()
    reference_tickers = set(reference["ticker"]) if not reference.empty else set()

    rows = []
    for _, position in positions.iterrows():
        ticker = str(position["ticker"])
        invested = float(position["quantity"] * position["avg_price"])
        market_value = float(position["market_value"])
        pnl = market_value - invested
        pnl_pct = pnl / invested if invested > 0 else 0.0
        target_row = target_map.loc[ticker] if ticker in target_map.index else None
        ref_row = reference_map.loc[ticker] if ticker in reference_map.index else None
        feature_row = feature_map.loc[ticker] if ticker in feature_map.index else None
        target_ratio = float(target_row["target_ratio"]) if target_row is not None else 0.0
        rows.append(
            {
                "ticker": ticker,
                "name": position["name"],
                "asset_scope": _asset_scope(ticker, model_universe, reference_tickers),
                "reference_name": str(ref_row.get("name", "")) if ref_row is not None else "",
                "market": str(ref_row.get("market", "")) if ref_row is not None else "",
                "quantity": float(position["quantity"]),
                "avg_price": float(position["avg_price"]),
                "current_price": float(position["current_price"]),
                "market_value": market_value,
                "current_weight_pct": round(market_value / total_value * 100.0, 2) if total_value > 0 else 0.0,
                "pnl": round(pnl, 0),
                "pnl_pct": round(pnl_pct * 100.0, 2),
                "model_target_ratio_pct": round(target_ratio * 100.0, 2),
                "model_cap_reason": str(target_row["cap_reason"]) if target_row is not None else "not_in_model_universe",
                "trend_status": _trend_status(feature_row) if feature_row is not None else "unknown",
                "return_20d_pct": round(float(feature_row.get("return_20d", 0.0)) * 100.0, 2) if feature_row is not None else None,
                "return_60d_pct": round(float(feature_row.get("return_60d", 0.0)) * 100.0, 2) if feature_row is not None else None,
                "return_120d_pct": round(float(feature_row.get("return_120d", 0.0)) * 100.0, 2) if feature_row is not None else None,
                "volatility_20d_pct": round(float(feature_row.get("volatility_20d", 0.0)) * 100.0, 2) if feature_row is not None else None,
                "drawdown_60d_pct": round(float(feature_row.get("drawdown_60d", 0.0)) * 100.0, 2) if feature_row is not None else None,
                "feature_date": str(pd.to_datetime(feature_row.get("date")).date()) if feature_row is not None else "",
                "note": _note_for_position(ticker, model_universe, reference_tickers),
            }
        )

    result = pd.DataFrame(rows).sort_values("market_value", ascending=False)
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.today().strftime("%Y%m%d")
    csv_path = output_dir / f"current_position_analysis_{as_of}.csv"
    md_path = output_dir / f"current_position_analysis_{as_of}.md"
    png_path = output_dir / f"current_position_analysis_{as_of}.png"
    result.to_csv(csv_path, index=False)
    render_position_analysis_png(result, png_path)
    _write_markdown(md_path, result, positions_path, total_value)
    return {"csv": csv_path, "markdown": md_path, "png": png_path}


def _note_for_position(ticker: str, model_universe: set[str], reference_tickers: set[str]) -> str:
    if ticker in model_universe:
        return "E032 target available"
    if ticker in reference_tickers:
        return "KRX stock; trend analysis only"
    if str(ticker).isdigit():
        return "KRX ETF/unmapped; model target unavailable"
    return "US/global asset; model target unavailable"


def _write_markdown(path: Path, frame: pd.DataFrame, positions_path: str, total_value: float) -> None:
    lines = [
        "# Current Position Analysis",
        "",
        f"- positions_source: `{positions_path}`",
        f"- portfolio_value: `{total_value:,.0f}`",
        f"- holdings: `{len(frame)}`",
        f"- model_universe_holdings: `{int((frame['asset_scope'] == 'model_universe').sum())}`",
        f"- krx_stock_outside_model: `{int((frame['asset_scope'] == 'krx_stock_outside_model').sum())}`",
        f"- out_of_model_holdings: `{int((frame['asset_scope'] != 'model_universe').sum())}`",
        f"- png: `{path.with_suffix('.png')}`",
        "",
        "## Holdings",
        "",
        _markdown_table(
            frame[
                [
                    "ticker",
                    "name",
                    "asset_scope",
                    "market_value",
                    "current_weight_pct",
                    "pnl_pct",
                    "model_target_ratio_pct",
                    "trend_status",
                    "return_20d_pct",
                    "drawdown_60d_pct",
                    "note",
                ]
            ]
        ),
        "",
        "## Korean Stocks",
        "",
        _markdown_table(
            frame[frame["asset_scope"].isin(["model_universe", "krx_stock_outside_model"])][
                [
                    "ticker",
                    "name",
                    "reference_name",
                    "market",
                    "pnl_pct",
                    "model_target_ratio_pct",
                    "trend_status",
                    "return_20d_pct",
                    "return_60d_pct",
                    "drawdown_60d_pct",
                    "feature_date",
                ]
            ]
        ),
        "",
        "## US / Global Assets",
        "",
        _markdown_table(
            frame[frame["asset_scope"] == "us_or_global"][
                [
                    "ticker",
                    "name",
                    "current_weight_pct",
                    "pnl_pct",
                    "trend_status",
                    "return_20d_pct",
                    "return_60d_pct",
                    "return_120d_pct",
                    "drawdown_60d_pct",
                    "volatility_20d_pct",
                    "feature_date",
                ]
            ]
        ),
        "",
        "## Note",
        "",
        "Only tickers in the 48-stock model universe receive E032 target ratios. KRX stocks outside the model universe and US/global assets use trend analysis only. KRX ETFs are currently marked as unavailable unless a separate ETF price source is added.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--positions", default="data_krx/raw/positions/current_positions.csv")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--no-collect-krx", action="store_true")
    parser.add_argument("--no-collect-us", action="store_true")
    parser.add_argument("--krx-start", default="2025-01-01")
    parser.add_argument("--us-start", default="2025-01-01")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in analyze_positions(
        args.config,
        args.positions,
        args.rule,
        collect_krx=not args.no_collect_krx,
        collect_us=not args.no_collect_us,
        krx_start=args.krx_start,
        us_start=args.us_start,
        out_dir=args.out_dir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
