from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.evaluate import performance_from_returns
from stock_rl.trading_env import normalize_ticker


@dataclass(frozen=True)
class StopReentryRule:
    name: str
    stop_loss_pct: float
    reentry_mode: str
    cooldown_days: int = 0


RULES = [
    StopReentryRule("stop15_reenter_ma_rs", 0.15, "ma_rs", 0),
    StopReentryRule("stop15_reenter_ma_rs_cool5", 0.15, "ma_rs", 5),
    StopReentryRule("stop15_reenter_ma_market_rs", 0.15, "ma_market_rs", 0),
    StopReentryRule("stop15_reenter_ma_market_rs_cool5", 0.15, "ma_market_rs", 5),
    StopReentryRule("stop20_reenter_ma_rs", 0.20, "ma_rs", 0),
    StopReentryRule("stop20_reenter_ma_market_rs", 0.20, "ma_market_rs", 0),
]


def _should_reenter(row: pd.Series, rule: StopReentryRule) -> bool:
    if rule.reentry_mode == "ma_rs":
        return bool(row["ma20_60_position"] > 0 and row["relative_strength_20d"] >= 0)
    if rule.reentry_mode == "ma_market_rs":
        return bool(
            row["ma20_60_position"] > 0
            and row["relative_strength_20d"] >= 0
            and row.get("market_trend_regime", 0.0) >= 0
            and row.get("market_drop_recent_20d", 0.0) <= 0
        )
    raise ValueError(f"unknown reentry mode: {rule.reentry_mode}")


def simulate_stop_reentry(
    frame: pd.DataFrame,
    rule: StopReentryRule,
    initial_cash: float,
    transaction_cost_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    data = frame.sort_values("date").reset_index(drop=True)
    in_position = True
    entry_price = float(data.iloc[0]["adj_close"])
    high_water_price = entry_price
    cooldown = 0
    cash = initial_cash
    shares = 0.0
    portfolio_value = initial_cash
    rows = []
    returns = []
    for _, row in data.iterrows():
        price = float(row["adj_close"])
        next_return = float(row["target_return_1d"])
        prev_value = portfolio_value
        if in_position:
            high_water_price = max(high_water_price, price)
            drawdown_from_high = price / max(high_water_price, 1e-12) - 1.0
            drawdown_from_entry = price / max(entry_price, 1e-12) - 1.0
            if drawdown_from_high <= -rule.stop_loss_pct or drawdown_from_entry <= -rule.stop_loss_pct:
                in_position = False
                cooldown = rule.cooldown_days
        else:
            if cooldown > 0:
                cooldown -= 1
            elif _should_reenter(row, rule):
                in_position = True
                entry_price = price
                high_water_price = price

        target_ratio = 1.0 if in_position else 0.0
        current_stock_value = shares * price
        current_value = cash + current_stock_value
        target_stock_value = current_value * target_ratio
        delta = target_stock_value - current_stock_value
        traded_value = abs(delta)
        if delta > 0:
            spend = min(delta * (1.0 + transaction_cost_pct), cash)
            buy_value = spend / (1.0 + transaction_cost_pct)
            shares += buy_value / price
            cash -= spend
        elif delta < 0:
            sell_value = min(-delta, current_stock_value)
            shares -= sell_value / price
            cash += sell_value * (1.0 - transaction_cost_pct)

        next_price = price * (1.0 + next_return)
        portfolio_value = cash + shares * next_price
        daily_return = portfolio_value / max(prev_value, 1e-12) - 1.0
        returns.append(daily_return)
        rows.append(
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "strategy": rule.name,
                "target_ratio": target_ratio,
                "daily_return": daily_return,
                "portfolio_value": portfolio_value,
                "turnover": traded_value / max(prev_value, 1e-12),
                "in_position": in_position,
                "cooldown": cooldown,
            }
        )
    return pd.Series(returns), pd.DataFrame(rows)


def evaluate_split(config_path: str, split: str, out_dir: str | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    initial_cash = float(config["trading"].get("initial_cash", 1_000_000.0))
    transaction_cost_pct = float(config["trading"].get("transaction_cost_pct", 0.001))
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    features = pd.read_parquet(features_path)
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["date"] = pd.to_datetime(features["date"])
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    trace_rows = []
    for ticker, ticker_frame in features.groupby("ticker", sort=True):
        for rule in RULES:
            returns, trace = simulate_stop_reentry(ticker_frame, rule, initial_cash, transaction_cost_pct)
            metrics = performance_from_returns(returns)
            metric_rows.append(
                {
                    "split": split,
                    "ticker": ticker,
                    "strategy": rule.name,
                    **metrics.to_dict(),
                    "avg_exposure": trace["target_ratio"].mean(),
                    "days_in_position": int(trace["target_ratio"].sum()),
                    "days_total": len(trace),
                }
            )
            trace_rows.append(trace)

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = (
        metrics_df.groupby(["split", "strategy"], as_index=False)
        .agg(
            tickers=("ticker", "count"),
            avg_cumulative_return=("cumulative_return", "mean"),
            avg_annualized_return=("annualized_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_exposure=("avg_exposure", "mean"),
        )
        .sort_values(["split", "avg_cumulative_return"], ascending=[True, False])
    )
    traces_df = pd.concat(trace_rows, ignore_index=True)

    metric_path = output_dir / f"stop_reentry_{split}_metrics.csv"
    summary_path = output_dir / f"stop_reentry_{split}_summary.csv"
    trace_path = output_dir / f"stop_reentry_{split}_trace.csv"
    metrics_df.to_csv(metric_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    traces_df.to_csv(trace_path, index=False)
    return {"metrics": metric_path, "summary": summary_path, "trace": trace_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in evaluate_split(args.config, args.split, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
