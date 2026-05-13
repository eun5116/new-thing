from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.evaluate import Performance, performance_from_returns
from stock_rl.trading_env import normalize_ticker


@dataclass(frozen=True)
class RuleSpec:
    name: str
    defensive_drawdown: float
    defensive_underperformance: float
    market_drop: float
    use_event_defense: bool
    prefer_ma_in_weak_trend: bool
    attack_choice: str = "e028"


RULES = [
    RuleSpec("gated_balanced_v1", -0.20, -0.05, -0.02, True, True),
    RuleSpec("gated_defensive_v1", -0.15, -0.03, -0.01, True, True),
    RuleSpec("gated_aggressive_v1", -0.25, -0.08, -0.03, True, False),
    RuleSpec("gated_no_event_v1", -0.20, -0.05, -0.02, False, True),
    RuleSpec("gated_bh_balanced_v1", -0.20, -0.05, -0.02, True, True, "buy_hold"),
    RuleSpec("gated_bh_aggressive_v1", -0.25, -0.08, -0.03, True, False, "buy_hold"),
    RuleSpec("gated_bh_no_event_v1", -0.20, -0.05, -0.02, False, True, "buy_hold"),
]


def _resolve_trace_path(config_path: str, split: str, model_name: str) -> Path:
    return project_path(config_path, "reports", f"{model_name}_{split}_action_trace.csv")


def _load_candidate_frame(config_path: str, split: str, e027_model: str, e028_model: str) -> pd.DataFrame:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    features = pd.read_parquet(features_path).copy()
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["date"] = pd.to_datetime(features["date"])

    key = ["date", "ticker"]
    e027 = pd.read_csv(_resolve_trace_path(config_path, split, e027_model), parse_dates=["date"])
    e028 = pd.read_csv(_resolve_trace_path(config_path, split, e028_model), parse_dates=["date"])
    e027["ticker"] = e027["ticker"].astype(str).map(normalize_ticker)
    e028["ticker"] = e028["ticker"].astype(str).map(normalize_ticker)

    frame = features.merge(e027[key + ["target_ratio", "daily_return"]], on=key, how="inner", suffixes=("", "_e027"))
    frame = frame.rename(columns={"target_ratio": "e027_target_ratio", "daily_return": "e027_daily_return"})
    frame = frame.merge(e028[key + ["target_ratio", "daily_return"]], on=key, how="inner", suffixes=("", "_e028"))
    frame = frame.rename(columns={"target_ratio": "e028_target_ratio", "daily_return": "e028_daily_return"})
    frame["ma20_60_target_ratio"] = frame["ma20_60_position"].astype(float).clip(0.0, 1.0)
    frame["buy_hold_target_ratio"] = 1.0
    frame["ma20_60_daily_return"] = frame["ma20_60_target_ratio"] * frame["target_return_1d"]
    frame["buy_hold_daily_return"] = frame["target_return_1d"]
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _choose_rule(row: pd.Series, rule: RuleSpec) -> str:
    defensive = (
        row["market_return_1d"] <= rule.market_drop
        or row["drawdown_60d"] <= rule.defensive_drawdown
        or row["drawdown_vs_market_60d"] <= rule.defensive_underperformance
        or (rule.use_event_defense and row.get("event_any", 0.0) > 0)
    )
    if defensive:
        return "e027"
    strong_trend = (
        row["ma20_60_position"] > 0
        and row["relative_strength_20d"] >= 0
        and row.get("market_trend_regime", 0.0) >= 0
    )
    if strong_trend:
        return rule.attack_choice
    if rule.prefer_ma_in_weak_trend:
        return "ma20_60"
    return rule.attack_choice


def _target_ratio_for_choice(row: pd.Series, choice: str) -> float:
    if choice == "e027":
        return float(row["e027_target_ratio"])
    if choice == "e028":
        return float(row["e028_target_ratio"])
    if choice == "ma20_60":
        return float(row["ma20_60_target_ratio"])
    if choice == "buy_hold":
        return 1.0
    raise ValueError(f"unknown choice: {choice}")


def _simulate_ratio_strategy(
    frame: pd.DataFrame,
    ratio_column: str,
    initial_cash: float,
    transaction_cost_pct: float,
) -> tuple[Performance, pd.DataFrame]:
    cash = initial_cash
    shares = 0.0
    portfolio_value = initial_cash
    rows = []
    for row in frame.sort_values("date").itertuples(index=False):
        price = float(row.adj_close)
        prev_value = portfolio_value
        current_stock_value = shares * price
        current_value = cash + current_stock_value
        target_ratio = float(getattr(row, ratio_column))
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
        next_price = price * (1.0 + float(row.target_return_1d))
        portfolio_value = cash + shares * next_price
        daily_return = portfolio_value / max(prev_value, 1e-12) - 1.0
        rows.append(
            {
                "date": row.date,
                "ticker": row.ticker,
                "target_ratio": target_ratio,
                "daily_return": daily_return,
                "portfolio_value": portfolio_value,
                "turnover": traded_value / max(prev_value, 1e-12),
            }
        )
    trace = pd.DataFrame(rows)
    return performance_from_returns(trace["daily_return"]), trace


def _simulate_gated_strategy(
    frame: pd.DataFrame,
    rule: RuleSpec,
    initial_cash: float,
    transaction_cost_pct: float,
) -> tuple[Performance, pd.DataFrame]:
    cash = initial_cash
    shares = 0.0
    portfolio_value = initial_cash
    rows = []
    for _, row in frame.sort_values("date").iterrows():
        choice = _choose_rule(row, rule)
        target_ratio = _target_ratio_for_choice(row, choice)
        price = float(row["adj_close"])
        prev_value = portfolio_value
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
        next_price = price * (1.0 + float(row["target_return_1d"]))
        portfolio_value = cash + shares * next_price
        daily_return = portfolio_value / max(prev_value, 1e-12) - 1.0
        rows.append(
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "choice": choice,
                "target_ratio": target_ratio,
                "daily_return": daily_return,
                "portfolio_value": portfolio_value,
                "turnover": traded_value / max(prev_value, 1e-12),
            }
        )
    trace = pd.DataFrame(rows)
    return performance_from_returns(trace["daily_return"]), trace


def _regime_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=frame.index),
        "event_any": frame["event_any"] > 0,
        "market_drop_2pct": frame["market_return_1d"] <= -0.02,
        "market_drop_1pct": frame["market_return_1d"] <= -0.01,
        "strong_trend": (
            (frame["ma20_60_position"] > 0)
            & (frame["relative_strength_20d"] >= 0)
            & (frame["market_trend_regime"] >= 0)
        ),
        "weak_relative_strength": frame["relative_strength_20d"] <= frame["relative_strength_20d"].quantile(0.2),
        "stock_drawdown_20pct": frame["drawdown_60d"] <= -0.20,
        "underperforming_market": frame["drawdown_vs_market_60d"] <= -0.05,
    }


def build_regime_comparison(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    strategies = {
        "buy_hold": "buy_hold_daily_return",
        "ma20_60": "ma20_60_daily_return",
        "e027": "e027_daily_return",
        "e028": "e028_daily_return",
    }
    rows = []
    for regime, mask in _regime_masks(frame).items():
        subset = frame[mask]
        for strategy, column in strategies.items():
            rows.append(
                {
                    "split": split,
                    "regime": regime,
                    "strategy": strategy,
                    "rows": len(subset),
                    "avg_daily_return": subset[column].mean() if len(subset) else np.nan,
                    "positive_day_rate": (subset[column] > 0).mean() if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def evaluate_split(
    config_path: str,
    split: str,
    e027_model: str,
    e028_model: str,
    out_dir: str | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    initial_cash = float(config["trading"].get("initial_cash", 1_000_000.0))
    transaction_cost_pct = float(config["trading"].get("transaction_cost_pct", 0.001))
    frame = _load_candidate_frame(config_path, split, e027_model, e028_model)
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    trace_rows = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=True):
        for strategy, ratio_column in {
            "buy_hold_costed": "buy_hold_target_ratio",
            "ma20_60_costed": "ma20_60_target_ratio",
            "e027_ratio_replay": "e027_target_ratio",
            "e028_ratio_replay": "e028_target_ratio",
        }.items():
            metrics, trace = _simulate_ratio_strategy(ticker_frame, ratio_column, initial_cash, transaction_cost_pct)
            metric_rows.append({"split": split, "ticker": ticker, "strategy": strategy, **metrics.to_dict()})
            trace["strategy"] = strategy
            trace_rows.append(trace)
        for rule in RULES:
            metrics, trace = _simulate_gated_strategy(ticker_frame, rule, initial_cash, transaction_cost_pct)
            metric_rows.append({"split": split, "ticker": ticker, "strategy": rule.name, **metrics.to_dict()})
            trace["strategy"] = rule.name
            trace_rows.append(trace)

    metrics_df = pd.DataFrame(metric_rows)
    traces_df = pd.concat(trace_rows, ignore_index=True)
    summary_df = (
        metrics_df.groupby(["split", "strategy"], as_index=False)
        .agg(
            tickers=("ticker", "count"),
            avg_cumulative_return=("cumulative_return", "mean"),
            avg_annualized_return=("annualized_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
        )
        .sort_values(["split", "avg_cumulative_return"], ascending=[True, False])
    )
    regime_df = build_regime_comparison(frame, split)

    metric_path = output_dir / f"gated_policy_{split}_metrics.csv"
    trace_path = output_dir / f"gated_policy_{split}_trace.csv"
    summary_path = output_dir / f"gated_policy_{split}_summary.csv"
    regime_path = output_dir / f"regime_candidate_comparison_{split}.csv"
    metrics_df.to_csv(metric_path, index=False)
    traces_df.to_csv(trace_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    regime_df.to_csv(regime_path, index=False)
    return {"metrics": metric_path, "trace": trace_path, "summary": summary_path, "regime": regime_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--e027-model", default="ppo_KRX_E027_liquid48_target_hybrid")
    parser.add_argument("--e028-model", default="ppo_KRX_E028_liquid48_target_hybrid_aggressive")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in evaluate_split(args.config, args.split, args.e027_model, args.e028_model, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
