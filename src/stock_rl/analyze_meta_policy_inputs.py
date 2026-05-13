from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.trading_env import normalize_ticker


STRATEGIES = ["buy_hold_costed", "ma20_60_costed", "e028_ratio_replay"]
FEATURE_COLUMNS = [
    "market_return_5d",
    "market_ma20_gap",
    "market_volatility_20d",
    "market_drop_recent_5d",
    "market_drop_recent_20d",
    "market_trend_regime",
    "return_5d",
    "return_20d",
    "ma20_gap",
    "ma20_60_gap",
    "ma20_60_position",
    "relative_strength_20d",
    "relative_strength_regime",
    "drawdown_20d",
    "drawdown_60d",
    "drawdown_vs_market_60d",
    "volatility_20d",
    "volatility_60d",
    "trading_value_zscore_20d",
    "volume_zscore_20d",
    "event_recent_5d",
    "event_recent_20d",
    "event_any",
]


def _compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns.astype(float)).prod() - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _load_features(config_path: str, split: str) -> pd.DataFrame:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    features = pd.read_parquet(path)
    features["date"] = pd.to_datetime(features["date"])
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["month"] = features["date"].dt.to_period("M").astype(str)
    return features


def _load_trace(config_path: str, split: str) -> pd.DataFrame:
    path = project_path(config_path, "reports", f"gated_policy_{split}_trace.csv")
    trace = pd.read_csv(path, parse_dates=["date"])
    trace["ticker"] = trace["ticker"].astype(str).map(normalize_ticker)
    trace["month"] = trace["date"].dt.to_period("M").astype(str)
    return trace[trace["strategy"].isin(STRATEGIES)].copy()


def build_monthly_dataset(config_path: str, split: str) -> pd.DataFrame:
    trace = _load_trace(config_path, split)
    features = _load_features(config_path, split)

    monthly_rows = []
    for (ticker, month, strategy), group in trace.groupby(["ticker", "month", "strategy"], sort=True):
        monthly_rows.append(
            {
                "ticker": ticker,
                "month": month,
                "strategy": strategy,
                "monthly_return": _compound_return(group["daily_return"]),
                "monthly_mdd": _max_drawdown(group["daily_return"]),
                "avg_target_ratio": group["target_ratio"].mean(),
                "avg_turnover": group["turnover"].mean(),
                "days": len(group),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    returns = monthly.pivot(index=["ticker", "month"], columns="strategy", values="monthly_return").reset_index()
    mdds = monthly.pivot(index=["ticker", "month"], columns="strategy", values="monthly_mdd").reset_index()
    ratios = monthly.pivot(index=["ticker", "month"], columns="strategy", values="avg_target_ratio").reset_index()
    data = returns.merge(mdds, on=["ticker", "month"], suffixes=("", "_mdd"))
    data = data.merge(ratios, on=["ticker", "month"], suffixes=("", "_avg_target_ratio"))

    feature_start = (
        features.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .first()[["ticker", "month", "date"] + FEATURE_COLUMNS]
    )
    feature_start = feature_start.rename(
        columns={column: f"start_{column}" for column in FEATURE_COLUMNS} | {"date": "month_start_date"}
    )
    data = data.merge(feature_start, on=["ticker", "month"], how="left")

    return_columns = STRATEGIES
    data["best_strategy"] = data[return_columns].idxmax(axis=1)
    data["best_return"] = data[return_columns].max(axis=1)
    data["worst_strategy"] = data[return_columns].idxmin(axis=1)
    data["worst_return"] = data[return_columns].min(axis=1)
    data["e028_gap_to_best"] = data["e028_ratio_replay"] - data["best_return"]
    data["e028_rank"] = data[return_columns].rank(axis=1, ascending=False, method="min")["e028_ratio_replay"]
    data["e028_failed"] = data["e028_rank"] > 1
    data["e028_large_failure"] = data["e028_gap_to_best"] <= -0.05
    return data.sort_values(["month", "ticker"]).reset_index(drop=True)


def build_winner_summary(monthly: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for strategy, group in monthly.groupby("best_strategy", sort=True):
        rows.append(
            {
                "split": split,
                "best_strategy": strategy,
                "months": len(group),
                "share": len(group) / len(monthly),
                "avg_best_return": group["best_return"].mean(),
                "avg_e028_gap_to_best": group["e028_gap_to_best"].mean(),
                "avg_start_market_ma20_gap": group["start_market_ma20_gap"].mean(),
                "avg_start_relative_strength_20d": group["start_relative_strength_20d"].mean(),
                "avg_start_drawdown_60d": group["start_drawdown_60d"].mean(),
                "avg_start_drawdown_vs_market_60d": group["start_drawdown_vs_market_60d"].mean(),
                "avg_start_market_volatility_20d": group["start_market_volatility_20d"].mean(),
                "avg_start_event_recent_20d": group["start_event_recent_20d"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def build_feature_signal(monthly: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        column = f"start_{feature}"
        e028_wins = monthly[monthly["best_strategy"] == "e028_ratio_replay"][column]
        e028_fails = monthly[monthly["e028_failed"]][column]
        large_fails = monthly[monthly["e028_large_failure"]][column]
        rows.append(
            {
                "split": split,
                "feature": column,
                "overall_mean": monthly[column].mean(),
                "e028_win_mean": e028_wins.mean(),
                "e028_fail_mean": e028_fails.mean(),
                "e028_large_fail_mean": large_fails.mean(),
                "fail_minus_win": e028_fails.mean() - e028_wins.mean(),
                "large_fail_minus_win": large_fails.mean() - e028_wins.mean(),
                "abs_large_fail_minus_win": abs(large_fails.mean() - e028_wins.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_large_fail_minus_win", ascending=False)


def build_month_summary(monthly: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for month, group in monthly.groupby("month", sort=True):
        rows.append(
            {
                "split": split,
                "month": month,
                "rows": len(group),
                "buy_hold_avg_return": group["buy_hold_costed"].mean(),
                "ma20_60_avg_return": group["ma20_60_costed"].mean(),
                "e028_avg_return": group["e028_ratio_replay"].mean(),
                "e028_win_share": (group["best_strategy"] == "e028_ratio_replay").mean(),
                "buy_hold_win_share": (group["best_strategy"] == "buy_hold_costed").mean(),
                "ma20_60_win_share": (group["best_strategy"] == "ma20_60_costed").mean(),
                "e028_large_failure_share": group["e028_large_failure"].mean(),
            }
        )
    return pd.DataFrame(rows)


def analyze(config_path: str, split: str, out_dir: str | None = None) -> dict[str, Path]:
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = build_monthly_dataset(config_path, split)
    winner_summary = build_winner_summary(monthly, split)
    feature_signal = build_feature_signal(monthly, split)
    month_summary = build_month_summary(monthly, split)
    failures = monthly.sort_values("e028_gap_to_best").head(80)

    paths = {
        "monthly": output_dir / f"meta_policy_monthly_strategy_returns_{split}.csv",
        "winner_summary": output_dir / f"meta_policy_monthly_winner_summary_{split}.csv",
        "feature_signal": output_dir / f"meta_policy_feature_signal_{split}.csv",
        "month_summary": output_dir / f"meta_policy_month_summary_{split}.csv",
        "failures": output_dir / f"meta_policy_e028_failure_cases_{split}.csv",
    }
    monthly.to_csv(paths["monthly"], index=False)
    winner_summary.to_csv(paths["winner_summary"], index=False)
    feature_signal.to_csv(paths["feature_signal"], index=False)
    month_summary.to_csv(paths["month_summary"], index=False)
    failures.to_csv(paths["failures"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in analyze(args.config, args.split, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
