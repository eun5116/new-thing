from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.config import project_path
from stock_rl.trading_env import normalize_ticker


DEFAULT_STRATEGIES = [
    "uncapped",
    "strong_trend_full_else070",
    "strong_trend_full_else075",
    "strong_trend_full_else080",
    "strong_trend_full_else085",
]


def _compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns.astype(float)).prod() - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _load_trace(config_path: str, split: str, strategies: list[str]) -> pd.DataFrame:
    path = project_path(config_path, "reports", f"regime_exposure_cap_{split}_trace.csv")
    trace = pd.read_csv(path, parse_dates=["date"])
    trace["ticker"] = trace["ticker"].astype(str).map(normalize_ticker)
    trace["month"] = trace["date"].dt.to_period("M").astype(str)
    return trace[trace["strategy"].isin(strategies)].copy()


def _load_metrics(config_path: str, split: str, strategies: list[str]) -> pd.DataFrame:
    path = project_path(config_path, "reports", f"regime_exposure_cap_{split}_metrics.csv")
    metrics = pd.read_csv(path)
    metrics["ticker"] = metrics["ticker"].astype(str).map(normalize_ticker)
    return metrics[metrics["strategy"].isin(strategies)].copy()


def build_monthly_summary(trace: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for (month, strategy), group in trace.groupby(["month", "strategy"], sort=True):
        ticker_returns = group.groupby("ticker")["daily_return"].apply(_compound_return)
        ticker_mdds = group.groupby("ticker")["daily_return"].apply(_max_drawdown)
        rows.append(
            {
                "split": split,
                "month": month,
                "strategy": strategy,
                "tickers": int(ticker_returns.size),
                "avg_monthly_return": float(ticker_returns.mean()),
                "median_monthly_return": float(ticker_returns.median()),
                "win_rate": float((ticker_returns > 0).mean()),
                "avg_monthly_mdd": float(ticker_mdds.mean()),
                "avg_target_ratio": float(group["target_ratio"].mean()),
                "avg_cap": float(group["cap"].mean()),
                "cap_day_rate": float((group["target_ratio"] < group["raw_target_ratio"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["month", "avg_monthly_return"], ascending=[True, False])


def build_monthly_comparison(monthly_summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    returns = monthly_summary.pivot(index=["split", "month"], columns="strategy", values="avg_monthly_return").reset_index()
    mdds = monthly_summary.pivot(index=["split", "month"], columns="strategy", values="avg_monthly_mdd").reset_index()
    ratios = monthly_summary.pivot(index=["split", "month"], columns="strategy", values="avg_target_ratio").reset_index()
    comparison = returns.merge(mdds, on=["split", "month"], suffixes=("", "_mdd"))
    comparison = comparison.merge(ratios, on=["split", "month"], suffixes=("", "_target_ratio"))
    comparison["candidate"] = candidate
    comparison["baseline"] = baseline
    comparison["candidate_minus_baseline_return"] = comparison[candidate] - comparison[baseline]
    comparison["candidate_minus_baseline_mdd"] = comparison[f"{candidate}_mdd"] - comparison[f"{baseline}_mdd"]
    comparison["candidate_minus_baseline_target_ratio"] = (
        comparison[f"{candidate}_target_ratio"] - comparison[f"{baseline}_target_ratio"]
    )
    return comparison.sort_values("candidate_minus_baseline_return")


def build_ticker_comparison(metrics: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    values = metrics.pivot(index=["split", "ticker"], columns="strategy")
    rows = []
    for split, ticker in values.index:
        row = values.loc[(split, ticker)]
        rows.append(
            {
                "split": split,
                "ticker": ticker,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_return": row[("cumulative_return", candidate)],
                "baseline_return": row[("cumulative_return", baseline)],
                "return_diff": row[("cumulative_return", candidate)] - row[("cumulative_return", baseline)],
                "candidate_sharpe": row[("sharpe", candidate)],
                "baseline_sharpe": row[("sharpe", baseline)],
                "sharpe_diff": row[("sharpe", candidate)] - row[("sharpe", baseline)],
                "candidate_mdd": row[("max_drawdown", candidate)],
                "baseline_mdd": row[("max_drawdown", baseline)],
                "mdd_diff": row[("max_drawdown", candidate)] - row[("max_drawdown", baseline)],
                "candidate_target_ratio": row[("avg_target_ratio", candidate)],
                "baseline_target_ratio": row[("avg_target_ratio", baseline)],
                "target_ratio_diff": row[("avg_target_ratio", candidate)] - row[("avg_target_ratio", baseline)],
            }
        )
    return pd.DataFrame(rows).sort_values("return_diff")


def analyze(
    config_path: str,
    split: str,
    baseline: str = "uncapped",
    candidate: str = "strong_trend_full_else070",
    out_dir: str | None = None,
) -> dict[str, Path]:
    strategies = sorted(set(DEFAULT_STRATEGIES + [baseline, candidate]))
    trace = _load_trace(config_path, split, strategies)
    metrics = _load_metrics(config_path, split, strategies)
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_summary = build_monthly_summary(trace, split)
    monthly_comparison = build_monthly_comparison(monthly_summary, baseline, candidate)
    ticker_comparison = build_ticker_comparison(metrics, baseline, candidate)
    worst_months = monthly_comparison.head(12)
    worst_tickers = ticker_comparison.head(20)

    prefix = f"regime_exposure_cap_{split}_{candidate}_vs_{baseline}"
    paths = {
        "monthly_summary": output_dir / f"regime_exposure_cap_{split}_monthly_summary.csv",
        "monthly_comparison": output_dir / f"{prefix}_monthly_comparison.csv",
        "ticker_comparison": output_dir / f"{prefix}_ticker_comparison.csv",
        "worst_months": output_dir / f"{prefix}_worst_months.csv",
        "worst_tickers": output_dir / f"{prefix}_worst_tickers.csv",
    }
    monthly_summary.to_csv(paths["monthly_summary"], index=False)
    monthly_comparison.to_csv(paths["monthly_comparison"], index=False)
    ticker_comparison.to_csv(paths["ticker_comparison"], index=False)
    worst_months.to_csv(paths["worst_months"], index=False)
    worst_tickers.to_csv(paths["worst_tickers"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--baseline", default="uncapped")
    parser.add_argument("--candidate", default="strong_trend_full_else070")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in analyze(args.config, args.split, args.baseline, args.candidate, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
