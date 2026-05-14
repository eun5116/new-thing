from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.build_features import FEATURE_COLUMNS
from stock_rl.config import load_config, project_path
from stock_rl.evaluate import performance_from_returns
from stock_rl.evaluate_regime_exposure_cap import RULES, _cap_for_row, _resolve_model_path, _target_ratio_from_action
from stock_rl.trading_env import TradingEnvConfig, normalize_ticker


def _feature_columns(features: pd.DataFrame) -> list[str]:
    event_columns = sorted(column for column in features.columns if column.startswith("event_"))
    return FEATURE_COLUMNS + event_columns


def _rebalance_dates(dates: pd.Series, frequency: str) -> set[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).sort_values().unique())
    if frequency == "daily":
        return set(unique)
    if frequency == "weekly":
        return set(unique.groupby(unique.dt.to_period("W-FRI")).tail(1))
    if frequency == "monthly":
        return set(unique.groupby(unique.dt.to_period("M")).tail(1))
    raise ValueError(f"unsupported rebalance frequency: {frequency}")


def _cap_and_normalize(scores: pd.Series, gross_cap: float, max_weight: float) -> pd.Series:
    scores = scores.astype(float).clip(lower=0.0)
    scores = scores[scores > 0]
    if scores.empty:
        return scores
    weights = pd.Series(0.0, index=scores.index)
    remaining = scores.copy()
    remaining_gross = gross_cap
    while not remaining.empty and remaining_gross > 1e-12:
        proposal = remaining / remaining.sum() * remaining_gross
        capped = proposal[proposal >= max_weight]
        if capped.empty:
            weights.loc[remaining.index] = proposal
            break
        weights.loc[capped.index] = max_weight
        remaining_gross -= max_weight * len(capped)
        remaining = remaining.drop(index=capped.index)
    return weights[weights > 0]


def _select_weights(day: pd.DataFrame, strategy: str, top_n: int, gross_cap: float, max_weight: float) -> pd.Series:
    data = day.copy()
    if strategy == "buy_hold_basket":
        scores = pd.Series(1.0, index=data["ticker"])
        return _cap_and_normalize(scores, gross_cap, max_weight)
    if strategy == "ma20_60_basket":
        data = data[data["ma20_60_position"].astype(float) > 0].sort_values(
            ["return_20d", "return_60d", "ticker"], ascending=[False, False, True]
        )
        selected = data.head(top_n)
        scores = pd.Series(1.0, index=selected["ticker"])
        return _cap_and_normalize(scores, gross_cap, max_weight)
    if strategy == "e032_target_basket":
        data = data.sort_values(["e032_target_ratio", "return_20d", "ticker"], ascending=[False, False, True])
        selected = data.head(top_n)
        scores = pd.Series(selected["e032_target_ratio"].astype(float).to_numpy(), index=selected["ticker"])
        return _cap_and_normalize(scores, gross_cap, max_weight)
    raise ValueError(f"unsupported strategy: {strategy}")


def add_e032_targets(
    config_path: str,
    features: pd.DataFrame,
    rule_name: str = "strong_trend_full_else070",
    model_name: str | None = None,
) -> pd.DataFrame:
    matplotlib_cache_dir = Path("/tmp/stock_rl_matplotlib")
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for portfolio allocator backtest") from exc

    config = load_config(config_path)
    env_config = TradingEnvConfig(**config["trading"])
    resolved_model_name = model_name or config["training"]["model_name"]
    model = PPO.load(_resolve_model_path(config_path, resolved_model_name))
    feature_columns = _feature_columns(features)
    rule = _resolve_rule(rule_name)

    rows = []
    for _, row in features.iterrows():
        obs = np.asarray([float(row[col]) for col in feature_columns] + [1.0, 0.0], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_target_ratio = _target_ratio_from_action(int(action), env_config)
        cap, cap_reason = _cap_for_row(row, rule)
        rows.append((raw_target_ratio, min(raw_target_ratio, cap), cap, cap_reason))
    result = features.copy()
    result[["e032_raw_target_ratio", "e032_target_ratio", "e032_cap", "e032_cap_reason"]] = pd.DataFrame(
        rows,
        index=result.index,
    )
    return result


def _resolve_rule(name: str):
    for rule in RULES:
        if rule.name == name:
            return rule
    available = ", ".join(rule.name for rule in RULES)
    raise ValueError(f"unknown rule: {name}. Available rules: {available}")


def simulate_portfolio(
    features: pd.DataFrame,
    strategy: str,
    top_n: int = 12,
    gross_cap: float = 0.90,
    max_weight: float = 0.20,
    transaction_cost_pct: float = 0.0015,
    rebalance_frequency: str = "weekly",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    data = features.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ticker"] = data["ticker"].astype(str).map(normalize_ticker)
    dates = sorted(data["date"].unique())
    rebalance_days = _rebalance_dates(data["date"], rebalance_frequency)
    weights = pd.Series(dtype=float)
    returns = []
    trace_rows = []
    allocation_rows = []

    for date in dates:
        day = data[data["date"] == date].copy()
        day_returns = pd.Series(day["target_return_1d"].astype(float).to_numpy(), index=day["ticker"])
        weights = weights.reindex(day_returns.index).fillna(0.0)
        turnover = 0.0
        if date in rebalance_days:
            target_weights = _select_weights(day, strategy, top_n, gross_cap, max_weight).reindex(day_returns.index).fillna(0.0)
            turnover = float((target_weights - weights).abs().sum())
            weights = target_weights
            for ticker, weight in weights[weights > 0].items():
                row = day[day["ticker"] == ticker].iloc[0]
                allocation_rows.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "ticker": ticker,
                        "target_weight": weight,
                        "return_20d": row.get("return_20d", 0.0),
                        "return_60d": row.get("return_60d", 0.0),
                        "e032_target_ratio": row.get("e032_target_ratio", np.nan),
                    }
                )
        cost = turnover * transaction_cost_pct
        gross_return = float((weights * day_returns).sum())
        daily_return = gross_return - cost
        returns.append(daily_return)
        trace_rows.append(
            {
                "date": date,
                "strategy": strategy,
                "daily_return": daily_return,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(weights.sum()),
                "holdings": int((weights > 0).sum()),
            }
        )
        denominator = 1.0 + daily_return
        if denominator > 1e-12:
            weights = weights * (1.0 + day_returns) / denominator
    return pd.Series(returns, index=dates), pd.DataFrame(trace_rows), pd.DataFrame(allocation_rows)


def backtest_split(
    config_path: str,
    split: str = "test",
    rule: str = "strong_trend_full_else070",
    top_n: int = 12,
    gross_cap: float = 0.90,
    max_weight: float = 0.20,
    transaction_cost_pct: float = 0.0015,
    rebalance_frequency: str = "weekly",
    out_dir: str | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    features_path = project_path(config_path, config["project"]["data_dir"], "processed", f"{split}.parquet")
    features = pd.read_parquet(features_path)
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["date"] = pd.to_datetime(features["date"])
    features = add_e032_targets(config_path, features, rule_name=rule)

    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    traces = []
    allocations = []
    for strategy in ["buy_hold_basket", "ma20_60_basket", "e032_target_basket"]:
        returns, trace, allocation = simulate_portfolio(
            features,
            strategy,
            top_n=top_n,
            gross_cap=gross_cap,
            max_weight=max_weight,
            transaction_cost_pct=transaction_cost_pct,
            rebalance_frequency=rebalance_frequency,
        )
        metrics = performance_from_returns(returns)
        metric_rows.append(
            {
                "split": split,
                "strategy": strategy,
                **metrics.to_dict(),
                "top_n": top_n,
                "gross_cap": gross_cap,
                "max_weight": max_weight,
                "transaction_cost_pct": transaction_cost_pct,
                "rebalance_frequency": rebalance_frequency,
                "avg_turnover": float(trace["turnover"].mean()),
                "total_cost": float(trace["cost"].sum()),
                "avg_holdings": float(trace["holdings"].mean()),
                "avg_gross_exposure": float(trace["gross_exposure"].mean()),
            }
        )
        traces.append(trace)
        allocations.append(allocation)

    metrics_df = pd.DataFrame(metric_rows).sort_values("cumulative_return", ascending=False)
    trace_df = pd.concat(traces, ignore_index=True)
    allocation_df = pd.concat(allocations, ignore_index=True)
    metrics_path = output_dir / f"portfolio_allocator_{split}_metrics.csv"
    trace_path = output_dir / f"portfolio_allocator_{split}_trace.csv"
    allocation_path = output_dir / f"portfolio_allocator_{split}_allocations.csv"
    report_path = output_dir / f"portfolio_allocator_{split}_report.md"
    metrics_df.to_csv(metrics_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    allocation_df.to_csv(allocation_path, index=False)
    _write_report(report_path, metrics_df, split, top_n, gross_cap, max_weight, transaction_cost_pct, rebalance_frequency)
    return {"metrics": metrics_path, "trace": trace_path, "allocations": allocation_path, "report": report_path}


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    split: str,
    top_n: int,
    gross_cap: float,
    max_weight: float,
    transaction_cost_pct: float,
    rebalance_frequency: str,
) -> None:
    display = metrics.copy()
    for column in ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "avg_turnover", "total_cost", "avg_gross_exposure"]:
        display[column] = (display[column].astype(float) * 100.0).round(2)
    display["sharpe"] = display["sharpe"].round(2)
    display["avg_holdings"] = display["avg_holdings"].round(1)
    keep = [
        "strategy",
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "total_cost",
        "avg_holdings",
        "avg_gross_exposure",
    ]
    labels = {
        "cumulative_return": "return_pct",
        "annualized_return": "annualized_return_pct",
        "max_drawdown": "mdd_pct",
        "avg_turnover": "avg_turnover_pct",
        "total_cost": "total_cost_pct",
        "avg_gross_exposure": "avg_exposure_pct",
    }
    table = display[keep].rename(columns=labels)
    lines = [
        f"# Portfolio Allocator Backtest - {split}",
        "",
        f"- top_n: `{top_n}`",
        f"- max_weight: `{max_weight * 100.0:.1f}%`",
        f"- gross_cap: `{gross_cap * 100.0:.1f}%`",
        f"- transaction_cost: `{transaction_cost_pct * 100.0:.2f}%` one-way",
        f"- rebalance: `{rebalance_frequency}`",
        "- note: `buy_hold_basket` is equal-weight across the full available universe; `top_n` applies to MA20/60 and E032 baskets.",
        "",
        _markdown_table(table),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--gross-cap", type=float, default=0.90)
    parser.add_argument("--max-weight", type=float, default=0.20)
    parser.add_argument("--transaction-cost-pct", type=float, default=0.0015)
    parser.add_argument("--rebalance-frequency", default="weekly", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for split in args.splits:
        paths = backtest_split(
            args.config,
            split,
            args.rule,
            args.top_n,
            args.gross_cap,
            args.max_weight,
            args.transaction_cost_pct,
            args.rebalance_frequency,
            args.out_dir,
        )
        for name, path in paths.items():
            print(f"{split} {name}: {path}")


if __name__ == "__main__":
    main()
