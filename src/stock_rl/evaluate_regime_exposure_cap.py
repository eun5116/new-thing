from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.build_features import FEATURE_COLUMNS
from stock_rl.config import load_config, project_path
from stock_rl.evaluate import performance_from_returns
from stock_rl.trading_env import TradingEnvConfig, normalize_ticker


@dataclass(frozen=True)
class CapRule:
    name: str
    default_cap: float = 1.0
    weak_market_cap: float | None = None
    recent_drop_cap: float | None = None
    drawdown_cap: float | None = None
    full_only_in_strong_trend: bool = False


RULES = [
    CapRule("uncapped"),
    CapRule("cap_weak_market_085", weak_market_cap=0.85),
    CapRule("cap_recent_drop_085", recent_drop_cap=0.85),
    CapRule("cap_weak_or_drop_085", weak_market_cap=0.85, recent_drop_cap=0.85),
    CapRule("cap_weak70_drop85", weak_market_cap=0.70, recent_drop_cap=0.85),
    CapRule("strong_trend_full_else085", default_cap=0.85, full_only_in_strong_trend=True),
    CapRule("strong_trend_full_else070", default_cap=0.70, full_only_in_strong_trend=True),
]


def _resolve_model_path(config_path: str, model_name: str) -> Path:
    path = project_path(config_path, "models", model_name)
    if path.exists():
        return path
    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"model not found: {path}")


def _feature_columns(features: pd.DataFrame) -> list[str]:
    event_columns = sorted(column for column in features.columns if column.startswith("event_"))
    return FEATURE_COLUMNS + event_columns


def _target_ratio_from_action(action: int, env_config: TradingEnvConfig) -> float:
    if env_config.action_mode != "target_position":
        raise ValueError("regime exposure cap evaluator currently supports target_position action mode only")
    action_ratio = action / (env_config.target_position_bins - 1)
    min_ratio = env_config.min_target_position_ratio
    max_ratio = env_config.max_position_ratio
    return float(min_ratio + (max_ratio - min_ratio) * action_ratio)


def _is_weak_market(row: pd.Series) -> bool:
    return bool(
        row.get("market_return_120d", 0.0) < 0
        or row.get("market_return_60d", 0.0) < 0
        or row.get("market_ma120_gap", 0.0) < 0
        or row.get("market_ma60_gap", 0.0) < 0
    )


def _is_recent_drop(row: pd.Series) -> bool:
    return bool(row.get("market_drop_recent_20d", 0.0) > 0 or row.get("market_return_1d", 0.0) <= -0.02)


def _is_stock_drawdown(row: pd.Series) -> bool:
    return bool(row.get("drawdown_60d", 0.0) <= -0.20 or row.get("drawdown_vs_market_60d", 0.0) <= -0.05)


def _is_strong_trend(row: pd.Series) -> bool:
    return bool(
        row.get("market_return_120d", 0.0) > 0
        and row.get("market_ma120_gap", 0.0) > 0
        and row.get("market_return_60d", 0.0) > 0
        and row.get("ma20_60_position", 0.0) > 0
        and row.get("relative_strength_20d", 0.0) >= 0
    )


def _cap_for_row(row: pd.Series, rule: CapRule) -> tuple[float, str]:
    cap = rule.default_cap
    reasons = []
    if rule.full_only_in_strong_trend and _is_strong_trend(row):
        cap = 1.0
        reasons.append("strong_trend")
    if rule.weak_market_cap is not None and _is_weak_market(row):
        cap = min(cap, rule.weak_market_cap)
        reasons.append("weak_market")
    if rule.recent_drop_cap is not None and _is_recent_drop(row):
        cap = min(cap, rule.recent_drop_cap)
        reasons.append("recent_drop")
    if rule.drawdown_cap is not None and _is_stock_drawdown(row):
        cap = min(cap, rule.drawdown_cap)
        reasons.append("stock_drawdown")
    return float(cap), "|".join(reasons) if reasons else "none"


def simulate_model_with_cap(
    model,
    ticker_frame: pd.DataFrame,
    feature_columns: list[str],
    env_config: TradingEnvConfig,
    rule: CapRule,
    initial_cash: float,
    transaction_cost_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    data = ticker_frame.sort_values("date").reset_index(drop=True)
    cash = initial_cash
    shares = 0.0
    portfolio_value = initial_cash
    rows = []
    returns = []
    for _, row in data.iterrows():
        price = float(row["adj_close"])
        prev_value = portfolio_value
        current_stock_value = shares * price
        current_value = cash + current_stock_value
        cash_ratio = cash / max(current_value, 1e-12)
        position_ratio = current_stock_value / max(current_value, 1e-12)
        obs = np.asarray([float(row[col]) for col in feature_columns] + [cash_ratio, position_ratio], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_target_ratio = _target_ratio_from_action(int(action), env_config)
        cap, cap_reason = _cap_for_row(row, rule)
        target_ratio = min(raw_target_ratio, cap)
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
        returns.append(daily_return)
        rows.append(
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "strategy": rule.name,
                "action": int(action),
                "raw_target_ratio": raw_target_ratio,
                "cap": cap,
                "cap_reason": cap_reason,
                "target_ratio": target_ratio,
                "daily_return": daily_return,
                "portfolio_value": portfolio_value,
                "turnover": traded_value / max(prev_value, 1e-12),
            }
        )
    return pd.Series(returns), pd.DataFrame(rows)


def evaluate_split(config_path: str, split: str, model_name: str | None = None, out_dir: str | None = None) -> dict[str, Path]:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for regime exposure cap evaluation") from exc

    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    features = pd.read_parquet(features_path)
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["date"] = pd.to_datetime(features["date"])
    feature_columns = _feature_columns(features)
    env_config = TradingEnvConfig(**config["trading"])
    resolved_model_name = model_name or config["training"]["model_name"]
    model = PPO.load(_resolve_model_path(config_path, resolved_model_name))
    initial_cash = float(config["trading"].get("initial_cash", 1_000_000.0))
    transaction_cost_pct = float(config["trading"].get("transaction_cost_pct", 0.001))
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    traces = []
    for ticker, ticker_frame in features.groupby("ticker", sort=True):
        for rule in RULES:
            returns, trace = simulate_model_with_cap(
                model,
                ticker_frame,
                feature_columns,
                env_config,
                rule,
                initial_cash,
                transaction_cost_pct,
            )
            metrics = performance_from_returns(returns)
            action_counts = trace["action"].value_counts().sort_index().to_dict()
            metric_rows.append(
                {
                    "split": split,
                    "ticker": ticker,
                    "strategy": rule.name,
                    **metrics.to_dict(),
                    "avg_target_ratio": trace["target_ratio"].mean(),
                    "avg_raw_target_ratio": trace["raw_target_ratio"].mean(),
                    "avg_cap": trace["cap"].mean(),
                    "cap_day_rate": (trace["target_ratio"] < trace["raw_target_ratio"]).mean(),
                    "action_counts": json.dumps({str(k): int(v) for k, v in action_counts.items()}),
                }
            )
            traces.append(trace)

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = (
        metrics_df.groupby(["split", "strategy"], as_index=False)
        .agg(
            tickers=("ticker", "count"),
            avg_cumulative_return=("cumulative_return", "mean"),
            avg_annualized_return=("annualized_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_target_ratio=("avg_target_ratio", "mean"),
            avg_raw_target_ratio=("avg_raw_target_ratio", "mean"),
            avg_cap=("avg_cap", "mean"),
            avg_cap_day_rate=("cap_day_rate", "mean"),
        )
        .sort_values(["split", "avg_cumulative_return"], ascending=[True, False])
    )
    trace_df = pd.concat(traces, ignore_index=True)

    prefix = f"regime_exposure_cap_{split}"
    metrics_path = output_dir / f"{prefix}_metrics.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    trace_path = output_dir / f"{prefix}_trace.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    return {"metrics": metrics_path, "summary": summary_path, "trace": trace_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in evaluate_split(args.config, args.split, args.model_name, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
