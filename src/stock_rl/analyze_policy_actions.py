from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.trading_env import StockTradingEnv, TradingEnvConfig


def load_ppo_model(model_path: str):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for policy action analysis. "
            "Install CPU dependencies with: pip install -r requirements-rl-cpu.txt"
        ) from exc
    return PPO.load(model_path)


def trace_policy_actions(
    model_path: str,
    features: pd.DataFrame,
    ticker: str,
    env_config: TradingEnvConfig,
) -> pd.DataFrame:
    env = StockTradingEnv(features, ticker=ticker, config=env_config)
    model = load_ppo_model(model_path)
    obs, _ = env.reset()
    done = False
    rows: list[dict[str, object]] = []

    while not done:
        step_index = env._step_index
        row = env.data.iloc[step_index]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        event_columns = [column for column in env.data.columns if column.startswith("event_")]
        rows.append(
            {
                "date": row["date"],
                "ticker": ticker,
                "action": int(action),
                "overlay": info.get("overlay", 0.0),
                "target_ratio": info.get("target_ratio", 0.0),
                "ma20_60_position": row.get("ma20_60_position", 0.0),
                "market_return_1d": row.get("market_return_1d", 0.0),
                "relative_strength_20d": row.get("relative_strength_20d", 0.0),
                "drawdown_60d": row.get("drawdown_60d", 0.0),
                "drawdown_vs_market_60d": row.get("drawdown_vs_market_60d", 0.0),
                "event_any": row.get("event_any", 0.0),
                "event_score_sum": float(row[event_columns].abs().sum()) if event_columns else 0.0,
                "daily_return": info["daily_return"],
                "ma20_60_return": info["ma20_60_return"],
                "portfolio_value": info["portfolio_value"],
                "policy_drawdown": info["drawdown"],
                "reward": reward,
            }
        )
        done = terminated or truncated

    return pd.DataFrame(rows)


def summarize_action_context(trace: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "all": pd.Series(True, index=trace.index),
        "event_any": trace["event_any"] > 0,
        "market_drop_2pct": trace["market_return_1d"] <= -0.02,
        "market_drop_1pct": trace["market_return_1d"] <= -0.01,
        "relative_strength_bottom_20pct": trace["relative_strength_20d"]
        <= trace["relative_strength_20d"].quantile(0.2),
        "stock_drawdown_20pct": trace["drawdown_60d"] <= -0.20,
        "underperforming_market": trace["drawdown_vs_market_60d"] <= -0.05,
    }
    rows = []
    for condition, mask in conditions.items():
        subset = trace[mask]
        if subset.empty:
            rows.append(
                {
                    "condition": condition,
                    "rows": 0,
                    "avg_action": None,
                    "avg_overlay": None,
                    "avg_target_ratio": None,
                    "avg_daily_return": None,
                    "action_counts": "{}",
                }
            )
            continue
        rows.append(
            {
                "condition": condition,
                "rows": len(subset),
                "avg_action": subset["action"].mean(),
                "avg_overlay": subset["overlay"].mean(),
                "avg_target_ratio": subset["target_ratio"].mean(),
                "avg_daily_return": subset["daily_return"].mean(),
                "action_counts": subset["action"].value_counts().sort_index().to_json(),
            }
        )
    return pd.DataFrame(rows)


def analyze_config(config_path: str, split: str, model_path: str, out_dir: str | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    features = pd.read_parquet(features_path)
    env_config = TradingEnvConfig(**config["trading"])
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    summaries = []
    for ticker in config["market"]["tickers"]:
        trace = trace_policy_actions(model_path, features, str(ticker).zfill(6), env_config)
        traces.append(trace)
        summary = summarize_action_context(trace)
        summary.insert(0, "ticker", str(ticker).zfill(6))
        summaries.append(summary)

    all_traces = pd.concat(traces, ignore_index=True)
    all_summaries = pd.concat(summaries, ignore_index=True)
    aggregate_summary = summarize_action_context(all_traces)
    aggregate_summary.insert(0, "ticker", "ALL")

    stem = Path(model_path).stem
    trace_path = output_dir / f"{stem}_{split}_action_trace.csv"
    summary_path = output_dir / f"{stem}_{split}_action_context_summary.csv"
    aggregate_path = output_dir / f"{stem}_{split}_action_context_aggregate.csv"
    all_traces.to_csv(trace_path, index=False)
    all_summaries.to_csv(summary_path, index=False)
    aggregate_summary.to_csv(aggregate_path, index=False)
    return {"trace": trace_path, "summary": summary_path, "aggregate": aggregate_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in analyze_config(args.config, args.split, args.model, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
