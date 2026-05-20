from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import pandas as pd

from stock_rl.evaluate import performance_from_returns
from stock_rl.trading_env import StockTradingEnv, TradingEnvConfig


@dataclass(frozen=True)
class PolicyEvaluation:
    steps: int
    final_portfolio_value: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    sell_actions: int
    hold_actions: int
    buy_actions: int
    action_counts: str

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_policy(
    model_path: str,
    features_path: str,
    ticker: str,
    initial_cash: float = 1_000_000.0,
    env_config: TradingEnvConfig | None = None,
    feature_columns: list[str] | None = None,
) -> PolicyEvaluation:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for policy evaluation. "
            "Install CPU dependencies with: pip install -r requirements-rl-cpu.txt"
        ) from exc

    features = pd.read_parquet(features_path)
    env = StockTradingEnv(
        features,
        ticker=ticker,
        feature_columns=feature_columns,
        config=env_config or TradingEnvConfig(initial_cash=initial_cash),
    )
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    actions: list[int] = []
    portfolio_values: list[float] = []
    daily_returns: list[float] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        actions.append(int(action))
        portfolio_values.append(float(info["portfolio_value"]))
        daily_returns.append(float(info["daily_return"]))

    metrics = performance_from_returns(pd.Series(daily_returns))
    final_value = portfolio_values[-1]
    return PolicyEvaluation(
        steps=len(actions),
        final_portfolio_value=final_value,
        cumulative_return=final_value / initial_cash - 1.0,
        annualized_return=metrics.annualized_return,
        annualized_volatility=metrics.annualized_volatility,
        sharpe=metrics.sharpe,
        max_drawdown=metrics.max_drawdown,
        sell_actions=actions.count(0),
        hold_actions=actions.count(1),
        buy_actions=actions.count(2),
        action_counts=json.dumps({str(action): actions.count(action) for action in sorted(set(actions))}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", default="data/processed/test.parquet")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    args = parser.parse_args()
    print(evaluate_policy(args.model, args.features, args.ticker, args.initial_cash).to_dict())


if __name__ == "__main__":
    main()
