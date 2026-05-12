from __future__ import annotations

import argparse

import pandas as pd

from stock_rl.trading_env import StockTradingEnv


def run_smoke(features_path: str, ticker: str, max_steps: int = 250) -> dict:
    features = pd.read_parquet(features_path)
    env = StockTradingEnv(features, ticker=ticker)
    obs, _ = env.reset()
    rewards: list[float] = []
    actions = [2, 1, 1, 0, 1]
    info = {"portfolio_value": env.config.initial_cash}

    for step in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(actions[step % len(actions)])
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "ticker": ticker,
        "steps": len(rewards),
        "final_portfolio_value": float(info["portfolio_value"]),
        "total_reward": float(sum(rewards)),
        "last_observation_size": int(obs.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/train.parquet")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()
    print(run_smoke(args.features, args.ticker, args.max_steps))


if __name__ == "__main__":
    main()
