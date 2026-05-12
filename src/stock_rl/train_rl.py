from __future__ import annotations

import argparse

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.trading_env import StockTradingEnv, TradingEnvConfig


def _load_algorithms():
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for RL training. "
            "Install CPU dependencies with: pip install -r requirements-rl-cpu.txt"
        ) from exc
    return {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def train(config_path: str, ticker: str | None = None) -> str:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    train_path = project_path(config_path, data_dir, "processed", "train.parquet")
    train_features = pd.read_parquet(train_path)
    ticker = ticker or config["market"]["tickers"][0]
    if ticker not in set(train_features["ticker"]):
        raise ValueError(f"ticker not found in training features: {ticker}")
    env_config = TradingEnvConfig(**config["trading"])
    env = StockTradingEnv(train_features, ticker=ticker, config=env_config)

    algo_name = config["training"].get("algorithm", "PPO")
    model_cls = _load_algorithms()[algo_name]
    model = model_cls("MlpPolicy", env, verbose=1, seed=config["training"].get("seed", 42))
    model.learn(total_timesteps=config["training"].get("total_timesteps", 50_000))

    model_name = config["training"].get("model_name", f"{algo_name.lower()}_{ticker}")
    out_path = project_path(config_path, "models", model_name)
    model.save(out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    print(train(args.config, args.ticker))


if __name__ == "__main__":
    main()
