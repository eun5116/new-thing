from __future__ import annotations

import argparse

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.trading_env import MultiTickerTradingEnv, StockTradingEnv, TradingEnvConfig, normalize_ticker


def _feature_columns_for_config(config: dict) -> list[str] | None:
    if config["training"].get("feature_scope") != "us_portfolio":
        if config["training"].get("feature_scope") == "krx_nps":
            from stock_rl.build_features import feature_columns_for_config

            return feature_columns_for_config(config)
        return None
    from stock_rl.build_us_portfolio_features import US_FEATURE_COLUMNS

    return US_FEATURE_COLUMNS


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
    env_config = TradingEnvConfig(**config["trading"])
    feature_columns = _feature_columns_for_config(config)
    train_scope = config["training"].get("train_scope", "single_ticker")
    if train_scope == "multi_ticker" and ticker is None:
        tickers = [normalize_ticker(ticker) for ticker in config["market"]["tickers"]]
        available = set(train_features["ticker"].astype(str).map(normalize_ticker))
        missing = sorted(set(tickers).difference(available))
        if missing:
            if config["training"].get("allow_missing_tickers"):
                print(f"Skipping tickers without training rows: {missing}")
                tickers = [ticker for ticker in tickers if ticker in available]
            else:
                raise ValueError(f"tickers not found in training features: {missing}")
        env = MultiTickerTradingEnv(train_features, tickers=tickers, feature_columns=feature_columns, config=env_config)
        model_ticker = "multi"
    else:
        ticker = ticker or config["market"]["tickers"][0]
        ticker = normalize_ticker(ticker)
        available = set(train_features["ticker"].astype(str).map(normalize_ticker))
        if ticker not in available:
            raise ValueError(f"ticker not found in training features: {ticker}")
        env = StockTradingEnv(train_features, ticker=ticker, feature_columns=feature_columns, config=env_config)
        model_ticker = ticker

    algo_name = config["training"].get("algorithm", "PPO")
    model_cls = _load_algorithms()[algo_name]
    model_kwargs = {
        "verbose": 1,
        "seed": config["training"].get("seed", 42),
    }
    for key in ["learning_rate", "ent_coef", "gamma", "gae_lambda", "clip_range"]:
        if key in config["training"]:
            model_kwargs[key] = config["training"][key]
    model = model_cls("MlpPolicy", env, **model_kwargs)
    model.learn(total_timesteps=config["training"].get("total_timesteps", 50_000))

    model_name = config["training"].get("model_name", f"{algo_name.lower()}_{model_ticker}")
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
