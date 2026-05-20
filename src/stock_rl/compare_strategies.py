from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.evaluate import buy_and_hold_metrics, cash_metrics, moving_average_metrics
from stock_rl.evaluate_policy import evaluate_policy
from stock_rl.trading_env import TradingEnvConfig


def _feature_columns_for_config(config: dict) -> list[str] | None:
    if config["training"].get("feature_scope") != "us_portfolio":
        return None
    from stock_rl.build_us_portfolio_features import US_FEATURE_COLUMNS

    return US_FEATURE_COLUMNS


def resolve_model_path(model_path: Path) -> Path | None:
    if model_path.exists():
        return model_path
    zip_path = model_path.with_suffix(model_path.suffix + ".zip") if model_path.suffix else model_path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    return None


def compare_strategies(
    config_path: str,
    split: str = "test",
    model_dir: str = "models",
    out_path: str | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", f"{split}.parquet")
    models_path = project_path(config_path, model_dir)
    features = pd.read_parquet(features_path)
    env_config = TradingEnvConfig(**config["trading"])
    feature_columns = _feature_columns_for_config(config)

    rows = []
    for ticker in config["market"]["tickers"]:
        bh = buy_and_hold_metrics(features, ticker)
        cash = cash_metrics(features, ticker)
        ma = moving_average_metrics(features, ticker)
        default_model_name = f"{config['training'].get('algorithm', 'PPO').lower()}_{ticker}"
        model_path = models_path / config["training"].get("model_name", default_model_name)
        resolved_model_path = resolve_model_path(model_path)
        row: dict[str, float | int | str] = {
            "ticker": ticker,
            "split": split,
            "cash_cumulative_return": cash.cumulative_return,
            "cash_sharpe": cash.sharpe,
            "buy_hold_cumulative_return": bh.cumulative_return,
            "buy_hold_annualized_return": bh.annualized_return,
            "buy_hold_sharpe": bh.sharpe,
            "buy_hold_max_drawdown": bh.max_drawdown,
            "ma20_60_cumulative_return": ma.cumulative_return,
            "ma20_60_sharpe": ma.sharpe,
            "ma20_60_max_drawdown": ma.max_drawdown,
        }
        if resolved_model_path is not None:
            try:
                policy = evaluate_policy(
                    str(resolved_model_path),
                    str(features_path),
                    ticker,
                    env_config=env_config,
                    feature_columns=feature_columns,
                )
                row.update(
                    {
                        "policy_cumulative_return": policy.cumulative_return,
                        "policy_annualized_return": policy.annualized_return,
                        "policy_sharpe": policy.sharpe,
                        "policy_max_drawdown": policy.max_drawdown,
                        "policy_final_portfolio_value": policy.final_portfolio_value,
                        "policy_sell_actions": policy.sell_actions,
                        "policy_hold_actions": policy.hold_actions,
                        "policy_buy_actions": policy.buy_actions,
                        "policy_action_counts": policy.action_counts,
                        "policy_minus_buy_hold_return": policy.cumulative_return - bh.cumulative_return,
                        "policy_minus_buy_hold_sharpe": policy.sharpe - bh.sharpe,
                    }
                )
            except ValueError as exc:
                row["policy_error"] = str(exc)
        else:
            row["model_missing"] = str(model_path)
        rows.append(row)

    result = pd.DataFrame(rows)
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = compare_strategies(args.config, args.split, args.model_dir, args.out)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
