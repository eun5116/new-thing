from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.evaluate_regime_exposure_cap import (
    RULES,
    _cap_for_row,
    _feature_columns_for_model,
    _resolve_model_path,
    _target_ratio_from_action,
)
from stock_rl.trading_env import TradingEnvConfig, normalize_ticker


def _load_positions(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    if not {"ticker", "position_ratio"}.issubset(frame.columns):
        raise ValueError("positions CSV must contain ticker and position_ratio columns")
    frame["ticker"] = frame["ticker"].astype(str).map(normalize_ticker)
    return dict(zip(frame["ticker"], frame["position_ratio"].astype(float)))


def _resolve_rule(name: str):
    for rule in RULES:
        if rule.name == name:
            return rule
    available = ", ".join(rule.name for rule in RULES)
    raise ValueError(f"unknown rule: {name}. Available rules: {available}")


def generate_targets(
    config_path: str,
    rule_name: str = "strong_trend_full_else070",
    features_name: str = "daily_features.parquet",
    model_name: str | None = None,
    default_position_ratio: float = 0.0,
    positions_path: str | None = None,
    out_path: str | None = None,
) -> Path:
    matplotlib_cache_dir = Path("/tmp/stock_rl_matplotlib")
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to generate current targets") from exc

    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", features_name)
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    as_of_date = features["date"].max()
    latest = features.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    latest = latest.sort_values("ticker").reset_index(drop=True)

    env_config = TradingEnvConfig(**config["trading"])
    resolved_model_name = model_name or config["training"]["model_name"]
    model = PPO.load(_resolve_model_path(config_path, resolved_model_name))
    rule = _resolve_rule(rule_name)
    feature_columns = _feature_columns_for_model(features, config, model.observation_space.shape[0])
    positions = _load_positions(positions_path)

    rows = []
    for _, row in latest.iterrows():
        ticker = str(row["ticker"])
        position_ratio = float(positions.get(ticker, default_position_ratio))
        position_ratio = min(max(position_ratio, 0.0), 1.0)
        cash_ratio = 1.0 - position_ratio
        obs = np.asarray([float(row[col]) for col in feature_columns] + [cash_ratio, position_ratio], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        raw_target_ratio = _target_ratio_from_action(action, env_config)
        cap, cap_reason = _cap_for_row(row, rule)
        target_ratio = min(raw_target_ratio, cap)
        rows.append(
            {
                "as_of_date": as_of_date.date().isoformat(),
                "feature_date": row["date"].date().isoformat(),
                "ticker": ticker,
                "rule": rule.name,
                "model_name": resolved_model_name,
                "assumed_position_ratio": position_ratio,
                "action": action,
                "raw_target_ratio": raw_target_ratio,
                "cap": cap,
                "cap_reason": cap_reason,
                "target_ratio": target_ratio,
                "market_return_60d": row.get("market_return_60d", 0.0),
                "market_return_120d": row.get("market_return_120d", 0.0),
                "market_ma60_gap": row.get("market_ma60_gap", 0.0),
                "market_ma120_gap": row.get("market_ma120_gap", 0.0),
                "relative_strength_20d": row.get("relative_strength_20d", 0.0),
                "return_20d": row.get("return_20d", 0.0),
                "return_60d": row.get("return_60d", 0.0),
                "drawdown_60d": row.get("drawdown_60d", 0.0),
            }
        )

    result = pd.DataFrame(rows).sort_values(["target_ratio", "raw_target_ratio", "ticker"], ascending=[False, False, True])
    if out_path:
        path = Path(out_path)
    else:
        safe_rule = rule.name.replace("/", "_")
        path = project_path(config_path, "reports", f"current_targets_{as_of_date:%Y%m%d}_{safe_rule}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--features-name", default="daily_features.parquet")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--default-position-ratio", type=float, default=0.0)
    parser.add_argument("--positions", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(
        generate_targets(
            args.config,
            args.rule,
            args.features_name,
            args.model_name,
            args.default_position_ratio,
            args.positions,
            args.out,
        )
    )


if __name__ == "__main__":
    main()
