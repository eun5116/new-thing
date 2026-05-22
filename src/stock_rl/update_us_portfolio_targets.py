from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.build_features import FEATURE_COLUMNS
from stock_rl.build_us_portfolio_features import US_FEATURE_COLUMNS, build_us_features, collect_us_prices
from stock_rl.config import load_config, project_path
from stock_rl.evaluate_regime_exposure_cap import _feature_columns as _legacy_feature_columns
from stock_rl.evaluate_regime_exposure_cap import _target_ratio_from_action
from stock_rl.sec_edgar import collect_sec_companyfacts
from stock_rl.trading_env import TradingEnvConfig


DEFAULT_POLICY_PATH = "configs/portfolio_policy.yaml"


def latest_raw_price_start(config_path: str | Path) -> str:
    config = load_config(config_path)
    tickers = [*config["market"]["tickers"], *config["market"].get("benchmark_tickers", [])]
    price_dir = project_path(config_path, config["project"]["data_dir"], "raw", "prices")
    latest_dates = []
    for ticker in tickers:
        path = price_dir / f"{ticker}.parquet"
        if not path.exists():
            return str(config["market"]["start"])
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            return str(config["market"]["start"])
        latest_dates.append(pd.to_datetime(frame["date"]).max())
    latest = min(latest_dates)
    return (latest + pd.Timedelta(days=1)).date().isoformat()


def generate_us_targets(
    config_path: str | Path,
    out_path: str | Path | None = None,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> Path:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to generate US targets") from exc

    config = load_config(config_path)
    data_dir = config["project"]["data_dir"]
    features_path = project_path(config_path, data_dir, "processed", "daily_features.parquet")
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])
    env_config = TradingEnvConfig(**config["trading"])
    model_name = config["training"]["model_name"]
    model_path = project_path(config_path, "models", f"{model_name}.zip")
    if not model_path.exists():
        model_path = project_path(config_path, "models", model_name)
    model = PPO.load(model_path)
    feature_columns = _feature_columns_for_model(features, model.observation_space.shape[0])
    policy = _load_policy(config_path, policy_path)

    rows = []
    for ticker in config["market"]["tickers"]:
        history = features[features["ticker"].astype(str).eq(str(ticker))].sort_values("date")
        if history.empty:
            continue
        row = history.iloc[-1]
        obs = np.asarray([float(row[col]) for col in feature_columns] + [1.0, 0.0], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        raw_target_ratio = _target_ratio_from_action(action, env_config)
        policy_cap, policy_group, policy_label = _policy_name_cap(ticker, policy)
        target_ratio = min(raw_target_ratio, policy_cap)
        policy_cap_reason = "name_cap" if target_ratio < raw_target_ratio else "none"
        rows.append(
            {
                "as_of_date": row["date"].date().isoformat(),
                "ticker": str(ticker),
                "model_name": model_name,
                "action": action,
                "raw_target_ratio": raw_target_ratio,
                "raw_target_ratio_pct": round(raw_target_ratio * 100.0, 1),
                "policy_group": policy_group,
                "policy_group_label": policy_label,
                "policy_name_cap": policy_cap,
                "policy_name_cap_pct": round(policy_cap * 100.0, 1),
                "policy_cap_reason": policy_cap_reason,
                "target_ratio": target_ratio,
                "target_ratio_pct": round(target_ratio * 100.0, 1),
                "return_20d": float(row["return_20d"]),
                "return_60d": float(row["return_60d"]),
                "drawdown_60d": float(row["drawdown_60d"]),
                "relative_strength_20d": float(row["relative_strength_20d"]),
                "ma20_60_position": float(row["ma20_60_position"]),
                "market_return_20d": float(row["market_return_1d"]) if "market_return_1d" in row else 0.0,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["target_ratio", "raw_target_ratio", "relative_strength_20d"],
        ascending=[False, False, False],
    )
    as_of = result["as_of_date"].max() if not result.empty else pd.Timestamp.today().date().isoformat()
    path = Path(out_path) if out_path else project_path(config_path, "reports", f"us_portfolio_targets_{as_of.replace('-', '')}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path


def _load_policy(config_path: str | Path, policy_path: str | Path) -> dict:
    path = Path(policy_path)
    if not path.is_absolute():
        path = project_path(config_path, policy_path)
    if not path.exists():
        return {}
    return load_config(path)


def _policy_name_cap(ticker: str, policy: dict) -> tuple[float, str, str]:
    ticker_groups = {str(key).upper(): str(value) for key, value in policy.get("ticker_groups", {}).items()}
    group = ticker_groups.get(str(ticker).upper(), "us_large_cap")
    settings = policy.get("groups", {}).get(group, policy.get("groups", {}).get("manual_review", {}))
    cap_pct = float(settings.get("max_name_weight_pct", 100.0))
    label = str(settings.get("label", group))
    return max(0.0, min(cap_pct / 100.0, 1.0)), group, label


def _feature_columns_for_model(features: pd.DataFrame, observation_size: int) -> list[str]:
    event_columns = sorted(
        column for column in features.columns if column.startswith("event_") and column not in US_FEATURE_COLUMNS
    )
    candidates = [
        US_FEATURE_COLUMNS,
        [*US_FEATURE_COLUMNS, *event_columns],
        FEATURE_COLUMNS,
        [*FEATURE_COLUMNS, *event_columns],
        _legacy_feature_columns(features),
    ]
    for columns in candidates:
        if len(columns) + 2 == observation_size:
            return columns
    sizes = ", ".join(str(len(columns) + 2) for columns in candidates)
    raise ValueError(f"no US feature column set matches model observation size {observation_size}; tried {sizes}")


def update_paper_log(config_path: str | Path, target_path: str | Path) -> Path:
    config = load_config(config_path)
    features = pd.read_parquet(project_path(config_path, config["project"]["data_dir"], "processed", "daily_features.parquet"))
    features["date"] = pd.to_datetime(features["date"])
    targets = pd.read_csv(target_path, dtype={"ticker": str})
    targets["as_of_date"] = pd.to_datetime(targets["as_of_date"])
    log_path = project_path(config_path, "reports", "us_portfolio_paper_log.csv")
    existing = pd.read_csv(log_path, dtype={"ticker": str}) if log_path.exists() else pd.DataFrame()

    rows = []
    for _, target in targets.iterrows():
        ticker = str(target["ticker"])
        as_of = pd.Timestamp(target["as_of_date"])
        history = features[features["ticker"].astype(str).eq(ticker)].sort_values("date")
        current_row = history[history["date"].eq(as_of)]
        next_row = history[history["date"].gt(as_of)].head(1)
        next_return = None
        realized_date = ""
        if not current_row.empty and pd.notna(current_row.iloc[0].get("target_return_1d")):
            next_return = float(current_row.iloc[0]["target_return_1d"])
            realized_date = next_row.iloc[0]["date"].date().isoformat() if not next_row.empty else ""
        rows.append(
            {
                "signal_date": as_of.date().isoformat(),
                "ticker": ticker,
                "target_ratio": float(target["target_ratio"]),
                "target_ratio_pct": float(target["target_ratio_pct"]),
                "next_trade_date": realized_date,
                "next_return_1d": next_return,
                "target_weighted_return_1d": None if next_return is None else float(target["target_ratio"]) * next_return,
            }
        )
    new_log = pd.DataFrame(rows)
    combined = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
    combined = combined.drop_duplicates(["signal_date", "ticker"], keep="last").sort_values(["signal_date", "ticker"])
    combined.to_csv(log_path, index=False)
    return log_path


def update_us_portfolio_targets(
    config_path: str = "configs/US_PORTFOLIO_HELD.yaml",
    start: str | None = None,
    end: str | None = None,
    skip_collect: bool = False,
    skip_sec: bool = False,
    policy_path: str = DEFAULT_POLICY_PATH,
) -> dict[str, object]:
    resolved_start = start or latest_raw_price_start(config_path)
    price_paths = [] if skip_collect else collect_us_prices(config_path, start=resolved_start, end=end)
    feature_paths = build_us_features(config_path)
    sec_paths = {} if skip_sec else collect_sec_companyfacts(config_path)
    target_path = generate_us_targets(config_path, policy_path=policy_path)
    paper_log_path = update_paper_log(config_path, target_path)
    targets = pd.read_csv(target_path)
    return {
        "collection_start": resolved_start,
        "collection_end": end or "today",
        "price_files": len(price_paths),
        "features": feature_paths,
        "sec": sec_paths,
        "target_path": target_path,
        "paper_log_path": paper_log_path,
        "summary": {
            "as_of_date": targets["as_of_date"].max() if not targets.empty else "",
            "tickers": int(targets["ticker"].nunique()) if not targets.empty else 0,
            "avg_target_pct": float(targets["target_ratio_pct"].mean()) if not targets.empty else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/US_PORTFOLIO_HELD.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-sec", action="store_true")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    args = parser.parse_args()
    result = update_us_portfolio_targets(args.config, args.start, args.end, args.skip_collect, args.skip_sec, args.policy)
    summary = result["summary"]
    print(f"collection: {result['collection_start']}..{result['collection_end']}")
    print(f"price_files: {result['price_files']}")
    print(f"target: {result['target_path']}")
    print(f"paper_log: {result['paper_log_path']}")
    if result["sec"]:
        print(f"sec_snapshot: {result['sec']['snapshot']}")
    print(f"summary: as_of={summary['as_of_date']} tickers={summary['tickers']} avg_target={summary['avg_target_pct']:.1f}%")


if __name__ == "__main__":
    main()
