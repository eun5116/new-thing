from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from stock_rl.compare_strategies import compare_strategies
from stock_rl.config import load_config, project_path
from stock_rl.train_rl import train


EXPERIMENTS = [
    {
        "experiment_id": "E001_trade_absolute",
        "action_mode": "trade",
        "reward_mode": "absolute",
        "model_name": "ppo_SPY_E001_trade_absolute",
    },
    {
        "experiment_id": "E002_target_absolute",
        "action_mode": "target_position",
        "reward_mode": "absolute",
        "model_name": "ppo_SPY_E002_target_absolute",
    },
    {
        "experiment_id": "E003_target_excess",
        "action_mode": "target_position",
        "reward_mode": "excess_return",
        "model_name": "ppo_SPY_E003_target_excess",
    },
    {
        "experiment_id": "E004_target_risk_adjusted",
        "action_mode": "target_position",
        "reward_mode": "risk_adjusted",
        "model_name": "ppo_SPY_E004_target_risk_adjusted",
        "drawdown_penalty": 0.2,
        "turnover_penalty": 0.01,
    },
    {
        "experiment_id": "E005_target_risk_stronger_dd",
        "action_mode": "target_position",
        "reward_mode": "risk_adjusted",
        "model_name": "ppo_SPY_E005_target_risk_stronger_dd",
        "drawdown_penalty": 0.5,
        "turnover_penalty": 0.01,
    },
    {
        "experiment_id": "E006_dd_budget_15_20",
        "action_mode": "target_position",
        "reward_mode": "drawdown_budget",
        "model_name": "ppo_SPY_E006_dd_budget_15_20",
        "target_position_bins": 5,
        "drawdown_soft_limit": 0.15,
        "drawdown_hard_limit": 0.20,
        "drawdown_soft_penalty": 0.5,
        "drawdown_hard_penalty": 2.0,
        "turnover_penalty": 0.01,
    },
    {
        "experiment_id": "E007_dd_budget_18_22",
        "action_mode": "target_position",
        "reward_mode": "drawdown_budget",
        "model_name": "ppo_SPY_E007_dd_budget_18_22",
        "target_position_bins": 5,
        "drawdown_soft_limit": 0.18,
        "drawdown_hard_limit": 0.22,
        "drawdown_soft_penalty": 0.5,
        "drawdown_hard_penalty": 2.0,
        "turnover_penalty": 0.01,
    },
    {
        "experiment_id": "E008_dd_budget_20_only",
        "action_mode": "target_position",
        "reward_mode": "drawdown_budget",
        "model_name": "ppo_SPY_E008_dd_budget_20_only",
        "target_position_bins": 5,
        "drawdown_soft_limit": 0.20,
        "drawdown_hard_limit": 0.20,
        "drawdown_soft_penalty": 0.0,
        "drawdown_hard_penalty": 2.0,
        "turnover_penalty": 0.01,
    },
    {
        "experiment_id": "E009_dd_budget_low_turnover",
        "action_mode": "target_position",
        "reward_mode": "drawdown_budget",
        "model_name": "ppo_SPY_E009_dd_budget_low_turnover",
        "target_position_bins": 5,
        "drawdown_soft_limit": 0.20,
        "drawdown_hard_limit": 0.20,
        "drawdown_soft_penalty": 0.0,
        "drawdown_hard_penalty": 2.0,
        "turnover_penalty": 0.001,
    },
    {
        "experiment_id": "E010_dd_budget_6_bins",
        "action_mode": "target_position",
        "reward_mode": "drawdown_budget",
        "model_name": "ppo_SPY_E010_dd_budget_6_bins",
        "target_position_bins": 6,
        "drawdown_soft_limit": 0.20,
        "drawdown_hard_limit": 0.20,
        "drawdown_soft_penalty": 0.0,
        "drawdown_hard_penalty": 2.0,
        "turnover_penalty": 0.001,
    },
]


def write_experiment_config(base_config_path: str, experiment: dict[str, str], out_dir: Path) -> Path:
    config = deepcopy(load_config(base_config_path))
    config["trading"]["action_mode"] = experiment["action_mode"]
    config["trading"]["reward_mode"] = experiment["reward_mode"]
    config["trading"]["target_position_bins"] = experiment.get("target_position_bins", 5)
    if "drawdown_penalty" in experiment:
        config["trading"]["drawdown_penalty"] = experiment["drawdown_penalty"]
    if "turnover_penalty" in experiment:
        config["trading"]["turnover_penalty"] = experiment["turnover_penalty"]
    for key in [
        "drawdown_soft_limit",
        "drawdown_hard_limit",
        "drawdown_soft_penalty",
        "drawdown_hard_penalty",
        "hard_drawdown_terminate",
    ]:
        if key in experiment:
            config["trading"][key] = experiment[key]
    config["training"]["model_name"] = experiment["model_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment['experiment_id']}.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out_path


def append_results(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    base_config_path: str,
    split: str = "valid",
    ticker: str = "SPY",
    experiment_ids: set[str] | None = None,
) -> Path:
    configs_dir = project_path(base_config_path, "configs")
    report_path = project_path(base_config_path, "reports", "experiments.csv")
    rows: list[dict[str, object]] = []

    for experiment in EXPERIMENTS:
        if experiment_ids is not None and experiment["experiment_id"] not in experiment_ids:
            continue
        config_path = write_experiment_config(base_config_path, experiment, configs_dir)
        model_path = train(str(config_path), ticker)
        comparison = compare_strategies(str(config_path), split=split)
        record = comparison.iloc[0].to_dict()
        record.update(
            {
                "experiment_id": experiment["experiment_id"],
                "config": str(config_path),
                "model_path": model_path,
                "action_mode": experiment["action_mode"],
                "reward_mode": experiment["reward_mode"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        rows.append(record)

    append_results(report_path, rows)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args()
    experiment_ids = set(args.experiments) if args.experiments else None
    print(run_experiments(args.config, args.split, args.ticker, experiment_ids))


if __name__ == "__main__":
    main()
