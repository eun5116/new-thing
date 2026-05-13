from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from stock_rl.config import project_path


STRATEGIES = ["buy_hold_costed", "ma20_60_costed", "e028_ratio_replay"]


def _load_monthly(config_path: str, split: str) -> pd.DataFrame:
    path = project_path(config_path, "reports", f"meta_policy_monthly_strategy_returns_{split}.csv")
    return pd.read_csv(path)


def _choose(row: pd.Series, rule: dict[str, object]) -> str:
    allow_e028 = row["start_relative_strength_20d"] >= rule["min_relative_strength"]
    if rule["require_relative_regime_positive"]:
        allow_e028 = allow_e028 and row["start_relative_strength_regime"] > 0
    if rule["require_ma20_60_position"]:
        allow_e028 = allow_e028 and row["start_ma20_60_position"] > 0
    allow_e028 = allow_e028 and row["start_market_drop_recent_20d"] <= rule["max_market_drop_recent_20d"]
    allow_e028 = allow_e028 and row["start_event_recent_20d"] <= rule["max_event_recent_20d"]
    if allow_e028:
        return "e028_ratio_replay"
    if rule["fallback"] == "trend_then_bh" and row["start_ma20_60_position"] > 0:
        return "ma20_60_costed"
    return str(rule["fallback"]).replace("trend_then_bh", "buy_hold_costed")


def _evaluate_rule(monthly: pd.DataFrame, split: str, rule: dict[str, object]) -> dict[str, object]:
    choices = monthly.apply(lambda row: _choose(row, rule), axis=1)
    returns = [row[choice] for choice, (_, row) in zip(choices, monthly.iterrows())]
    result = monthly[["ticker", "month", "best_strategy", "best_return"]].copy()
    result["choice"] = choices
    result["chosen_return"] = returns
    result["regret_to_best"] = result["chosen_return"] - result["best_return"]
    return {
        "split": split,
        **rule,
        "rows": len(result),
        "avg_monthly_return": result["chosen_return"].mean(),
        "avg_regret_to_best": result["regret_to_best"].mean(),
        "match_best_share": (result["choice"] == result["best_strategy"]).mean(),
        "e028_choice_share": (result["choice"] == "e028_ratio_replay").mean(),
        "buy_hold_choice_share": (result["choice"] == "buy_hold_costed").mean(),
        "ma20_60_choice_share": (result["choice"] == "ma20_60_costed").mean(),
    }


def _rule_grid() -> list[dict[str, object]]:
    rules = []
    for (
        min_relative_strength,
        require_relative_regime_positive,
        require_ma20_60_position,
        max_market_drop_recent_20d,
        max_event_recent_20d,
        fallback,
    ) in product(
        [-0.05, 0.0, 0.05, 0.10],
        [False, True],
        [False, True],
        [0.0, 1.0],
        [0.0, 1.0],
        ["buy_hold_costed", "ma20_60_costed", "trend_then_bh"],
    ):
        name = (
            f"rs_{min_relative_strength:g}"
            f"_reg_{int(require_relative_regime_positive)}"
            f"_ma_{int(require_ma20_60_position)}"
            f"_drop20_{max_market_drop_recent_20d:g}"
            f"_event20_{max_event_recent_20d:g}"
            f"_{fallback}"
        )
        rules.append(
            {
                "rule": name,
                "min_relative_strength": min_relative_strength,
                "require_relative_regime_positive": require_relative_regime_positive,
                "require_ma20_60_position": require_ma20_60_position,
                "max_market_drop_recent_20d": max_market_drop_recent_20d,
                "max_event_recent_20d": max_event_recent_20d,
                "fallback": fallback,
            }
        )
    return rules


def evaluate(config_path: str, out_dir: str | None = None) -> dict[str, Path]:
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    valid = _load_monthly(config_path, "valid")
    test = _load_monthly(config_path, "test")
    valid_rows = [_evaluate_rule(valid, "valid", rule) for rule in _rule_grid()]
    valid_df = pd.DataFrame(valid_rows).sort_values("avg_monthly_return", ascending=False)
    best_rule = valid_df.iloc[0]["rule"]
    rule_by_name = {rule["rule"]: rule for rule in _rule_grid()}
    test_df = pd.DataFrame([_evaluate_rule(test, "test", rule_by_name[best_rule])])

    all_test_rows = [_evaluate_rule(test, "test", rule) for rule in _rule_grid()]
    all_test_df = pd.DataFrame(all_test_rows).sort_values("avg_monthly_return", ascending=False)

    valid_path = output_dir / "meta_policy_monthly_rule_grid_valid.csv"
    selected_test_path = output_dir / "meta_policy_monthly_rule_selected_test.csv"
    all_test_path = output_dir / "meta_policy_monthly_rule_grid_test.csv"
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(selected_test_path, index=False)
    all_test_df.to_csv(all_test_path, index=False)
    return {"valid_grid": valid_path, "selected_test": selected_test_path, "test_grid": all_test_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in evaluate(args.config, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
