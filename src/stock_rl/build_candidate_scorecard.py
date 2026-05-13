from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.config import project_path


EXPERIMENT_REPORTS = {
    "E028_policy": "KRX_E028_liquid48_target_hybrid_aggressive",
    "E030_policy": "KRX_E030_liquid48_long_trend_aggressive",
    "E032_policy": "KRX_E032_liquid48_long_trend_min_exposure",
    "E034_policy": "KRX_E034_liquid48_long_trend_max_exposure_085",
}

REGIME_STRATEGIES = [
    "uncapped",
    "strong_trend_full_else070",
    "strong_trend_full_else080",
    "strong_trend_or_very_strong_stock_full_else070",
    "strong_trend_or_very_strong_stock_full_else080",
]


def _mean_row(
    split: str,
    strategy: str,
    source: str,
    frame: pd.DataFrame,
    return_col: str,
    annualized_return_col: str | None,
    sharpe_col: str,
    mdd_col: str,
    target_ratio_col: str | None = None,
) -> dict[str, float | int | str | None]:
    return {
        "split": split,
        "strategy": strategy,
        "source": source,
        "tickers": int(len(frame)),
        "avg_cumulative_return": float(frame[return_col].mean()),
        "avg_annualized_return": float(frame[annualized_return_col].mean()) if annualized_return_col else None,
        "avg_sharpe": float(frame[sharpe_col].mean()),
        "avg_max_drawdown": float(frame[mdd_col].mean()),
        "avg_target_ratio": float(frame[target_ratio_col].mean()) if target_ratio_col else None,
    }


def _load_experiment_rows(reports_dir: Path, split: str) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    reference_path = reports_dir / f"{EXPERIMENT_REPORTS['E028_policy']}_{split}.csv"
    reference = pd.read_csv(reference_path)
    rows.append(
        _mean_row(
            split,
            "buy_hold",
            reference_path.name,
            reference,
            "buy_hold_cumulative_return",
            "buy_hold_annualized_return",
            "buy_hold_sharpe",
            "buy_hold_max_drawdown",
        )
    )
    rows.append(
        _mean_row(
            split,
            "ma20_60",
            reference_path.name,
            reference,
            "ma20_60_cumulative_return",
            None,
            "ma20_60_sharpe",
            "ma20_60_max_drawdown",
        )
    )
    for strategy, stem in EXPERIMENT_REPORTS.items():
        path = reports_dir / f"{stem}_{split}.csv"
        frame = pd.read_csv(path)
        rows.append(
            _mean_row(
                split,
                strategy,
                path.name,
                frame,
                "policy_cumulative_return",
                "policy_annualized_return",
                "policy_sharpe",
                "policy_max_drawdown",
            )
        )
    return rows


def _load_regime_rows(reports_dir: Path, split: str) -> list[dict[str, float | int | str | None]]:
    path = reports_dir / f"regime_exposure_cap_{split}_summary.csv"
    summary = pd.read_csv(path)
    rows = []
    for strategy in REGIME_STRATEGIES:
        frame = summary[summary["strategy"] == strategy]
        if frame.empty:
            continue
        row = frame.iloc[0]
        rows.append(
            {
                "split": split,
                "strategy": f"E032_replay_{strategy}",
                "source": path.name,
                "tickers": int(row["tickers"]),
                "avg_cumulative_return": float(row["avg_cumulative_return"]),
                "avg_annualized_return": float(row["avg_annualized_return"]),
                "avg_sharpe": float(row["avg_sharpe"]),
                "avg_max_drawdown": float(row["avg_max_drawdown"]),
                "avg_target_ratio": float(row["avg_target_ratio"]),
            }
        )
    return rows


def build_scorecard(config_path: str, out_dir: str | None = None) -> dict[str, Path]:
    reports_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    paths: dict[str, Path] = {}
    all_rows = []
    for split in ["valid", "test"]:
        rows = _load_experiment_rows(reports_dir, split) + _load_regime_rows(reports_dir, split)
        frame = pd.DataFrame(rows).sort_values("avg_cumulative_return", ascending=False)
        path = reports_dir / f"candidate_scorecard_{split}.csv"
        frame.to_csv(path, index=False)
        paths[split] = path
        all_rows.append(frame)

    combined = pd.concat(all_rows, ignore_index=True)
    pivot = combined.pivot(index="strategy", columns="split", values=["avg_cumulative_return", "avg_sharpe", "avg_max_drawdown"])
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot = pivot.reset_index()
    pivot["valid_rank_return"] = pivot["avg_cumulative_return_valid"].rank(ascending=False, method="min")
    pivot["test_rank_return"] = pivot["avg_cumulative_return_test"].rank(ascending=False, method="min")
    pivot["valid_test_return_gap"] = pivot["avg_cumulative_return_test"] - pivot["avg_cumulative_return_valid"]
    combined_path = reports_dir / "candidate_scorecard_valid_test.csv"
    pivot.sort_values(["valid_rank_return", "test_rank_return"]).to_csv(combined_path, index=False)
    paths["combined"] = combined_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in build_scorecard(args.config, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
