from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.build_trading_sheet import _load_reference, _markdown_table
from stock_rl.config import project_path
from stock_rl.trading_env import normalize_ticker


KEEP_COLUMNS = [
    "as_of_date",
    "previous_as_of_date",
    "ticker",
    "name",
    "market",
    "previous_target_pct",
    "target_pct",
    "target_delta_pct",
    "rebalance_action",
    "previous_cap_reason",
    "cap_reason",
    "return_20d_pct",
    "return_60d_pct",
    "drawdown_60d_pct",
]


def _target_paths(config_path: str, rule: str) -> list[Path]:
    reports_dir = project_path(config_path, "reports")
    return sorted(reports_dir.glob(f"current_targets_*_{rule}.csv"))


def _previous_target_path(config_path: str, rule: str, current_path: Path) -> Path | None:
    paths = [path for path in _target_paths(config_path, rule) if path != current_path]
    previous = [path for path in paths if path.name < current_path.name]
    return previous[-1] if previous else None


def _pct(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 100.0).round(2)


def _rebalance_action(delta_pct: float, threshold_pct: float) -> str:
    if delta_pct >= threshold_pct:
        return "increase"
    if delta_pct <= -threshold_pct:
        return "reduce"
    return "hold"


def build_target_change_report(
    config_path: str,
    rule: str = "strong_trend_full_else070",
    current_target_path: str | None = None,
    previous_target_path: str | None = None,
    threshold_pct: float = 5.0,
    out_dir: str | None = None,
) -> dict[str, Path] | None:
    current_path = Path(current_target_path) if current_target_path else _target_paths(config_path, rule)[-1]
    previous_path = Path(previous_target_path) if previous_target_path else _previous_target_path(config_path, rule, current_path)
    if previous_path is None:
        return None

    current = pd.read_csv(current_path, dtype={"ticker": str})
    previous = pd.read_csv(previous_path, dtype={"ticker": str})
    current["ticker"] = current["ticker"].map(normalize_ticker)
    previous["ticker"] = previous["ticker"].map(normalize_ticker)

    frame = current.merge(
        previous[["ticker", "as_of_date", "target_ratio", "cap_reason"]],
        how="left",
        on="ticker",
        suffixes=("", "_previous"),
    )
    frame["previous_target_ratio"] = frame["target_ratio_previous"].fillna(0.0)
    frame["previous_cap_reason"] = frame["cap_reason_previous"].fillna("new")
    frame["previous_as_of_date"] = frame["as_of_date_previous"].fillna("")
    frame["target_delta_pct"] = ((frame["target_ratio"] - frame["previous_target_ratio"]) * 100.0).round(2)
    frame["target_pct"] = _pct(frame["target_ratio"])
    frame["previous_target_pct"] = _pct(frame["previous_target_ratio"])
    frame["return_20d_pct"] = _pct(frame["return_20d"])
    frame["return_60d_pct"] = _pct(frame["return_60d"])
    frame["drawdown_60d_pct"] = _pct(frame["drawdown_60d"])
    frame["rebalance_action"] = frame["target_delta_pct"].map(lambda value: _rebalance_action(float(value), threshold_pct))

    reference = _load_reference(config_path)
    frame = frame.merge(reference, how="left", on="ticker")
    frame["name"] = frame["name"].fillna("")
    frame["market"] = frame["market"].fillna("")
    frame = frame[KEEP_COLUMNS].sort_values(
        ["target_delta_pct", "target_pct", "return_20d_pct"],
        ascending=[False, False, False],
    )

    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(current["as_of_date"].max()).replace("-", "")
    csv_path = output_dir / f"target_changes_{as_of}_{rule}.csv"
    md_path = output_dir / f"target_changes_{as_of}_{rule}.md"
    frame.to_csv(csv_path, index=False)
    _write_markdown(md_path, frame, rule, current_path, previous_path, threshold_pct)
    return {"csv": csv_path, "markdown": md_path}


def _write_markdown(path: Path, frame: pd.DataFrame, rule: str, current_path: Path, previous_path: Path, threshold_pct: float) -> None:
    as_of = str(frame["as_of_date"].max())
    previous_as_of = str(frame["previous_as_of_date"].replace("", pd.NA).dropna().max())
    lines = [
        f"# Target Changes - {as_of}",
        "",
        f"- rule: `{rule}`",
        f"- current: `{current_path}`",
        f"- previous: `{previous_path}`",
        f"- threshold: `{threshold_pct:.1f}%`",
        f"- increase count: `{int((frame['rebalance_action'] == 'increase').sum())}`",
        f"- reduce count: `{int((frame['rebalance_action'] == 'reduce').sum())}`",
        f"- hold count: `{int((frame['rebalance_action'] == 'hold').sum())}`",
        "",
        "## Increase",
        "",
        _markdown_table(
            frame[frame["rebalance_action"] == "increase"].head(20)[
                [
                    "ticker",
                    "name",
                    "market",
                    "previous_target_pct",
                    "target_pct",
                    "target_delta_pct",
                    "cap_reason",
                    "return_20d_pct",
                    "drawdown_60d_pct",
                ]
            ]
        ),
        "",
        "## Reduce",
        "",
        _markdown_table(
            frame[frame["rebalance_action"] == "reduce"].sort_values("target_delta_pct").head(20)[
                [
                    "ticker",
                    "name",
                    "market",
                    "previous_target_pct",
                    "target_pct",
                    "target_delta_pct",
                    "cap_reason",
                    "return_20d_pct",
                    "drawdown_60d_pct",
                ]
            ]
        ),
        "",
        "## Context",
        "",
        f"Compared `{previous_as_of}` to `{as_of}`. `target_delta_pct` is percentage-point change in target exposure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--current-target-path", default=None)
    parser.add_argument("--previous-target-path", default=None)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    result = build_target_change_report(
        args.config,
        args.rule,
        args.current_target_path,
        args.previous_target_path,
        args.threshold_pct,
        args.out_dir,
    )
    if result is None:
        print("no previous target file found")
        return
    for name, path in result.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
