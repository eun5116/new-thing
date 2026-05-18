from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from stock_rl.build_portfolio_decision_sheet import _latest_path
from stock_rl.build_trading_sheet import _markdown_table
from stock_rl.config import load_config, project_path
from stock_rl.trading_env import normalize_ticker


DEFAULT_POLICY_PATH = "configs/portfolio_policy.yaml"


def _num(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _group_settings(policy: dict[str, Any], group: str) -> dict[str, Any]:
    return policy.get("groups", {}).get(group, policy.get("groups", {}).get("manual_review", {}))


def classify_asset(row: pd.Series, policy: dict[str, Any]) -> str:
    ticker = normalize_ticker(str(row.get("ticker", "")))
    ticker_groups = {normalize_ticker(str(key)): value for key, value in policy.get("ticker_groups", {}).items()}
    if ticker in ticker_groups:
        return str(ticker_groups[ticker])

    scope = str(row.get("asset_scope", ""))
    if scope == "model_universe":
        return "krx_model"
    if scope == "krx_stock_outside_model":
        return "krx_trend"
    if scope == "krx_etf_or_unmapped":
        return "manual_review"
    if scope == "us_or_global":
        vol = _num(row, "volatility_20d_pct")
        weight = _num(row, "current_weight_pct")
        if vol >= 80 or weight <= 3:
            return "us_speculative"
        return "us_large_cap"
    return "manual_review"


def _policy_decision(row: pd.Series) -> tuple[str, str]:
    weight = _num(row, "current_weight_pct")
    group_weight = _num(row, "group_weight_pct")
    max_group = _num(row, "group_max_total_weight_pct")
    max_name = _num(row, "group_max_name_weight_pct")
    trend = str(row.get("trend_status", "unknown"))
    pnl = _num(row, "pnl_pct")
    vol = _num(row, "volatility_20d_pct")
    group = str(row.get("policy_group", "manual_review"))

    reasons = []
    if group_weight > max_group:
        reasons.append("group_over_cap")
    if weight > max_name:
        reasons.append("name_over_cap")
    if trend == "downtrend":
        reasons.append("downtrend")
    if pnl < 0:
        reasons.append("unrealized_loss")
    if vol >= 80:
        reasons.append("high_volatility")

    if group == "manual_review":
        return "manual_review", "manual_group"
    if weight > max_name:
        return "trim_to_name_cap", "|".join(reasons)
    if group_weight > max_group and (trend == "downtrend" or pnl < 0 or vol >= 80):
        return "trim_group_candidate", "|".join(reasons)
    if group in {"us_speculative", "etf_theme"}:
        return "cap_watch", "|".join(reasons or ["satellite_cap"])
    if trend == "downtrend" and pnl < 0:
        return "trim_candidate", "|".join(reasons)
    return "within_policy", "|".join(reasons or ["within_caps"])


def build_portfolio_policy_sheet(
    config_path: str,
    policy_path: str = DEFAULT_POLICY_PATH,
    position_analysis_path: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Path]:
    policy = load_config(policy_path)
    position_path = (
        Path(position_analysis_path)
        if position_analysis_path
        else _latest_path(config_path, "current_position_analysis_*.csv")
    )
    positions = pd.read_csv(position_path, dtype={"ticker": str})
    positions["ticker"] = positions["ticker"].map(normalize_ticker)
    positions["policy_group"] = positions.apply(lambda row: classify_asset(row, policy), axis=1)

    group_weights = positions.groupby("policy_group")["current_weight_pct"].sum().to_dict()
    rows = []
    for _, row in positions.iterrows():
        group = str(row["policy_group"])
        settings = _group_settings(policy, group)
        row = row.copy()
        row["policy_group_label"] = settings.get("label", group)
        row["group_weight_pct"] = round(float(group_weights.get(group, 0.0)), 2)
        row["group_max_total_weight_pct"] = float(settings.get("max_total_weight_pct", 0.0))
        row["group_max_name_weight_pct"] = float(settings.get("max_name_weight_pct", 0.0))
        row["group_excess_pct"] = round(max(0.0, row["group_weight_pct"] - row["group_max_total_weight_pct"]), 2)
        row["name_excess_pct"] = round(max(0.0, _num(row, "current_weight_pct") - row["group_max_name_weight_pct"]), 2)
        row["policy_rule"] = settings.get("action", "")
        decision, reason = _policy_decision(row)
        row["policy_decision"] = decision
        row["policy_reason"] = reason
        rows.append(row)

    frame = pd.DataFrame(rows)
    keep = [
        "ticker",
        "name",
        "asset_scope",
        "policy_group",
        "policy_group_label",
        "current_weight_pct",
        "group_weight_pct",
        "group_max_total_weight_pct",
        "group_excess_pct",
        "group_max_name_weight_pct",
        "name_excess_pct",
        "pnl_pct",
        "trend_status",
        "return_20d_pct",
        "drawdown_60d_pct",
        "volatility_20d_pct",
        "policy_decision",
        "policy_reason",
    ]
    frame = frame[[column for column in keep if column in frame.columns]].sort_values(
        ["policy_decision", "current_weight_pct"],
        ascending=[True, False],
    )

    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.today().strftime("%Y%m%d")
    csv_path = output_dir / f"portfolio_policy_sheet_{as_of}.csv"
    md_path = output_dir / f"portfolio_policy_sheet_{as_of}.md"
    frame.to_csv(csv_path, index=False)
    _write_markdown(md_path, frame, position_path, Path(policy_path))
    return {"csv": csv_path, "markdown": md_path}


def _write_markdown(path: Path, frame: pd.DataFrame, position_path: Path, policy_path: Path) -> None:
    group_summary = (
        frame.groupby(["policy_group", "policy_group_label"], as_index=False)
        .agg(
            group_weight_pct=("current_weight_pct", "sum"),
            group_max_total_weight_pct=("group_max_total_weight_pct", "first"),
            holdings=("ticker", "count"),
        )
        .sort_values("group_weight_pct", ascending=False)
    )
    group_summary["group_weight_pct"] = group_summary["group_weight_pct"].round(2)
    group_summary["group_excess_pct"] = (
        group_summary["group_weight_pct"] - group_summary["group_max_total_weight_pct"]
    ).clip(lower=0.0).round(2)

    lines = [
        "# Portfolio Policy Sheet",
        "",
        f"- positions: `{position_path}`",
        f"- policy: `{policy_path}`",
        f"- holdings: `{len(frame)}`",
        "",
        "## Group Summary",
        "",
        _markdown_table(group_summary),
        "",
        "## Policy Decisions",
        "",
        _markdown_table(frame),
        "",
        "## Notes",
        "",
        "- This sheet is a portfolio risk overlay, not an automatic order list.",
        "- E032 KRX targets remain separate; this sheet controls non-model assets with group and name caps.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--position-analysis", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in build_portfolio_policy_sheet(args.config, args.policy, args.position_analysis, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
