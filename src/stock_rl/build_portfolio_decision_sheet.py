from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.build_trading_sheet import _markdown_table
from stock_rl.config import load_config, project_path
from stock_rl.report_png import render_decision_sheet_png


def _latest_path(config_path: str, pattern: str) -> Path:
    paths = sorted(project_path(config_path, "reports").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no report files found: {pattern}")
    return paths[-1]


def _num(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _decision(row: pd.Series) -> tuple[str, str]:
    scope = str(row.get("asset_scope", ""))
    trend = str(row.get("trend_status", "unknown"))
    pnl = _num(row, "pnl_pct")
    weight = _num(row, "current_weight_pct")
    vol = _num(row, "volatility_20d_pct")
    dd = _num(row, "drawdown_60d_pct")
    order_action = str(row.get("order_action", ""))

    reasons = []
    if weight >= 20:
        reasons.append("large_weight")
    if trend == "downtrend":
        reasons.append("downtrend")
    if vol >= 80:
        reasons.append("high_volatility")
    if dd <= -15:
        reasons.append("drawdown_warning")
    if pnl < 0:
        reasons.append("unrealized_loss")

    if scope == "model_universe":
        if order_action == "buy":
            return "add_candidate", "model_allocator_buy"
        if order_action == "sell":
            return "trim_to_allocator", "above_allocator_target"
        return "keep", "model_target_available"

    if scope == "krx_stock_outside_model":
        if dd <= -15 or pnl < 0:
            return "watch_or_trim", "|".join(reasons + ["krx_trend_only"])
        if trend == "uptrend":
            return "keep_watch", "uptrend|krx_trend_only"
        return "watch", "krx_trend_only"

    if scope == "us_or_global":
        if weight >= 20 and (trend == "downtrend" or vol >= 80):
            return "trim_candidate", "|".join(reasons + ["outside_model"])
        if trend == "downtrend" and pnl < 0:
            return "trim_candidate", "|".join(reasons + ["outside_model"])
        if vol >= 80:
            return "speculative_watch", "|".join(reasons + ["outside_model"])
        if trend == "uptrend":
            return "keep_watch", "uptrend|outside_model"
        return "watch", "outside_model"

    if scope == "krx_etf_or_unmapped":
        return "manual_review", "etf_or_unmapped|outside_model"

    return "watch", "unclassified"


def build_decision_sheet(
    config_path: str,
    position_analysis_path: str | None = None,
    rebalance_orders_path: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Path]:
    position_path = Path(position_analysis_path) if position_analysis_path else _latest_path(config_path, "current_position_analysis_*.csv")
    rebalance_path = Path(rebalance_orders_path) if rebalance_orders_path else _latest_path(config_path, "rebalance_orders_*_strong_trend_full_else070.csv")
    positions = pd.read_csv(position_path, dtype={"ticker": str})
    orders = pd.read_csv(rebalance_path, dtype={"ticker": str})
    order_keep = ["ticker", "target_weight_pct", "weight_delta_pct", "order_amount", "order_action"]
    frame = positions.merge(orders[order_keep], how="left", on="ticker")

    def _coalesce(name: str, default: float | str = 0.0) -> None:
        if name in frame.columns:
            return
        left = f"{name}_x"
        right = f"{name}_y"
        if left in frame.columns or right in frame.columns:
            frame[name] = frame.get(left, pd.Series(default, index=frame.index)).combine_first(frame.get(right, pd.Series(default, index=frame.index)))
        else:
            frame[name] = default

    for column in ["target_weight_pct", "weight_delta_pct", "order_amount", "order_action"]:
        _coalesce(column, "none" if column == "order_action" else 0.0)

    frame["target_weight_pct"] = pd.to_numeric(frame["target_weight_pct"], errors="coerce").fillna(0.0)
    frame["weight_delta_pct"] = pd.to_numeric(frame["weight_delta_pct"], errors="coerce").fillna(0.0)
    frame["order_amount"] = pd.to_numeric(frame["order_amount"], errors="coerce").fillna(0.0)
    frame["order_action"] = frame["order_action"].fillna("none")

    decisions = frame.apply(_decision, axis=1, result_type="expand")
    frame["decision"] = decisions[0]
    frame["decision_reason"] = decisions[1]
    keep = [
        "ticker",
        "name",
        "asset_scope",
        "current_weight_pct",
        "target_weight_pct",
        "weight_delta_pct",
        "order_amount",
        "pnl_pct",
        "trend_status",
        "return_20d_pct",
        "drawdown_60d_pct",
        "volatility_20d_pct",
        "decision",
        "decision_reason",
    ]
    frame = frame[keep].sort_values(["decision", "current_weight_pct"], ascending=[True, False])

    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.today().strftime("%Y%m%d")
    csv_path = output_dir / f"portfolio_decision_sheet_{as_of}.csv"
    md_path = output_dir / f"portfolio_decision_sheet_{as_of}.md"
    png_path = output_dir / f"portfolio_decision_sheet_{as_of}.png"
    frame.to_csv(csv_path, index=False)
    render_decision_sheet_png(frame, png_path)
    model_name = str(load_config(config_path).get("training", {}).get("model_name", "configured model"))
    _write_markdown(md_path, frame, position_path, rebalance_path, model_name)
    return {"csv": csv_path, "markdown": md_path, "png": png_path}


def _write_markdown(path: Path, frame: pd.DataFrame, position_path: Path, rebalance_path: Path, model_name: str) -> None:
    lines = [
        "# Portfolio Decision Sheet",
        "",
        f"- positions: `{position_path}`",
        f"- rebalance: `{rebalance_path}`",
        f"- png: `{path.with_suffix('.png')}`",
        f"- holdings: `{len(frame)}`",
        "",
        "## Action Summary",
        "",
        _markdown_table(frame["decision"].value_counts().rename_axis("decision").reset_index(name="count")),
        "",
        "## Decisions",
        "",
        _markdown_table(frame),
        "",
        "## Notes",
        "",
        "- Decisions are rule-based labels, not automatic orders.",
        f"- `{model_name}` target applies only to its configured model universe.",
        "- US/global and ETF assets are evaluated by trend/risk rules only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--position-analysis", default=None)
    parser.add_argument("--rebalance-orders", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in build_decision_sheet(args.config, args.position_analysis, args.rebalance_orders, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
