from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.backtest_portfolio_allocator import _cap_and_normalize
from stock_rl.build_trading_sheet import _load_reference, _markdown_table
from stock_rl.config import project_path
from stock_rl.trading_env import normalize_ticker


KEEP_COLUMNS = [
    "ticker",
    "name",
    "asset_scope",
    "market_value",
    "current_weight_pct",
    "target_weight_pct",
    "weight_delta_pct",
    "order_amount",
    "order_action",
    "target_ratio_pct",
    "cap_reason",
    "return_20d_pct",
    "drawdown_60d_pct",
]


def _latest_target_path(config_path: str, rule: str) -> Path:
    paths = sorted(project_path(config_path, "reports").glob(f"current_targets_*_{rule}.csv"))
    if not paths:
        raise FileNotFoundError(f"no current target files found for rule: {rule}")
    return paths[-1]


def _load_positions(path: str | Path) -> pd.DataFrame:
    positions = pd.read_csv(path, dtype={"ticker": str})
    required = {"ticker", "market_value"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"positions CSV missing columns: {sorted(missing)}")
    positions["ticker"] = positions["ticker"].map(normalize_ticker)
    positions["market_value"] = pd.to_numeric(positions["market_value"], errors="coerce").fillna(0.0)
    if "name" not in positions.columns:
        positions["name"] = ""
    return positions


def _target_weights_from_current_targets(
    targets: pd.DataFrame,
    top_n: int,
    gross_cap: float,
    max_weight: float,
) -> pd.Series:
    ranked = targets.sort_values(["target_ratio", "return_20d", "ticker"], ascending=[False, False, True]).head(top_n)
    scores = pd.Series(ranked["target_ratio"].astype(float).to_numpy(), index=ranked["ticker"])
    return _cap_and_normalize(scores, gross_cap, max_weight)


def build_rebalance_orders(
    config_path: str,
    positions_path: str,
    rule: str = "strong_trend_full_else070",
    target_path: str | None = None,
    top_n: int = 12,
    gross_cap: float = 0.90,
    max_weight: float = 0.20,
    min_order_amount: float = 5000.0,
    cash: float = 0.0,
    out_dir: str | None = None,
) -> dict[str, Path]:
    resolved_target_path = Path(target_path) if target_path else _latest_target_path(config_path, rule)
    positions = _load_positions(positions_path)
    targets = pd.read_csv(resolved_target_path, dtype={"ticker": str})
    targets["ticker"] = targets["ticker"].map(normalize_ticker)
    target_weights = _target_weights_from_current_targets(targets, top_n, gross_cap, max_weight)

    total_value = float(positions["market_value"].sum() + cash)
    if total_value <= 0:
        raise ValueError("portfolio value must be positive")
    current_weights = positions.groupby("ticker")["market_value"].sum() / total_value

    universe = set(targets["ticker"])
    all_tickers = sorted(set(current_weights.index).union(set(target_weights.index)))
    reference = _load_reference(config_path)
    target_context = targets.set_index("ticker")
    rows = []
    for ticker in all_tickers:
        current_weight = float(current_weights.get(ticker, 0.0))
        target_weight = float(target_weights.get(ticker, 0.0))
        weight_delta = target_weight - current_weight
        order_amount = weight_delta * total_value
        held_row = positions[positions["ticker"] == ticker]
        target_row = target_context.loc[ticker] if ticker in target_context.index else None
        ref_row = reference[reference["ticker"] == ticker]
        name = ""
        if not held_row.empty:
            name = str(held_row.iloc[0].get("name", ""))
        if not name and not ref_row.empty:
            name = str(ref_row.iloc[0].get("name", ""))
        if abs(order_amount) < min_order_amount:
            action = "hold"
        elif order_amount > 0:
            action = "buy"
        else:
            action = "sell"
        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "asset_scope": "model_universe" if ticker in universe else "out_of_universe",
                "market_value": float(held_row["market_value"].sum()) if not held_row.empty else 0.0,
                "current_weight_pct": round(current_weight * 100.0, 2),
                "target_weight_pct": round(target_weight * 100.0, 2),
                "weight_delta_pct": round(weight_delta * 100.0, 2),
                "order_amount": round(order_amount, 0),
                "order_action": action,
                "target_ratio_pct": round(float(target_row["target_ratio"]) * 100.0, 2) if target_row is not None else 0.0,
                "cap_reason": str(target_row["cap_reason"]) if target_row is not None else "not_in_model_universe",
                "return_20d_pct": round(float(target_row["return_20d"]) * 100.0, 2) if target_row is not None else 0.0,
                "drawdown_60d_pct": round(float(target_row["drawdown_60d"]) * 100.0, 2) if target_row is not None else 0.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame[KEEP_COLUMNS].sort_values(["order_action", "order_amount"], ascending=[True, False])
    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(targets["as_of_date"].max()).replace("-", "")
    csv_path = output_dir / f"rebalance_orders_{as_of}_{rule}.csv"
    md_path = output_dir / f"rebalance_orders_{as_of}_{rule}.md"
    frame.to_csv(csv_path, index=False)
    _write_markdown(md_path, frame, resolved_target_path, positions_path, total_value, cash, top_n, gross_cap, max_weight)
    return {"csv": csv_path, "markdown": md_path}


def _write_markdown(
    path: Path,
    frame: pd.DataFrame,
    target_path: Path,
    positions_path: str | Path,
    total_value: float,
    cash: float,
    top_n: int,
    gross_cap: float,
    max_weight: float,
) -> None:
    buys = frame[frame["order_action"] == "buy"]
    sells = frame[frame["order_action"] == "sell"].sort_values("order_amount")
    out_of_universe = frame[(frame["asset_scope"] == "out_of_universe") & (frame["market_value"] > 0)]
    lines = [
        "# Rebalance Orders",
        "",
        f"- target_source: `{target_path}`",
        f"- positions_source: `{positions_path}`",
        f"- portfolio_value: `{total_value:,.0f}`",
        f"- cash_assumed: `{cash:,.0f}`",
        f"- top_n: `{top_n}`",
        f"- gross_cap: `{gross_cap * 100.0:.1f}%`",
        f"- max_weight: `{max_weight * 100.0:.1f}%`",
        "",
        "## Buy",
        "",
        _markdown_table(buys.head(20)[KEEP_COLUMNS]),
        "",
        "## Sell",
        "",
        _markdown_table(sells.head(20)[KEEP_COLUMNS]),
        "",
        "## Out Of Universe Holdings",
        "",
        _markdown_table(out_of_universe[["ticker", "name", "market_value", "current_weight_pct", "order_action", "order_amount"]]),
        "",
        "## Note",
        "",
        "This table compares current holdings against the model universe allocator. Assets outside the 48-stock KRX universe receive a target weight of 0 by construction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--positions", default="data_krx/raw/positions/current_positions.csv")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--gross-cap", type=float, default=0.90)
    parser.add_argument("--max-weight", type=float, default=0.20)
    parser.add_argument("--min-order-amount", type=float, default=5000.0)
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in build_rebalance_orders(
        args.config,
        args.positions,
        args.rule,
        args.target_path,
        args.top_n,
        args.gross_cap,
        args.max_weight,
        args.min_order_amount,
        args.cash,
        args.out_dir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
