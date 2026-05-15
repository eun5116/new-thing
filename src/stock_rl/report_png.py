from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stock_rl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TARGET_COLORS = {
    100.0: "#2f6db3",
    88.0: "#f28e2b",
    70.0: "#d65f5f",
}

_SCHEME = {
    "model_universe": "#2f6db3",
    "krx_stock_outside_model": "#59a14f",
    "krx_etf_or_unmapped": "#f28e2b",
    "us_or_global": "#d65f5f",
}

_DECISION_COLORS = {
    "trim_candidate": "#d65f5f",
    "trim_to_allocator": "#f28e2b",
    "keep_watch": "#59a14f",
    "speculative_watch": "#b07aa1",
    "manual_review": "#8c8c8c",
    "watch_or_trim": "#e15759",
    "add_candidate": "#2f6db3",
    "keep": "#4e79a7",
    "watch": "#76b7b2",
}


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _summary_text(ax: plt.Axes, lines: list[str]) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def render_report_dashboard_png(image_paths: dict[str, str | Path], path: str | Path, title: str = "Report Dashboard") -> Path:
    ordered_keys = [
        "trading_sheet",
        "target_changes",
        "rebalance_orders",
        "current_position_analysis",
        "portfolio_decision_sheet",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(18, 20), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")

    for ax, key in zip(axes.flat, ordered_keys + ["summary"]):
        ax.axis("off")
        if key == "summary":
            lines = ["Included reports:"]
            for name in ordered_keys:
                value = image_paths.get(name)
                lines.append(f"{name}: {value if value else 'missing'}")
            ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
            continue
        image_path = image_paths.get(key)
        if not image_path:
            ax.text(0.5, 0.5, f"{key}\nmissing", ha="center", va="center", fontsize=13)
            continue
        image_path = Path(image_path)
        if not image_path.exists():
            ax.text(0.5, 0.5, f"{key}\nmissing", ha="center", va="center", fontsize=13)
            continue
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=11)

    return _save_figure(fig, Path(path))


def render_trading_sheet_png(sheet: pd.DataFrame, path: str | Path, rule: str) -> Path:
    frame = sheet.copy()
    frame["target_pct"] = _safe_numeric(frame, "target_pct")
    frame["return_20d_pct"] = _safe_numeric(frame, "return_20d_pct")
    frame["drawdown_60d_pct"] = _safe_numeric(frame, "drawdown_60d_pct")

    counts = frame["target_pct"].value_counts().sort_index(ascending=False).reset_index()
    counts.columns = ["target_pct", "count"]
    top = frame.sort_values(["target_pct", "return_20d_pct", "trading_value_bil_krw"], ascending=[False, False, False]).head(12)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Trading Sheet Snapshot - {rule}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(counts["target_pct"].astype(str), counts["count"], color="#2f6db3")
    ax.set_title("Target Mix")
    ax.set_xlabel("Target %")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    top_plot = top.sort_values(["target_pct", "ticker"], ascending=[True, False])
    colors = top_plot["target_pct"].map(_TARGET_COLORS).fillna("#7f7f7f")
    ax.barh(top_plot["ticker"], top_plot["target_pct"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_title("Top Targets")
    ax.set_xlabel("Target %")
    for idx, value in enumerate(top_plot["target_pct"]):
        ax.text(min(value + 1, 98), idx, f"{value:.0f}%", va="center", fontsize=8)

    ax = axes[1, 0]
    scatter_colors = frame["target_pct"].map(_TARGET_COLORS).fillna("#7f7f7f")
    ax.scatter(frame["return_20d_pct"], frame["drawdown_60d_pct"], c=scatter_colors, s=34, alpha=0.85)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.axhline(0, color="#cccccc", lw=1)
    ax.set_title("Return vs Drawdown")
    ax.set_xlabel("20D Return %")
    ax.set_ylabel("60D Drawdown %")

    ax = axes[1, 1]
    _summary_text(
        ax,
        [
            f"rows: {len(frame)}",
            f"avg_target: {frame['target_pct'].mean():.1f}%",
            f"full_target: {int((frame['target_pct'] == 100.0).sum())}",
            f"capped: {int((frame['cap_reason'] == 'none').sum())}",
            f"stale_rows: {int((pd.to_datetime(frame['feature_date']) < pd.to_datetime(frame['as_of_date'])).sum()) if 'feature_date' in frame.columns and 'as_of_date' in frame.columns else 0}",
        ],
    )
    return _save_figure(fig, Path(path))


def render_target_changes_png(frame: pd.DataFrame, path: str | Path, rule: str) -> Path:
    data = frame.copy()
    data["target_pct"] = _safe_numeric(data, "target_pct")
    data["previous_target_pct"] = _safe_numeric(data, "previous_target_pct")
    data["target_delta_pct"] = _safe_numeric(data, "target_delta_pct")
    counts = data["rebalance_action"].value_counts().reindex(["increase", "reduce", "hold"], fill_value=0).reset_index()
    counts.columns = ["action", "count"]

    increases = data[data["rebalance_action"] == "increase"].sort_values("target_delta_pct", ascending=False).head(8)
    decreases = data[data["rebalance_action"] == "reduce"].sort_values("target_delta_pct").head(8)
    movers = pd.concat([increases, decreases], ignore_index=True).sort_values("target_delta_pct")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Target Changes Snapshot - {rule}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(counts["action"], counts["count"], color=["#2f6db3", "#d65f5f", "#8c8c8c"])
    ax.set_title("Rebalance Actions")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    if not movers.empty:
        colors = movers["target_delta_pct"].map(lambda value: "#2f6db3" if value > 0 else "#d65f5f")
        ax.barh(movers["ticker"], movers["target_delta_pct"], color=colors)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.set_title("Largest Movers")
    ax.set_xlabel("Target Delta (pct points)")

    ax = axes[1, 0]
    colors = data["rebalance_action"].map({"increase": "#2f6db3", "reduce": "#d65f5f", "hold": "#8c8c8c"}).fillna("#8c8c8c")
    ax.scatter(data["previous_target_pct"], data["target_pct"], c=colors, s=34, alpha=0.85)
    max_target = max(float(data["previous_target_pct"].max()), float(data["target_pct"].max()), 100.0)
    ax.plot([0, max_target], [0, max_target], color="#cccccc", lw=1, ls="--")
    ax.set_title("Previous vs Current")
    ax.set_xlabel("Previous Target %")
    ax.set_ylabel("Current Target %")

    ax = axes[1, 1]
    _summary_text(
        ax,
        [
            f"rows: {len(data)}",
            f"increase: {int((data['rebalance_action'] == 'increase').sum())}",
            f"reduce: {int((data['rebalance_action'] == 'reduce').sum())}",
            f"hold: {int((data['rebalance_action'] == 'hold').sum())}",
        ],
    )
    return _save_figure(fig, Path(path))


def render_rebalance_orders_png(frame: pd.DataFrame, path: str | Path, rule: str) -> Path:
    data = frame.copy()
    data["current_weight_pct"] = _safe_numeric(data, "current_weight_pct")
    data["target_weight_pct"] = _safe_numeric(data, "target_weight_pct")
    data["weight_delta_pct"] = _safe_numeric(data, "weight_delta_pct")
    data["order_amount"] = _safe_numeric(data, "order_amount")
    data["order_action"] = data["order_action"].astype(str)

    movers = data.reindex(data["weight_delta_pct"].abs().sort_values(ascending=False).head(10).index)
    order_view = data.reindex(data["order_amount"].abs().sort_values(ascending=False).head(12).index)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Rebalance Orders Snapshot - {rule}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    counts = data["order_action"].value_counts().reindex(["buy", "sell", "hold"], fill_value=0)
    ax.bar(counts.index, counts.values, color=["#2f6db3", "#d65f5f", "#8c8c8c"])
    ax.set_title("Order Actions")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    if not movers.empty:
        y = np.arange(len(movers))
        ax.barh(y - 0.18, movers["current_weight_pct"], height=0.35, color="#aab7c4", label="Current")
        ax.barh(y + 0.18, movers["target_weight_pct"], height=0.35, color="#2f6db3", alpha=0.75, label="Target")
        ax.set_yticks(y)
        ax.set_yticklabels(movers["ticker"])
        ax.legend(loc="lower right")
    ax.set_title("Current vs Target Weights")
    ax.set_xlabel("Weight %")

    ax = axes[1, 0]
    if not order_view.empty:
        colors = order_view["order_amount"].map(lambda value: "#2f6db3" if value > 0 else "#d65f5f")
        ax.barh(order_view["ticker"], order_view["order_amount"], color=colors)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.set_title("Largest Orders")
    ax.set_xlabel("Order Amount")

    ax = axes[1, 1]
    _summary_text(
        ax,
        [
            f"portfolio_value: {float(data['market_value'].sum()):,.0f}",
            f"buys: {int((data['order_action'] == 'buy').sum())}",
            f"sells: {int((data['order_action'] == 'sell').sum())}",
            f"holds: {int((data['order_action'] == 'hold').sum())}",
        ],
    )
    return _save_figure(fig, Path(path))


def render_position_analysis_png(frame: pd.DataFrame, path: str | Path) -> Path:
    data = frame.copy()
    data["current_weight_pct"] = _safe_numeric(data, "current_weight_pct")
    data["pnl_pct"] = _safe_numeric(data, "pnl_pct")
    data["market_value"] = _safe_numeric(data, "market_value")

    top = data.sort_values("current_weight_pct", ascending=False).head(10)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Current Position Snapshot", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    weight_sum = data.groupby("asset_scope")["current_weight_pct"].sum().reindex(_SCHEME.keys(), fill_value=0.0)
    ax.bar(weight_sum.index, weight_sum.values, color=[_SCHEME[key] for key in weight_sum.index])
    ax.set_title("Weight by Scope")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Weight %")

    ax = axes[0, 1]
    colors = data["asset_scope"].map(_SCHEME).fillna("#8c8c8c")
    sizes = data["market_value"] / max(float(data["market_value"].max()), 1.0) * 800 + 20
    ax.scatter(data["current_weight_pct"], data["pnl_pct"], c=colors, s=sizes, alpha=0.85)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.axhline(0, color="#cccccc", lw=1)
    ax.set_title("Weight vs PnL")
    ax.set_xlabel("Current Weight %")
    ax.set_ylabel("PnL %")

    ax = axes[1, 0]
    if not top.empty:
        colors = top["asset_scope"].map(_SCHEME).fillna("#8c8c8c")
        ax.barh(top["ticker"], top["current_weight_pct"], color=colors)
    ax.set_title("Largest Holdings")
    ax.set_xlabel("Weight %")

    ax = axes[1, 1]
    _summary_text(
        ax,
        [
            f"holdings: {len(data)}",
            f"model_universe: {int((data['asset_scope'] == 'model_universe').sum())}",
            f"outside_model: {int((data['asset_scope'] != 'model_universe').sum())}",
            f"us_or_global: {int((data['asset_scope'] == 'us_or_global').sum())}",
        ],
    )
    return _save_figure(fig, Path(path))


def render_decision_sheet_png(frame: pd.DataFrame, path: str | Path) -> Path:
    data = frame.copy()
    data["current_weight_pct"] = _safe_numeric(data, "current_weight_pct")
    data["pnl_pct"] = _safe_numeric(data, "pnl_pct")
    counts = data["decision"].value_counts().sort_values(ascending=False).reset_index()
    counts.columns = ["decision", "count"]
    top = data.sort_values("current_weight_pct", ascending=False).head(10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Portfolio Decision Snapshot", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(counts["decision"], counts["count"], color=[_DECISION_COLORS.get(name, "#8c8c8c") for name in counts["decision"]])
    ax.set_title("Decision Counts")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    if not top.empty:
        colors = top["decision"].map(_DECISION_COLORS).fillna("#8c8c8c")
        ax.barh(top["ticker"], top["current_weight_pct"], color=colors)
    ax.set_title("Largest Holdings by Decision")
    ax.set_xlabel("Weight %")

    ax = axes[1, 0]
    colors = data["decision"].map(_DECISION_COLORS).fillna("#8c8c8c")
    ax.scatter(data["current_weight_pct"], data["pnl_pct"], c=colors, s=34, alpha=0.85)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.axhline(0, color="#cccccc", lw=1)
    ax.set_title("Weight vs PnL")
    ax.set_xlabel("Current Weight %")
    ax.set_ylabel("PnL %")

    ax = axes[1, 1]
    _summary_text(
        ax,
        [
            f"holdings: {len(data)}",
            f"trim_candidate: {int((data['decision'] == 'trim_candidate').sum())}",
            f"keep_watch: {int((data['decision'] == 'keep_watch').sum())}",
            f"manual_review: {int((data['decision'] == 'manual_review').sum())}",
        ],
    )
    return _save_figure(fig, Path(path))
