from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from stock_rl.build_trading_sheet import _markdown_table
from stock_rl.config import project_path
from stock_rl.trading_env import normalize_ticker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data_weekly_market" / "history"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "weekly_market"
DEFAULT_RULE = "strong_trend_full_else070"

WATCH_COLUMNS = [
    "report_date",
    "market",
    "symbol",
    "name",
    "end_price",
    "latest_weekly_change_pct",
    "four_week_return_pct",
    "market_four_week_return_pct",
    "relative_gap_pct",
    "discount_from_8w_high_pct",
    "momentum_triggered",
    "target_pct",
    "watch_score",
    "bucket",
    "action_hint",
]


def _read_history(path: Path, market: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"symbol": str})
    if frame.empty:
        return frame
    frame["market"] = market
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], errors="coerce")
    frame["end_price"] = pd.to_numeric(frame["end_price"], errors="coerce")
    frame["change_pct"] = pd.to_numeric(frame["change_pct"], errors="coerce")
    frame = frame.dropna(subset=["report_date", "symbol", "end_price"])
    if market == "KOSPI":
        frame["symbol"] = frame["symbol"].map(normalize_ticker)
    else:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame = frame.sort_values(["report_date", "market", "symbol", "captured_at"])
    return frame.drop_duplicates(["report_date", "market", "symbol"], keep="last")


def _load_market_history(history_dir: Path) -> pd.DataFrame:
    frames = [
        _read_history(history_dir / "kospi_top20_history.csv", "KOSPI"),
        _read_history(history_dir / "sp500_top20_history.csv", "SP500"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_latest_targets(reports_dir: Path, rule: str) -> pd.DataFrame:
    paths = sorted(reports_dir.glob(f"current_targets_*_{rule}.csv"))
    if not paths:
        return pd.DataFrame(columns=["symbol", "target_pct"])
    targets = pd.read_csv(paths[-1], dtype={"ticker": str})
    if targets.empty or "ticker" not in targets.columns or "target_ratio" not in targets.columns:
        return pd.DataFrame(columns=["symbol", "target_pct"])
    targets["symbol"] = targets["ticker"].map(normalize_ticker)
    targets["target_pct"] = (pd.to_numeric(targets["target_ratio"], errors="coerce") * 100.0).round(1)
    return targets[["symbol", "target_pct"]].drop_duplicates("symbol")


def _load_latest_alerts(history_dir: Path) -> pd.DataFrame:
    path = history_dir / "momentum_alert_history.csv"
    if not path.exists():
        return pd.DataFrame(columns=["market", "symbol", "momentum_triggered"])
    alerts = pd.read_csv(path, dtype={"symbol": str})
    if alerts.empty:
        return pd.DataFrame(columns=["market", "symbol", "momentum_triggered"])
    alerts["report_date"] = pd.to_datetime(alerts["report_date"], errors="coerce")
    latest_date = alerts["report_date"].dropna().max()
    if pd.isna(latest_date):
        return pd.DataFrame(columns=["market", "symbol", "momentum_triggered"])
    latest = alerts[alerts["report_date"] == latest_date].copy()
    latest["market"] = latest["market"].astype(str).str.upper().str.strip()
    latest["symbol"] = latest["symbol"].astype(str).str.upper().str.strip()
    latest.loc[latest["market"] == "KOSPI", "symbol"] = latest.loc[latest["market"] == "KOSPI", "symbol"].map(
        normalize_ticker
    )
    latest["momentum_triggered"] = latest["triggered"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        latest.groupby(["market", "symbol"], as_index=False)["momentum_triggered"]
        .max()
        .reset_index(drop=True)
    )


def _pct_return(first: float, last: float) -> float:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def _bucket(row: pd.Series) -> tuple[str, str]:
    discount = float(row["discount_from_8w_high_pct"])
    relative_gap = float(row["relative_gap_pct"])
    latest = float(row["latest_weekly_change_pct"])
    momentum = bool(row["momentum_triggered"])
    target_pct = float(row["target_pct"]) if pd.notna(row["target_pct"]) else 0.0

    if discount <= -12.0 and latest <= -5.0:
        return "falling_knife", "관찰만. 반등 확인 전 분할매수 보류"
    if discount <= -5.0 and relative_gap <= -3.0 and (latest >= 0.0 or momentum):
        return "early_value_recovery", "소액 분할 후보. 다음 주에도 하락폭 축소 확인"
    if discount <= -3.0 and relative_gap <= -2.0 and target_pct >= 70.0:
        return "model_supported_laggard", "기존 모델 target과 함께 보는 후보"
    if momentum and relative_gap <= 0.0:
        return "momentum_turnaround", "저평가보다는 회복 모멘텀 후보"
    return "watch_only", "가격 매력은 있으나 확인 신호 부족"


def _score(row: pd.Series) -> float:
    discount = abs(min(float(row["discount_from_8w_high_pct"]), 0.0))
    relative_gap = abs(min(float(row["relative_gap_pct"]), 0.0))
    latest = float(row["latest_weekly_change_pct"])
    target_pct = float(row["target_pct"]) if pd.notna(row["target_pct"]) else 0.0
    score = min(discount, 25.0) / 25.0 * 35.0
    score += min(relative_gap, 20.0) / 20.0 * 25.0
    if latest >= 0:
        score += min(latest, 10.0) / 10.0 * 20.0
    elif latest > -2.0:
        score += 5.0
    if bool(row["momentum_triggered"]):
        score += 10.0
    if target_pct >= 70.0:
        score += 10.0
    return round(score, 1)


def build_weekly_value_watchlist(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    reports_dir: str | Path = PROJECT_ROOT / "reports",
    rule: str = DEFAULT_RULE,
    out_dir: str | Path = DEFAULT_REPORT_DIR,
    lookback_reports: int = 8,
    limit: int = 25,
) -> dict[str, Path]:
    history_path = Path(history_dir)
    history = _load_market_history(history_path)
    if history.empty:
        raise FileNotFoundError(f"weekly market history is empty: {history_path}")

    latest_date = history["report_date"].max()
    recent_dates = sorted(history["report_date"].dropna().unique())[-lookback_reports:]
    recent = history[history["report_date"].isin(recent_dates)].copy()

    market_returns = []
    rows = []
    for market, market_frame in recent.groupby("market"):
        pivot = market_frame.pivot_table(index="report_date", columns="symbol", values="end_price", aggfunc="last").sort_index()
        if pivot.empty:
            continue
        market_start = pivot.iloc[max(len(pivot) - 4, 0)].dropna()
        market_end = pivot.iloc[-1].dropna()
        common = market_start.index.intersection(market_end.index)
        market_four_week_return = (
            ((market_end[common] / market_start[common] - 1.0) * 100.0).mean() if len(common) else 0.0
        )
        market_returns.append({"market": market, "market_four_week_return_pct": market_four_week_return})

        latest_rows = market_frame[market_frame["report_date"] == latest_date].copy()
        for _, latest in latest_rows.iterrows():
            symbol = latest["symbol"]
            series = pivot[symbol].dropna() if symbol in pivot.columns else pd.Series(dtype=float)
            if series.empty:
                continue
            four_week_start = series.iloc[max(len(series) - 4, 0)]
            four_week_return = _pct_return(float(four_week_start), float(series.iloc[-1]))
            high_8w = float(series.max())
            discount = _pct_return(high_8w, float(series.iloc[-1]))
            rows.append(
                {
                    "report_date": latest_date.date().isoformat(),
                    "market": market,
                    "symbol": symbol,
                    "name": latest.get("name", symbol),
                    "end_price": round(float(latest["end_price"]), 2),
                    "latest_weekly_change_pct": round(float(latest.get("change_pct", 0.0)), 2),
                    "four_week_return_pct": round(four_week_return, 2),
                    "market_four_week_return_pct": round(float(market_four_week_return), 2),
                    "relative_gap_pct": round(four_week_return - float(market_four_week_return), 2),
                    "discount_from_8w_high_pct": round(discount, 2),
                }
            )

    watch = pd.DataFrame(rows)
    if watch.empty:
        raise RuntimeError("no watchlist rows could be built from weekly market history")

    alerts = _load_latest_alerts(history_path)
    targets = _load_latest_targets(Path(reports_dir), rule)
    watch = watch.merge(alerts, how="left", on=["market", "symbol"])
    watch = watch.merge(targets, how="left", on="symbol")
    watch["momentum_triggered"] = watch["momentum_triggered"].fillna(False).astype(bool)
    watch["target_pct"] = watch["target_pct"].fillna(0.0)
    buckets = watch.apply(_bucket, axis=1)
    watch["bucket"] = [bucket for bucket, _ in buckets]
    watch["action_hint"] = [hint for _, hint in buckets]
    watch["watch_score"] = watch.apply(_score, axis=1)
    watch = watch.sort_values(
        ["watch_score", "discount_from_8w_high_pct", "relative_gap_pct"],
        ascending=[False, True, True],
    ).head(limit)
    watch = watch[WATCH_COLUMNS]

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = latest_date.strftime("%Y%m%d")
    csv_path = output_dir / f"weekly_value_watchlist_{suffix}.csv"
    md_path = output_dir / f"weekly_value_watchlist_{suffix}.md"
    watch.to_csv(csv_path, index=False)
    _write_markdown(md_path, watch, csv_path, rule)
    return {"csv": csv_path, "markdown": md_path}


def _write_markdown(path: Path, watch: pd.DataFrame, csv_path: Path, rule: str) -> None:
    report_date = str(watch["report_date"].max())
    bucket_counts = watch["bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
    actionable = watch[watch["bucket"].isin(["early_value_recovery", "model_supported_laggard", "momentum_turnaround"])]
    falling = watch[watch["bucket"] == "falling_knife"]
    lines = [
        f"# Weekly Value Watchlist - {report_date}",
        "",
        f"- source: `data_weekly_market/history/*.csv`",
        f"- csv: `{csv_path}`",
        f"- target rule: `{rule}`",
        f"- candidates: `{len(watch)}`",
        "",
        "## 기준",
        "",
        "이 표는 PER/PBR 같은 재무 밸류에이션으로 만든 저평가 판정이 아니다. 현재 `weekly_market_report`가 보유한 가격/모멘텀 기록을 이용해 `최근 고점 대비 할인율`, `최근 4주 상대 부진`, `회복 모멘텀`, `KRX RL target`을 함께 본 선매수 감시 리스트다.",
        "",
        "## Bucket Summary",
        "",
        _markdown_table(bucket_counts),
        "",
        "## Accumulation Candidates",
        "",
        _markdown_table(
            actionable[
                [
                    "market",
                    "symbol",
                    "name",
                    "latest_weekly_change_pct",
                    "four_week_return_pct",
                    "relative_gap_pct",
                    "discount_from_8w_high_pct",
                    "momentum_triggered",
                    "target_pct",
                    "watch_score",
                    "bucket",
                    "action_hint",
                ]
            ].head(15)
        ),
        "",
        "## Falling Knife / Wait",
        "",
        _markdown_table(
            falling[
                [
                    "market",
                    "symbol",
                    "name",
                    "latest_weekly_change_pct",
                    "four_week_return_pct",
                    "relative_gap_pct",
                    "discount_from_8w_high_pct",
                    "watch_score",
                    "action_hint",
                ]
            ].head(15)
        ),
        "",
        "## Full Watchlist",
        "",
        _markdown_table(watch.head(25)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--rule", default=DEFAULT_RULE)
    parser.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--lookback-reports", type=int, default=8)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    result = build_weekly_value_watchlist(
        history_dir=args.history_dir,
        reports_dir=args.reports_dir,
        rule=args.rule,
        out_dir=args.out_dir,
        lookback_reports=args.lookback_reports,
        limit=args.limit,
    )
    for name, path in result.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
