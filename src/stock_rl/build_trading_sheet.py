from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_rl.config import load_config, project_path
from stock_rl.report_png import render_trading_sheet_png
from stock_rl.trading_env import normalize_ticker


KEEP_COLUMNS = [
    "as_of_date",
    "feature_date",
    "feature_lag_days",
    "ticker",
    "name",
    "market",
    "close",
    "change_pct",
    "target_pct",
    "raw_target_pct",
    "cap_pct",
    "cap_reason",
    "return_20d_pct",
    "return_60d_pct",
    "relative_strength_20d_pct",
    "drawdown_60d_pct",
    "market_return_60d_pct",
    "trading_value_bil_krw",
    "market_cap_bil_krw",
]


def _latest_target_path(config_path: str, rule: str) -> Path:
    reports_dir = project_path(config_path, "reports")
    paths = sorted(reports_dir.glob(f"current_targets_*_{rule}.csv"))
    if not paths:
        raise FileNotFoundError(f"no current target files found for rule: {rule}")
    return paths[-1]


def _load_reference(config_path: str) -> pd.DataFrame:
    reference_dir = project_path(config_path, load_config(config_path)["project"]["data_dir"], "raw", "reference")
    frames = []
    for path in sorted(reference_dir.glob("*_issue_base.parquet")):
        frame = pd.read_parquet(path)
        keep = [column for column in ["ticker", "abbrv", "name", "market"] if column in frame.columns]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "market"])
    reference = pd.concat(frames, ignore_index=True)
    reference["ticker"] = reference["ticker"].astype(str).map(normalize_ticker)
    if "abbrv" in reference.columns:
        fallback_name = reference["name"] if "name" in reference.columns else ""
        reference["name"] = reference["abbrv"].fillna(fallback_name)
    return reference[["ticker", "name", "market"]].drop_duplicates("ticker")


def _load_latest_features(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    features_path = project_path(config_path, config["project"]["data_dir"], "processed", "daily_features.parquet")
    features = pd.read_parquet(features_path)
    features["ticker"] = features["ticker"].astype(str).map(normalize_ticker)
    features["date"] = pd.to_datetime(features["date"])
    latest = features.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    keep = [
        "date",
        "ticker",
        "market",
        "close",
        "change_pct",
        "trading_value",
        "market_cap",
    ]
    return latest[[column for column in keep if column in latest.columns]]


def _pct(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 100.0).round(2)


def _first_available(frame: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    result = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            result = result.mask(result.isna() | (result == ""), frame[column])
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
    text = frame.copy()
    text = text.fillna("")
    headers = list(text.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in text.iterrows():
        values = [str(int(row[column])) if column == "count" and str(row[column]) else str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_trading_sheet(
    config_path: str,
    rule: str = "strong_trend_full_else070",
    target_path: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Path]:
    resolved_target_path = Path(target_path) if target_path else _latest_target_path(config_path, rule)
    targets = pd.read_csv(resolved_target_path, dtype={"ticker": str})
    targets["ticker"] = targets["ticker"].map(normalize_ticker)
    targets["as_of_date"] = pd.to_datetime(targets["as_of_date"])
    targets["feature_date"] = pd.to_datetime(targets["feature_date"])

    latest_features = _load_latest_features(config_path)
    reference = _load_reference(config_path)
    sheet = targets.merge(latest_features, how="left", on="ticker", suffixes=("", "_latest"))
    sheet = sheet.merge(reference, how="left", on="ticker", suffixes=("", "_reference"))
    sheet["market"] = _first_available(sheet, ["market_reference", "market_latest", "market"])
    sheet["name"] = _first_available(sheet, ["name"], default="")
    sheet["feature_lag_days"] = (sheet["as_of_date"] - sheet["feature_date"]).dt.days.astype(int)

    sheet["target_pct"] = _pct(sheet["target_ratio"])
    sheet["raw_target_pct"] = _pct(sheet["raw_target_ratio"])
    sheet["cap_pct"] = _pct(sheet["cap"])
    sheet["change_pct"] = pd.to_numeric(sheet.get("change_pct", 0.0), errors="coerce").fillna(0.0)
    sheet["return_20d_pct"] = _pct(sheet["return_20d"])
    sheet["return_60d_pct"] = _pct(sheet["return_60d"])
    sheet["relative_strength_20d_pct"] = _pct(sheet["relative_strength_20d"])
    sheet["drawdown_60d_pct"] = _pct(sheet["drawdown_60d"])
    sheet["market_return_60d_pct"] = _pct(sheet["market_return_60d"])
    sheet["trading_value_bil_krw"] = (sheet.get("trading_value", pd.Series(0.0, index=sheet.index)).astype(float) / 1e9).round(1)
    sheet["market_cap_bil_krw"] = (sheet.get("market_cap", pd.Series(0.0, index=sheet.index)).astype(float) / 1e9).round(1)
    sheet["change_pct"] = sheet["change_pct"].round(2)
    sheet["close"] = sheet["close"].round(0).astype("Int64")

    sheet = sheet[KEEP_COLUMNS].sort_values(
        ["target_pct", "return_20d_pct", "trading_value_bil_krw"],
        ascending=[False, False, False],
    )

    output_dir = Path(out_dir) if out_dir else project_path(config_path, "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = targets["as_of_date"].max().strftime("%Y%m%d")
    model_name = str(targets["model_name"].dropna().iloc[0]) if "model_name" in targets.columns and not targets["model_name"].dropna().empty else "trained PPO policy"
    csv_path = output_dir / f"trading_sheet_{as_of}_{rule}.csv"
    md_path = output_dir / f"trading_sheet_{as_of}_{rule}.md"
    png_path = output_dir / f"trading_sheet_{as_of}_{rule}.png"
    sheet.to_csv(csv_path, index=False)
    render_trading_sheet_png(sheet, png_path, rule)
    _write_markdown(md_path, sheet, rule, resolved_target_path, model_name)
    return {"csv": csv_path, "markdown": md_path, "png": png_path}


def _write_markdown(path: Path, sheet: pd.DataFrame, rule: str, target_path: Path, model_name: str) -> None:
    as_of = sheet["as_of_date"].max().date().isoformat()
    lines = [
        f"# Trading Sheet - {as_of}",
        "",
        f"- rule: `{rule}`",
        f"- source: `{target_path}`",
        f"- png: `{path.with_suffix('.png')}`",
        f"- tickers: `{len(sheet)}`",
        f"- avg target: `{sheet['target_pct'].mean():.1f}%`",
        f"- full target count: `{int((sheet['target_pct'] == 100.0).sum())}`",
        f"- capped count: `{int((sheet['cap_reason'] == 'none').sum())}`",
        "",
        "## How To Read This",
        "",
        f"`target_pct` is the exposure allowed by `{model_name}` plus the regime cap overlay. It is not a probability that the stock will rise.",
        "",
        "- `100%`: PPO selected max exposure and the regime overlay allowed full risk.",
        "- `88%`: PPO selected a high but not max target, or the raw target was already below full exposure.",
        "- `70%`: the regime overlay capped exposure because the row did not satisfy the strong-trend condition.",
        "",
        "Use this sheet as a position-sizing/ranking input, not as a standalone return forecast.",
        "",
        "## Target Summary",
        "",
        _markdown_table(
            sheet["target_pct"]
            .value_counts()
            .sort_index(ascending=False)
            .rename_axis("target_pct")
            .reset_index(name="count")
            .astype({"count": int})
        ),
        "",
        "## Top Targets",
        "",
        _markdown_table(
            sheet.head(15)[
                [
                    "ticker",
                    "name",
                    "market",
                    "close",
                    "change_pct",
                    "target_pct",
                    "cap_reason",
                    "return_20d_pct",
                    "return_60d_pct",
                    "drawdown_60d_pct",
                ]
            ]
        ),
        "",
        "## Capped Names",
        "",
        _markdown_table(
            sheet[sheet["cap_reason"] == "none"].head(20)[
                [
                    "ticker",
                    "name",
                    "market",
                    "close",
                    "target_pct",
                    "return_20d_pct",
                    "return_60d_pct",
                    "relative_strength_20d_pct",
                    "drawdown_60d_pct",
                ]
            ]
        ),
        "",
        "## Stale Feature Rows",
        "",
    ]
    stale = sheet[sheet["feature_lag_days"] > 0]
    if stale.empty:
        lines.append("None.")
    else:
        lines.append(_markdown_table(stale[["ticker", "name", "feature_date", "feature_lag_days", "target_pct"]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--rule", default="strong_trend_full_else070")
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name, path in build_trading_sheet(args.config, args.rule, args.target_path, args.out_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
