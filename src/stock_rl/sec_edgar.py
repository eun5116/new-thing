from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_rl.config import load_config, project_path


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

DEFAULT_USER_AGENT = "stock-rl-project/0.1 jack@example.com"

FACT_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

POINT_IN_TIME_FACTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}


def _headers() -> dict[str, str]:
    return {"User-Agent": os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT)}


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, headers=_headers(), timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected SEC payload type: {type(payload).__name__}")
    return payload


def fetch_company_tickers(out_path: Path) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = _get_json(SEC_TICKERS_URL)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
    rows = list(payload.values())
    frame = pd.DataFrame(rows)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["cik_str"] = pd.to_numeric(frame["cik_str"], errors="coerce").astype("Int64")
    return frame


def _latest_fact(facts: dict[str, Any], tags: list[str], unit_preference: list[str]) -> dict[str, Any] | None:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        units = us_gaap.get(tag, {}).get("units", {})
        for unit in unit_preference:
            values = units.get(unit, [])
            if not values:
                continue
            frame = pd.DataFrame(values)
            if frame.empty or "val" not in frame.columns:
                continue
            if "filed" in frame.columns:
                frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
            if "end" in frame.columns:
                frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
            sort_cols = [column for column in ["filed", "end"] if column in frame.columns]
            if sort_cols:
                frame = frame.sort_values(sort_cols)
            latest = frame.dropna(subset=["val"]).tail(1)
            if latest.empty:
                continue
            row = latest.iloc[0]
            return {
                "tag": tag,
                "unit": unit,
                "value": float(row["val"]),
                "filed": row.get("filed", pd.NaT),
                "period_end": row.get("end", pd.NaT),
                "form": row.get("form", ""),
                "fy": row.get("fy", None),
                "fp": row.get("fp", ""),
            }
    return None


def collect_sec_companyfacts(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    tickers = [str(ticker).upper() for ticker in config["market"]["tickers"]]
    sec_dir = project_path(config_path, config["project"]["data_dir"], "raw", "sec")
    processed_dir = project_path(config_path, config["project"]["data_dir"], "processed")
    sec_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mapping = fetch_company_tickers(sec_dir / "company_tickers.json")
    mapping = mapping[mapping["ticker"].isin(tickers)]
    rows = []
    for _, company in mapping.iterrows():
        ticker = str(company["ticker"]).upper()
        cik = int(company["cik_str"])
        raw_path = sec_dir / f"companyfacts_{ticker}.json"
        try:
            payload = _get_json(SEC_COMPANYFACTS_URL.format(cik=cik))
        except requests.HTTPError as exc:
            print(f"SEC {ticker}: skipped ({exc})", flush=True)
            continue
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        row: dict[str, Any] = {
            "ticker": ticker,
            "cik": f"{cik:010d}",
            "entity_name": payload.get("entityName", company.get("title", "")),
        }
        for name, tags in FACT_TAGS.items():
            fact = _latest_fact(payload, tags, ["USD", "USD/shares", "shares"])
            if fact is None:
                row[f"{name}_value"] = None
                row[f"{name}_tag"] = ""
                row[f"{name}_filed"] = ""
                row[f"{name}_period_end"] = ""
                continue
            row[f"{name}_value"] = fact["value"]
            row[f"{name}_tag"] = fact["tag"]
            row[f"{name}_filed"] = _date_text(fact["filed"])
            row[f"{name}_period_end"] = _date_text(fact["period_end"])
            row[f"{name}_form"] = fact["form"]
        rows.append(row)

    snapshot = pd.DataFrame(rows).sort_values("ticker") if rows else pd.DataFrame(columns=["ticker", "cik"])
    snapshot_path = processed_dir / "sec_companyfacts_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)
    point_in_time = build_point_in_time_features(sec_dir, snapshot["ticker"].tolist() if not snapshot.empty else [])
    point_in_time_path = processed_dir / "sec_point_in_time_features.csv"
    point_in_time.to_csv(point_in_time_path, index=False)
    return {"snapshot": snapshot_path, "point_in_time": point_in_time_path}


def build_point_in_time_features(sec_dir: Path, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        raw_path = sec_dir / f"companyfacts_{ticker}.json"
        if not raw_path.exists():
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        facts = extract_point_in_time_facts(str(ticker), payload)
        if not facts.empty:
            frames.append(facts)
    if not frames:
        return pd.DataFrame(columns=["ticker", "filed_date", *POINT_IN_TIME_FACTS.keys()])
    combined = pd.concat(frames, ignore_index=True)
    combined["filed_date"] = pd.to_datetime(combined["filed_date"], errors="coerce")
    combined["period_end"] = pd.to_datetime(combined["period_end"], errors="coerce")
    combined = combined.dropna(subset=["filed_date"])
    combined = combined[combined["form"].astype(str).str.startswith(("10-Q", "10-K"))]
    combined = (
        combined.sort_values(["ticker", "filed_date", "period_end"])
        .drop_duplicates(["ticker", "filed_date"], keep="last")
        .reset_index(drop=True)
    )
    return add_fundamental_ratios(combined)


def extract_point_in_time_facts(ticker: str, payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, tags in POINT_IN_TIME_FACTS.items():
        fact_frame = _fact_history(payload, tags, ["USD", "USD/shares"])
        if fact_frame.empty:
            continue
        fact_frame = fact_frame[["filed_date", "period_end", "form", "value"]].copy()
        fact_frame["metric"] = name
        rows.append(fact_frame)
    if not rows:
        return pd.DataFrame()
    facts = pd.concat(rows, ignore_index=True)
    facts["ticker"] = ticker
    pivot = facts.pivot_table(
        index=["ticker", "filed_date", "period_end", "form"],
        columns="metric",
        values="value",
        aggfunc="last",
    ).reset_index()
    return pivot


def _fact_history(payload: dict[str, Any], tags: list[str], unit_preference: list[str]) -> pd.DataFrame:
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    frames = []
    for tag in tags:
        units = us_gaap.get(tag, {}).get("units", {})
        for unit in unit_preference:
            values = units.get(unit, [])
            if not values:
                continue
            frame = pd.DataFrame(values)
            if frame.empty or "val" not in frame.columns or "filed" not in frame.columns:
                continue
            frame["filed_date"] = pd.to_datetime(frame["filed"], errors="coerce")
            frame["period_end"] = pd.to_datetime(frame.get("end"), errors="coerce")
            frame["value"] = pd.to_numeric(frame["val"], errors="coerce")
            frame["form"] = frame.get("form", "")
            frame["tag"] = tag
            frame["unit"] = unit
            frame = frame.dropna(subset=["filed_date", "period_end", "value"])
            frames.append(frame[["filed_date", "period_end", "form", "value", "tag", "unit"]])
            break
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["filed_date", "period_end"]).drop_duplicates(
        ["filed_date", "period_end", "form"], keep="last"
    )
    return combined


def add_fundamental_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in POINT_IN_TIME_FACTS:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    assets = result["assets"].replace(0, pd.NA)
    revenue = result["revenue"].replace(0, pd.NA)
    equity = result["equity"].replace(0, pd.NA)
    result["sec_revenue_to_assets"] = result["revenue"] / assets
    result["sec_net_margin"] = result["net_income"] / revenue
    result["sec_operating_margin"] = result["operating_income"] / revenue
    result["sec_liabilities_to_assets"] = result["liabilities"] / assets
    result["sec_cash_to_assets"] = result["cash"] / assets
    result["sec_ocf_to_assets"] = result["operating_cash_flow"] / assets
    result["sec_roe"] = result["net_income"] / equity
    result["sec_profitable"] = (result["net_income"] > 0).astype(float)
    result["sec_positive_ocf"] = (result["operating_cash_flow"] > 0).astype(float)
    keep = [
        "ticker",
        "filed_date",
        "period_end",
        "form",
        "sec_revenue_to_assets",
        "sec_net_margin",
        "sec_operating_margin",
        "sec_liabilities_to_assets",
        "sec_cash_to_assets",
        "sec_ocf_to_assets",
        "sec_roe",
        "sec_profitable",
        "sec_positive_ocf",
    ]
    return result[keep]


def _date_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/US_PORTFOLIO_HELD.yaml")
    args = parser.parse_args()
    for name, path in collect_sec_companyfacts(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
