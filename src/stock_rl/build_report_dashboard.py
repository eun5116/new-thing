from __future__ import annotations

import argparse
import re
from pathlib import Path

from stock_rl.config import project_path
from stock_rl.report_png import render_report_dashboard_png


REPORT_KEYS = {
    "trading_sheet": "trading_sheet_*_strong_trend_full_else070.png",
    "target_changes": "target_changes_*_strong_trend_full_else070.png",
    "rebalance_orders": "rebalance_orders_*_strong_trend_full_else070.png",
    "current_position_analysis": "current_position_analysis_*.png",
    "portfolio_decision_sheet": "portfolio_decision_sheet_*.png",
}


def _latest_report_path(reports_dir: Path, pattern: str) -> Path | None:
    paths = sorted(reports_dir.glob(pattern))
    return paths[-1] if paths else None


def _extract_date(path: Path) -> str:
    match = re.search(r"(\d{8})", path.name)
    return match.group(1) if match else ""


def build_report_dashboard(config_path: str, out_dir: str | None = None) -> dict[str, Path] | None:
    reports_dir = project_path(config_path, "reports")
    selected = {name: _latest_report_path(reports_dir, pattern) for name, pattern in REPORT_KEYS.items()}
    if not any(selected.values()):
        return None
    valid_dates = [_extract_date(path) for path in selected.values() if path is not None]
    as_of = max(valid_dates) if valid_dates else "dashboard"
    output_dir = Path(out_dir) if out_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"report_dashboard_{as_of}.png"
    render_report_dashboard_png(selected, png_path, title=f"Report Dashboard - {as_of}")
    return {"png": png_path, **{name: path for name, path in selected.items() if path is not None}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/KRX_E032_liquid48_long_trend_min_exposure.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    result = build_report_dashboard(args.config, args.out_dir)
    if result is None:
        print("no report PNGs found")
        return
    for name, path in result.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
