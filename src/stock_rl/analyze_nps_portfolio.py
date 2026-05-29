from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class Holding:
    asset_class: str
    year: int
    rank: int
    name: str
    value_억원: float
    weight_pct: float
    stake_pct: float | None


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + ord(ch) - 64
    return idx - 1


def _shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", XLSX_NS):
        strings.append("".join(t.text or "" for t in item.findall(".//a:t", XLSX_NS)).strip())
    return strings


def _xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:sheetData/a:row", XLSX_NS):
            values: list[str] = []
            for cell in row.findall("a:c", XLSX_NS):
                idx = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", XLSX_NS)
                inline_node = cell.find("a:is", XLSX_NS)
                value = ""
                if cell_type == "s" and value_node is not None:
                    value = shared[int(value_node.text or "0")]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(t.text or "" for t in inline_node.findall(".//a:t", XLSX_NS))
                elif value_node is not None:
                    value = value_node.text or ""
                values[idx] = value.strip()
            rows.append(values)
        return rows


def _parse_float(value: str) -> float | None:
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _year_and_class(path: Path) -> tuple[int, str]:
    year_match = re.search(r"(20\d{2})", path.name)
    if not year_match:
        raise ValueError(f"cannot parse year from {path.name}")
    asset_class = "domestic" if "국내" in path.name else "global"
    return int(year_match.group(1)), asset_class


def _parse_file(path: Path) -> list[Holding]:
    year, asset_class = _year_and_class(path)
    rows = _xlsx_rows(path)
    header_idx = None
    for idx, row in enumerate(rows):
        if len(row) >= 4 and row[0] == "번호" and "종목명" in row[1]:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"cannot find header in {path}")

    data_rows = rows[header_idx + 1 :]
    sample_weights = [_parse_float(row[3]) for row in data_rows[:20] if len(row) > 3]
    sample_stakes = [_parse_float(row[4]) for row in data_rows[:20] if len(row) > 4]
    sample_weights = [x for x in sample_weights if x is not None]
    sample_stakes = [x for x in sample_stakes if x is not None]
    weight_scale = 1.0 if sample_weights and max(sample_weights) > 1.0 else 100.0
    # The source files mix ratio and percent notation across years.
    # If the weight column is already percent, the stake column in that file is
    # also percent. Domestic 2020 additionally has stake values above 1.
    if sample_stakes and max(sample_stakes) > 1.0:
        stake_scale = 1.0
    else:
        stake_scale = 1.0 if weight_scale == 1.0 else 100.0

    holdings: list[Holding] = []
    for row in data_rows:
        if len(row) < 4:
            continue
        rank = _parse_float(row[0])
        value = _parse_float(row[2])
        weight_raw = _parse_float(row[3])
        if rank is None or value is None or weight_raw is None or not row[1]:
            continue
        stake = _parse_float(row[4]) if len(row) > 4 else None
        weight_pct = weight_raw * weight_scale
        stake_pct = stake * stake_scale if stake is not None else None
        holdings.append(
            Holding(
                asset_class=asset_class,
                year=year,
                rank=int(rank),
                name=row[1].strip(),
                value_억원=value,
                weight_pct=weight_pct,
                stake_pct=stake_pct,
            )
        )
    return holdings


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _holding_dict(row: Holding) -> dict[str, object]:
    return {
        "asset_class": row.asset_class,
        "year": row.year,
        "rank": row.rank,
        "name": row.name,
        "value_억원": round(row.value_억원, 4),
        "weight_pct": round(row.weight_pct, 6),
        "stake_pct": "" if row.stake_pct is None else round(row.stake_pct, 6),
    }


def _stats(holdings: list[Holding]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[Holding]] = {}
    for row in holdings:
        by_key.setdefault((row.asset_class, row.name), []).append(row)

    stats: list[dict[str, object]] = []
    for (asset_class, name), items in by_key.items():
        items = sorted(items, key=lambda x: x.year)
        by_year = {x.year: x for x in items}
        y2020 = by_year.get(2020)
        y2024 = by_year.get(2024)
        latest = items[-1]
        weight_2020 = y2020.weight_pct if y2020 else 0.0
        weight_2024 = y2024.weight_pct if y2024 else 0.0
        value_2020 = y2020.value_억원 if y2020 else 0.0
        value_2024 = y2024.value_억원 if y2024 else 0.0
        stats.append(
            {
                "asset_class": asset_class,
                "name": name,
                "years_held": len({x.year for x in items}),
                "first_year": items[0].year,
                "last_year": latest.year,
                "rank_2024": y2024.rank if y2024 else "",
                "weight_2020_pct": round(weight_2020, 6),
                "weight_2024_pct": round(weight_2024, 6),
                "weight_change_2020_2024_pp": round(weight_2024 - weight_2020, 6),
                "value_2020_억원": round(value_2020, 4),
                "value_2024_억원": round(value_2024, 4),
                "value_change_2020_2024_억원": round(value_2024 - value_2020, 4),
                "best_rank": min(x.rank for x in items),
                "latest_rank": latest.rank,
                "latest_weight_pct": round(latest.weight_pct, 6),
            }
        )
    return stats


def _load_current_positions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def _match_current_positions(
    current_positions: list[dict[str, str]], stats: list[dict[str, object]]
) -> list[dict[str, object]]:
    domestic_by_name = {(str(row["asset_class"]), str(row["name"])): row for row in stats}
    global_aliases = {
        "NVDA": "NVIDIA CORP",
        "AMD": "ADVANCED MICRO DEVICES",
        "PG": "PROCTER + GAMBLE CO/THE",
        "GLD": "SPDR GOLD SHARES",
        "DVN": "DEVON ENERGY CORP",
        "COP": "CONOCOPHILLIPS",
        "QUBT": "QUANTUM COMPUTING INC",
        "IONQ": "IONQ INC",
    }
    rows: list[dict[str, object]] = []
    for pos in current_positions:
        ticker = pos.get("ticker", "")
        name = pos.get("name", "")
        key = ("domestic", name)
        matched = domestic_by_name.get(key)
        if not matched and ticker in global_aliases:
            matched = domestic_by_name.get(("global", global_aliases[ticker]))
        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "quantity": pos.get("quantity", ""),
                "market_value": pos.get("market_value", ""),
                "currency": pos.get("currency", ""),
                "nps_asset_class": matched.get("asset_class", "") if matched else "",
                "nps_years_held": matched.get("years_held", "") if matched else "",
                "nps_rank_2024": matched.get("rank_2024", "") if matched else "",
                "nps_weight_2024_pct": matched.get("weight_2024_pct", "") if matched else "",
                "nps_weight_change_pp": matched.get("weight_change_2020_2024_pp", "") if matched else "",
            }
        )
    return rows


def _top(rows: list[dict[str, object]], asset_class: str, predicate, n: int) -> list[dict[str, object]]:
    candidates = [r for r in rows if r["asset_class"] == asset_class and predicate(r)]
    candidates.sort(key=lambda r: (r["rank_2024"] == "", int(r["rank_2024"] or 999999)))
    return candidates[:n]


def _fmt_pct(value: object) -> str:
    if value == "":
        return "-"
    return f"{float(value):.2f}%"


def _fmt_pp(value: object) -> str:
    if value == "":
        return "-"
    return f"{float(value):+.2f}pp"


def _table(rows: list[dict[str, object]], cols: list[str], labels: list[str], limit: int = 15) -> str:
    out = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows[:limit]:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if "pct" in col:
                vals.append(_fmt_pct(val))
            elif "pp" in col:
                vals.append(_fmt_pp(val))
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def _write_report(
    path: Path,
    holdings: list[Holding],
    stats: list[dict[str, object]],
    current_matches: list[dict[str, object]],
) -> None:
    domestic_2024 = [h for h in holdings if h.asset_class == "domestic" and h.year == 2024]
    global_2024 = [h for h in holdings if h.asset_class == "global" and h.year == 2024]

    domestic_core = _top(
        stats,
        "domestic",
        lambda r: r["years_held"] == 5 and r["rank_2024"] != "" and int(r["rank_2024"]) <= 30,
        12,
    )
    global_core = _top(
        stats,
        "global",
        lambda r: r["years_held"] == 5 and r["rank_2024"] != "" and int(r["rank_2024"]) <= 30,
        12,
    )
    domestic_risers = sorted(
        [
            r
            for r in stats
            if r["asset_class"] == "domestic"
            and r["rank_2024"] != ""
            and int(r["rank_2024"]) <= 80
            and float(r["weight_change_2020_2024_pp"]) > 0
        ],
        key=lambda r: float(r["weight_change_2020_2024_pp"]),
        reverse=True,
    )[:15]
    global_risers = sorted(
        [
            r
            for r in stats
            if r["asset_class"] == "global"
            and r["rank_2024"] != ""
            and int(r["rank_2024"]) <= 80
            and float(r["weight_change_2020_2024_pp"]) > 0
        ],
        key=lambda r: float(r["weight_change_2020_2024_pp"]),
        reverse=True,
    )[:15]
    new_leaders = sorted(
        [
            r
            for r in stats
            if r["rank_2024"] != "" and int(r["rank_2024"]) <= 80 and int(r["first_year"]) >= 2022
        ],
        key=lambda r: (str(r["asset_class"]), int(r["rank_2024"])),
    )[:20]

    lines = [
        "# 국민연금 2020-2024 종목별 투자 현황 분석",
        "",
        "## 목적",
        "",
        "국민연금기금운용본부의 국내주식/해외주식 종목별 투자 현황 XLSX를 2020년 말부터 2024년 말까지 정규화해, 장기 보유 핵심 종목과 비중 증가 종목을 확인했다. 이 결과는 단기 매수 신호가 아니라 장기 운용 기준선과 공부용 후보군을 만들기 위한 자료다.",
        "",
        "## 사용 데이터",
        "",
        f"- 국내주식 원천 파일: 5개, 2024년 종목 수 `{len(domestic_2024)}`",
        f"- 해외주식 원천 파일: 5개, 2024년 종목 수 `{len(global_2024)}`",
        "- 금액 단위: 원본 기준 `억원`",
        "- 비중: 원본의 ratio 값을 `%`로 변환",
        "",
        "## 2024년 말 상위 보유",
        "",
        "### 국내주식",
        "",
        _table([_holding_dict(h) for h in domestic_2024[:15]], ["rank", "name", "value_억원", "weight_pct", "stake_pct"], ["순위", "종목", "평가액(억원)", "비중", "지분율"]),
        "",
        "### 해외주식",
        "",
        _table([_holding_dict(h) for h in global_2024[:15]], ["rank", "name", "value_억원", "weight_pct", "stake_pct"], ["순위", "종목", "평가액(억원)", "비중", "지분율"]),
        "",
        "## 5년 연속 보유한 2024년 상위 핵심 종목",
        "",
        "### 국내",
        "",
        _table(domestic_core, ["rank_2024", "name", "years_held", "weight_2024_pct", "weight_change_2020_2024_pp"], ["2024순위", "종목", "보유연수", "2024비중", "20→24 변화"]),
        "",
        "### 해외",
        "",
        _table(global_core, ["rank_2024", "name", "years_held", "weight_2024_pct", "weight_change_2020_2024_pp"], ["2024순위", "종목", "보유연수", "2024비중", "20→24 변화"]),
        "",
        "## 2020→2024 비중 증가 상위",
        "",
        "### 국내",
        "",
        _table(domestic_risers, ["rank_2024", "name", "weight_2024_pct", "weight_change_2020_2024_pp", "value_change_2020_2024_억원"], ["2024순위", "종목", "2024비중", "비중증가", "평가액증가(억원)"]),
        "",
        "### 해외",
        "",
        _table(global_risers, ["rank_2024", "name", "weight_2024_pct", "weight_change_2020_2024_pp", "value_change_2020_2024_억원"], ["2024순위", "종목", "2024비중", "비중증가", "평가액증가(억원)"]),
        "",
        "## 2022년 이후 편입된 2024년 상위권 종목",
        "",
        _table(new_leaders, ["asset_class", "rank_2024", "name", "first_year", "weight_2024_pct"], ["자산군", "2024순위", "종목", "최초등장", "2024비중"]),
        "",
        "## 현재 보유종목과 국민연금 데이터 비교",
        "",
        _table(current_matches, ["ticker", "name", "nps_asset_class", "nps_years_held", "nps_rank_2024", "nps_weight_2024_pct", "nps_weight_change_pp"], ["티커", "내 종목", "NPS자산군", "보유연수", "2024순위", "2024비중", "20→24 변화"], limit=40),
        "",
        "## 매수 준비 후보를 고르는 방식",
        "",
        "- 1순위 관찰군: 5년 연속 보유했고 2024년에도 상위권인 종목. 국민연금의 장기 핵심 노출로 해석할 수 있다.",
        "- 2순위 관찰군: 2020년 대비 2024년 비중이 늘었고 2024년 상위권에 남아 있는 종목. 시장 주도권 변화가 반영됐을 가능성이 있다.",
        "- 3순위 관찰군: 2022년 이후 상위권으로 들어온 종목. 새 성장 산업이나 지수 편입 효과를 공부하기 좋지만 변동성 검증이 필요하다.",
        "",
        "## 내 포트폴리오 관점의 해석",
        "",
        "- 삼성전자와 현대모비스는 국민연금 2024년 국내 상위 보유 종목이며, 현재 보유와 장기 대형주 기준선이 겹친다.",
        "- LG전자는 국민연금 보유 데이터에 있으나 2024년 최상위 핵심군보다는 비중/순위 확인 후 분할 매수 기준을 따로 세우는 편이 맞다.",
        "- 국내 ETF를 모으는 전략은 국민연금의 개별 종목 상위 포트폴리오를 그대로 복제하기보다, 대형주/반도체/자동차/금융/바이오 노출을 ETF로 얼마나 가져가는지 점검하는 용도로 쓰는 편이 적절하다.",
        "- 해외는 NVIDIA, Microsoft, Apple, Amazon, Meta, Alphabet, Broadcom 같은 초대형 성장주와 미국 ETF 노출이 핵심이다. 개별주로 따라가기보다 미국 지수 ETF를 기본축으로 두고, 개별주는 과도한 쏠림을 제한하는 방식이 더 현실적이다.",
        "",
        "## 주의",
        "",
        "이 분석은 2024년 말 공시 기준의 과거 보유 내역이다. 현재 가격, 실적, 환율, 금리, 세금, 본인 현금흐름을 반영하지 않으므로 즉시 매수 지시로 쓰면 안 된다. 실제 매수는 별도의 가격 기준과 리스크 한도를 둔 분할 접근이 필요하다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_nps/raw")
    parser.add_argument("--positions", default="data_krx/raw/positions/current_positions.csv")
    parser.add_argument("--out-dir", default="reports/nps")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"no xlsx files found in {raw_dir}")

    holdings: list[Holding] = []
    for file in files:
        holdings.extend(_parse_file(file))
    holdings.sort(key=lambda h: (h.asset_class, h.year, h.rank))

    holding_rows = [_holding_dict(row) for row in holdings]
    holding_fields = ["asset_class", "year", "rank", "name", "value_억원", "weight_pct", "stake_pct"]
    _write_csv(out_dir / "nps_holdings_2020_2024.csv", holding_rows, holding_fields)

    stats = _stats(holdings)
    stat_fields = [
        "asset_class",
        "name",
        "years_held",
        "first_year",
        "last_year",
        "rank_2024",
        "weight_2020_pct",
        "weight_2024_pct",
        "weight_change_2020_2024_pp",
        "value_2020_억원",
        "value_2024_억원",
        "value_change_2020_2024_억원",
        "best_rank",
        "latest_rank",
        "latest_weight_pct",
    ]
    _write_csv(out_dir / "nps_holding_changes_2020_2024.csv", stats, stat_fields)

    current_matches = _match_current_positions(_load_current_positions(Path(args.positions)), stats)
    current_fields = [
        "ticker",
        "name",
        "quantity",
        "market_value",
        "currency",
        "nps_asset_class",
        "nps_years_held",
        "nps_rank_2024",
        "nps_weight_2024_pct",
        "nps_weight_change_pp",
    ]
    _write_csv(out_dir / "nps_current_position_overlap.csv", current_matches, current_fields)
    _write_report(out_dir / "nps_portfolio_analysis_2020_2024.md", holdings, stats, current_matches)

    print(out_dir / "nps_holdings_2020_2024.csv")
    print(out_dir / "nps_holding_changes_2020_2024.csv")
    print(out_dir / "nps_current_position_overlap.csv")
    print(out_dir / "nps_portfolio_analysis_2020_2024.md")


if __name__ == "__main__":
    main()
