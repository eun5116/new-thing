# 실행 코드: python weekly_report.py
import argparse
import os
import sys
import smtplib
import datetime as dt
import time
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf
from pykrx import stock
from pykrx.website.krx.market import wrap as krx_wrap
from pathlib import Path
import json

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
ALERT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "dataset"
ALERT_CONFIG_PATHS = [
    ALERT_CONFIG_DIR / "us_equities_alerts.yaml",
    ALERT_CONFIG_DIR / "kospi_alerts.yaml",
]
SP500_CACHE_PATH = CACHE_DIR / "sp500_top20.json"
KOSPI_CACHE_PATH = CACHE_DIR / "kospi_top20.json"
SP500_REPORT_CACHE_PATH = CACHE_DIR / "sp500_weekly_rows.json"
SP500_COLUMNS = ["Ticker", "StartDate", "EndDate", "Start", "End", "ChangePct"]
KOSPI_COLUMNS = ["Name", "Ticker", "StartDate", "EndDate", "Start", "End", "ChangePct"]
ALERT_COLUMNS = [
    "market",
    "name",
    "alert_name",
    "symbol",
    "as_of_date",
    "window_trading_days",
    "total_return",
    "up_days",
    "max_consecutive_down_days",
    "triggered",
]
DEFAULT_KOSPI20 = [
    ("삼성전자", "005930"),
    ("SK하이닉스", "000660"),
    ("LG에너지솔루션", "373220"),
    ("삼성바이오로직스", "207940"),
    ("현대차", "005380"),
    ("셀트리온", "068270"),
    ("NAVER", "035420"),
    ("삼성전자우", "005935"),
    ("기아", "000270"),
    ("KB금융", "105560"),
    ("신한지주", "055550"),
    ("POSCO홀딩스", "005490"),
    ("삼성SDI", "006400"),
    ("현대모비스", "012330"),
    ("LG화학", "051910"),
    ("삼성물산", "028260"),
    ("SK이노베이션", "096770"),
    ("한국전력", "015760"),
    ("LG전자", "066570"),
    ("SK텔레콤", "017670"),
]
OFFLINE_MODE = False


def _finalize_report_df(df):
    if "ChangePct" in df.columns:
        df["ChangePct"] = pd.to_numeric(df["ChangePct"], errors="coerce")
        if not df.empty:
            df["ChangePct"] = df["ChangePct"].round(2)
    if "total_return" in df.columns:
        df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce")
        if not df.empty:
            df["total_return"] = df["total_return"].round(4)
    return df


def _to_html_table(df, empty_message):
    if df is None or df.empty:
        return f"<p>{empty_message}</p>"
    return df.to_html(index=False)


def _to_text_table(df, empty_message):
    if df is None or df.empty:
        return empty_message
    return df.to_string(index=False)


def _safe_config_name(config_path):
    try:
        cfg = load_alert_config(config_path)
        return str(cfg.get("name", config_path.stem))
    except Exception:
        return config_path.stem


def _build_market_snapshot(df, label, name_col):
    if df is None or df.empty or "ChangePct" not in df.columns:
        return [f"{label}: data unavailable"]
    clean = df.dropna(subset=["ChangePct"]).copy()
    if clean.empty:
        return [f"{label}: no valid weekly change rows"]
    top_row = clean.sort_values("ChangePct", ascending=False).iloc[0]
    bottom_row = clean.sort_values("ChangePct", ascending=True).iloc[0]
    top_name = str(top_row.get(name_col, "N/A"))
    bottom_name = str(bottom_row.get(name_col, "N/A"))
    return [
        f"{label}: {len(clean)} rows, strongest {top_name} ({top_row['ChangePct']:.2f}%), weakest {bottom_name} ({bottom_row['ChangePct']:.2f}%)"
    ]


def _build_alert_snapshot(alert_results, alert_statuses):
    lines = []
    status_map = {config_path.name: status for config_path, status in alert_statuses}
    for config_path, alerts_df in alert_results:
        section_title = _safe_config_name(config_path)
        triggered_count = 0
        top_symbol = ""
        top_return = None
        if alerts_df is not None and not alerts_df.empty and "triggered" in alerts_df.columns:
            triggered = alerts_df[alerts_df["triggered"]].copy()
            triggered_count = len(triggered)
            if not triggered.empty and "total_return" in triggered.columns:
                top_row = triggered.sort_values("total_return", ascending=False).iloc[0]
                top_symbol = str(top_row.get("symbol", "")) or str(top_row.get("name", ""))
                top_return = top_row.get("total_return")
        line = f"{section_title}: {triggered_count} triggered"
        if pd.notna(top_return):
            line += f", top {top_symbol} ({float(top_return) * 100:.2f}%)"
        status = status_map.get(config_path.name, "")
        if status:
            line += f" [{status}]"
        lines.append(line)
    return lines or ["Momentum alerts: unavailable"]


def _build_summary_sections(kospi_df, kospi_status, sp_df, alert_results, alert_statuses):
    summary_lines = []
    summary_lines.extend(_build_market_snapshot(kospi_df, "KOSPI", "Name"))
    if kospi_status:
        summary_lines.append(f"KOSPI status: {kospi_status}")
    summary_lines.extend(_build_market_snapshot(sp_df, "S&P 500", "Ticker"))
    summary_lines.extend(_build_alert_snapshot(alert_results, alert_statuses))
    summary_html = "".join(f"<li>{line}</li>" for line in summary_lines)
    return (
        "<h3>Weekly Summary</h3>\n<ul>" + summary_html + "</ul>",
        "Weekly Summary:\n" + "\n".join(f"- {line}" for line in summary_lines),
    )


def _parse_simple_yaml_scalar(raw):
    value = raw.strip().strip('"').strip("'")
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path):
    parsed = {}
    current_list_key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current_list_key is None:
                continue
            parsed[current_list_key].append(_parse_simple_yaml_scalar(stripped[2:]))
            continue
        current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parsed[key] = _parse_simple_yaml_scalar(value)
        else:
            parsed[key] = []
            current_list_key = key
    return parsed


def load_alert_config(path):
    cfg = _load_simple_yaml(path)
    cfg.setdefault("name", path.stem)
    cfg.setdefault("alert_name", "two_week_momentum_watch")
    cfg.setdefault("window_trading_days", 10)
    cfg.setdefault("min_total_return", 0.10)
    cfg.setdefault("trend_policy", "tolerant")
    cfg.setdefault("min_up_days", 8)
    cfg.setdefault("max_consecutive_down_days", 2)
    cfg.setdefault("price_basis", "adjusted_close")
    cfg.setdefault("interval", "1d")
    cfg.setdefault("universe", [])
    cfg.setdefault("market_scope", "custom")
    return cfg

def _safe_json_dump(path, payload):
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _safe_json_load(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def set_offline_mode(enabled):
    global OFFLINE_MODE
    OFFLINE_MODE = bool(enabled)


def _is_offline_mode():
    return OFFLINE_MODE

def _retry_pykrx(callable_fn, tries=3, delay=1.0):
    last_error = None
    for attempt in range(tries):
        try:
            return callable_fn(), None
        except Exception as e:
            last_error = e
            if attempt < tries - 1:
                time.sleep(delay)
    return None, last_error


def _slugify_filename(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return slug or "weekly_market_report"


def _timestamp_now():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_report_artifacts(subject, text, html, output_dir=OUTPUT_DIR, metadata=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{_timestamp_now()}_{_slugify_filename(subject)}"
    text_path = output_dir / f"{base_name}.txt"
    html_path = output_dir / f"{base_name}.html"
    meta_path = output_dir / f"{base_name}.json"
    text_path.write_text(text, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    payload = {
        "saved_at": dt.datetime.now().isoformat(),
        "subject": subject,
        "text_path": str(text_path),
        "html_path": str(html_path),
    }
    if metadata:
        payload.update(metadata)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"text": text_path, "html": html_path, "meta": meta_path}


def _append_history_rows(path, rows, columns):
    new_df = pd.DataFrame(rows, columns=columns)
    if path.exists() and not new_df.empty:
        existing_df = pd.read_csv(path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif path.exists():
        combined_df = pd.read_csv(path)
    else:
        combined_df = new_df
    combined_df.to_csv(path, index=False)


def _normalize_sp500_symbol(symbol):
    return str(symbol).strip().upper().replace(".", "-")


def save_market_history(report_date, kospi_df, sp_df, alert_results, history_dir=HISTORY_DIR):
    history_dir.mkdir(parents=True, exist_ok=True)
    captured_at = dt.datetime.now().isoformat()
    market_columns = [
        "captured_at",
        "report_date",
        "market",
        "symbol",
        "name",
        "start_date",
        "end_date",
        "start_price",
        "end_price",
        "change_pct",
    ]
    alert_columns = [
        "captured_at",
        "report_date",
        "section",
        "market",
        "symbol",
        "name",
        "alert_name",
        "as_of_date",
        "window_trading_days",
        "total_return",
        "up_days",
        "max_consecutive_down_days",
        "triggered",
    ]

    kospi_rows = []
    if kospi_df is not None and not kospi_df.empty:
        for row in kospi_df.to_dict(orient="records"):
            kospi_rows.append(
                {
                    "captured_at": captured_at,
                    "report_date": report_date,
                    "market": "KOSPI",
                    "symbol": row.get("Ticker", ""),
                    "name": row.get("Name", ""),
                    "start_date": row.get("StartDate", ""),
                    "end_date": row.get("EndDate", ""),
                    "start_price": row.get("Start", ""),
                    "end_price": row.get("End", ""),
                    "change_pct": row.get("ChangePct", ""),
                }
            )

    sp_rows = []
    if sp_df is not None and not sp_df.empty:
        for row in sp_df.to_dict(orient="records"):
            sp_rows.append(
                {
                    "captured_at": captured_at,
                    "report_date": report_date,
                    "market": "SP500",
                    "symbol": row.get("Ticker", ""),
                    "name": row.get("Ticker", ""),
                    "start_date": row.get("StartDate", ""),
                    "end_date": row.get("EndDate", ""),
                    "start_price": row.get("Start", ""),
                    "end_price": row.get("End", ""),
                    "change_pct": row.get("ChangePct", ""),
                }
            )

    alert_rows = []
    for config_path, alerts_df in alert_results:
        section_name = _safe_config_name(config_path)
        if alerts_df is None or alerts_df.empty:
            continue
        for row in alerts_df.to_dict(orient="records"):
            alert_rows.append(
                {
                    "captured_at": captured_at,
                    "report_date": report_date,
                    "section": section_name,
                    "market": row.get("market", ""),
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "alert_name": row.get("alert_name", ""),
                    "as_of_date": row.get("as_of_date", ""),
                    "window_trading_days": row.get("window_trading_days", ""),
                    "total_return": row.get("total_return", ""),
                    "up_days": row.get("up_days", ""),
                    "max_consecutive_down_days": row.get("max_consecutive_down_days", ""),
                    "triggered": row.get("triggered", ""),
                }
            )

    _append_history_rows(history_dir / "kospi_top20_history.csv", kospi_rows, market_columns)
    _append_history_rows(history_dir / "sp500_top20_history.csv", sp_rows, market_columns)
    _append_history_rows(history_dir / "momentum_alert_history.csv", alert_rows, alert_columns)

    return {
        "kospi": history_dir / "kospi_top20_history.csv",
        "sp500": history_dir / "sp500_top20_history.csv",
        "alerts": history_dir / "momentum_alert_history.csv",
    }


def _find_latest_kospi_caps(max_lookback_days=14):
    if _is_offline_mode():
        return None, None, RuntimeError("Offline mode enabled; skipping live KOSPI market cap fetch.")
    today = dt.datetime.now().date()
    last_error = None
    for offset in range(max_lookback_days + 1):
        target = (today - dt.timedelta(days=offset)).strftime("%Y%m%d")
        # Use low-level wrapper to avoid pykrx stock_api holiday-column bug.
        caps, err = _retry_pykrx(lambda: krx_wrap.get_market_cap_by_ticker(target, market="KOSPI"))
        if err:
            last_error = err
            continue
        if caps is not None and not caps.empty and "시가총액" in caps.columns:
            return target, caps, None
    if last_error is not None:
        return None, None, RuntimeError(f"KRX fetch error: {last_error}")
    return None, None, RuntimeError(
        "Unable to find non-empty KOSPI market cap data in recent business days."
    )

def _load_kospi_cache_df():
    cached = _safe_json_load(KOSPI_CACHE_PATH)
    if not cached:
        return pd.DataFrame(columns=KOSPI_COLUMNS), ""
    rows = cached.get("rows", [])
    note = cached.get("note", "")
    df = pd.DataFrame(rows, columns=KOSPI_COLUMNS)
    return _finalize_report_df(df), note


def _save_sp500_report_cache_df(df):
    _safe_json_dump(
        SP500_REPORT_CACHE_PATH,
        {
            "saved_at": dt.datetime.now().isoformat(),
            "rows": df.to_dict(orient="records"),
            "note": "Cached S&P 500 weekly rows.",
        },
    )


def _load_sp500_report_cache_df():
    cached = _safe_json_load(SP500_REPORT_CACHE_PATH)
    if not cached:
        return pd.DataFrame(columns=SP500_COLUMNS), ""
    rows = cached.get("rows", [])
    note = cached.get("note", "")
    df = pd.DataFrame(rows, columns=SP500_COLUMNS)
    return _finalize_report_df(df), note


def _get_cached_kospi_symbols():
    cached_df, _ = _load_kospi_cache_df()
    if cached_df.empty or "Ticker" not in cached_df.columns:
        return []
    return [str(symbol).strip() for symbol in cached_df["Ticker"].dropna().tolist() if str(symbol).strip()]

def _get_df_latest_end_date(df, default=""):
    if df is None or df.empty or "EndDate" not in df.columns:
        return default
    dates = pd.to_datetime(df["EndDate"], errors="coerce").dropna()
    if dates.empty:
        return default
    return dates.max().strftime("%Y%m%d")

def _save_kospi_cache_df(df, end_date):
    _safe_json_dump(
        KOSPI_CACHE_PATH,
        {
            "saved_at": dt.datetime.now().isoformat(),
            "end_date": end_date,
            "rows": df.to_dict(orient="records"),
            "note": f"Cached KOSPI snapshot (end date: {end_date}).",
        },
    )

def _fetch_naver_kospi_top20():
    if _is_offline_mode():
        raise RuntimeError("Offline mode enabled; skipping live Naver fetch.")
    headers = {"User-Agent": "Mozilla/5.0"}
    found = []
    seen = set()
    for page in range(1, 6):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}"
        html = requests.get(url, headers=headers, timeout=10).text
        matches = re.findall(r'href="/item/main\.naver\?code=(\d{6})"[^>]*>([^<]+)</a>', html)
        for code, name in matches:
            if code in seen:
                continue
            seen.add(code)
            found.append((name.strip(), code.strip()))
            if len(found) >= 20:
                return found
    return found

def _extract_close_series(data):
    if data is None or data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close_df = data["Close"]
            if isinstance(close_df, pd.DataFrame):
                return close_df.iloc[:, 0].dropna()
            return close_df.dropna()
        return pd.Series(dtype=float)
    if "Close" not in data.columns:
        return pd.Series(dtype=float)
    return data["Close"].dropna()


def _extract_symbol_close_series(data, symbol):
    if data is None or getattr(data, "empty", True):
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return pd.Series(dtype=float)
        symbol_frame = data[symbol]
        if isinstance(symbol_frame, pd.DataFrame) and "Close" in symbol_frame.columns:
            return symbol_frame["Close"].dropna()
        return pd.Series(dtype=float)
    return _extract_close_series(data)


def _max_consecutive_negative(values):
    max_run = 0
    current_run = 0
    for value in values:
        if value < 0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def _chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _get_latest_kospi_date():
    end_date, caps, caps_error = _find_latest_kospi_caps(max_lookback_days=14)
    if caps_error:
        raise RuntimeError(caps_error)
    return end_date, caps


def _resolve_alert_universe(config):
    explicit_universe = [str(symbol).strip().upper() for symbol in config.get("universe", []) if str(symbol).strip()]
    if explicit_universe:
        return explicit_universe, {}

    market_scope = str(config.get("market_scope", "custom")).lower()
    if market_scope == "sp500":
        symbols = [str(symbol).strip().upper() for symbol in get_sp500_tickers() if str(symbol).strip()]
        return symbols, {}
    if market_scope == "kospi":
        if _is_offline_mode():
            return _get_cached_kospi_symbols(), {}
        end_date, _ = _get_latest_kospi_date()
        symbols = stock.get_market_ticker_list(date=end_date, market="KOSPI")
        return [str(symbol).strip() for symbol in symbols if str(symbol).strip()], {}
    return [], {}


def _download_yfinance_close_map(symbols, interval, auto_adjust, suffix="", chunk_size=100):
    if _is_offline_mode():
        return {}
    close_map = {}
    for chunk in _chunked(symbols, chunk_size):
        actual_symbols = [f"{symbol}{suffix}" for symbol in chunk]
        try:
            data = yf.download(
                tickers=" ".join(actual_symbols),
                period="1mo",
                interval=interval,
                auto_adjust=auto_adjust,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue
        if len(actual_symbols) == 1:
            close_map[chunk[0]] = _extract_close_series(data)
            continue
        for symbol, actual_symbol in zip(chunk, actual_symbols):
            close_map[symbol] = _extract_symbol_close_series(data, actual_symbol)
    return close_map


def _lookup_kospi_names(symbols):
    names = {}
    for symbol in symbols:
        try:
            name, name_err = _retry_pykrx(lambda symbol=symbol: stock.get_market_ticker_name(symbol))
            if name_err or not name:
                names[symbol] = symbol
            else:
                names[symbol] = name
        except Exception:
            names[symbol] = symbol
    return names


def evaluate_two_week_momentum(series, config):
    clean = pd.Series(series).dropna().astype(float)
    window_trading_days = int(config["window_trading_days"])
    alert_name = str(config["alert_name"])
    result = {
        "market": str(config.get("market_scope", "custom")).upper(),
        "name": "",
        "alert_name": alert_name,
        "symbol": "",
        "as_of_date": "",
        "window_trading_days": window_trading_days,
        "total_return": None,
        "up_days": 0,
        "max_consecutive_down_days": 0,
        "triggered": False,
    }
    if len(clean) < window_trading_days:
        return result

    window = clean.tail(window_trading_days)
    daily_returns = window.pct_change().dropna()
    if daily_returns.empty:
        return result

    total_return = float(window.iloc[-1] / window.iloc[0] - 1.0)
    up_days = int((daily_returns > 0).sum())
    max_consecutive_down_days = _max_consecutive_negative(daily_returns.tolist())
    trend_policy = str(config["trend_policy"]).lower()
    if trend_policy == "strict":
        trend_ok = bool((daily_returns > 0).all())
    else:
        trend_ok = (
            up_days >= int(config["min_up_days"])
            and max_consecutive_down_days <= int(config["max_consecutive_down_days"])
        )

    as_of_date = ""
    if hasattr(window.index, "__len__") and len(window.index) > 0:
        try:
            as_of_date = pd.to_datetime(window.index[-1]).strftime("%Y-%m-%d")
        except Exception:
            as_of_date = str(window.index[-1])

    result.update(
        {
            "as_of_date": as_of_date,
            "total_return": total_return,
            "up_days": up_days,
            "max_consecutive_down_days": max_consecutive_down_days,
            "triggered": bool(total_return >= float(config["min_total_return"]) and trend_ok),
        }
    )
    return result


def get_two_week_momentum_alerts(config_path):
    try:
        cfg = load_alert_config(config_path)
    except Exception as e:
        return pd.DataFrame(columns=ALERT_COLUMNS), f"Alert config load failed: {e}"

    try:
        symbols, metadata = _resolve_alert_universe(cfg)
    except Exception as e:
        return pd.DataFrame(columns=ALERT_COLUMNS), f"Alert universe load failed: {e}"
    if not symbols:
        return pd.DataFrame(columns=ALERT_COLUMNS), "Alert universe is empty."

    market_scope = str(cfg.get("market_scope", "custom")).lower()
    auto_adjust = str(cfg["price_basis"]).lower() == "adjusted_close"
    if market_scope == "kospi":
        close_map = _download_yfinance_close_map(symbols, str(cfg["interval"]), auto_adjust, suffix=".KS", chunk_size=50)
        metadata = {symbol: symbol for symbol in symbols} if _is_offline_mode() else _lookup_kospi_names(symbols)
    else:
        close_map = _download_yfinance_close_map(symbols, str(cfg["interval"]), auto_adjust, chunk_size=100)

    rows = []
    missing_symbols = []
    for symbol in symbols:
        close = close_map.get(symbol, pd.Series(dtype=float))
        evaluated = evaluate_two_week_momentum(close, cfg)
        if evaluated["total_return"] is None:
            missing_symbols.append(symbol)
            continue
        evaluated["symbol"] = symbol
        evaluated["name"] = metadata.get(symbol, symbol)
        rows.append(evaluated)

    df = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    df = _finalize_report_df(df)
    if not df.empty:
        df = df.sort_values(["triggered", "total_return"], ascending=[False, False]).reset_index(drop=True)

    if missing_symbols and df.empty:
        return df, f"Alert price data unavailable for: {', '.join(missing_symbols)}"
    if missing_symbols:
        return df, f"Some alert symbols were skipped due to insufficient price history: {', '.join(missing_symbols)}"
    return df, ""


def get_market_momentum_alerts(config_paths=ALERT_CONFIG_PATHS):
    results = []
    statuses = []
    for config_path in config_paths:
        df, status = get_two_week_momentum_alerts(config_path)
        if not df.empty:
            results.append((config_path, df))
        else:
            results.append((config_path, pd.DataFrame(columns=ALERT_COLUMNS)))
        if status:
            statuses.append((config_path, status))
    return results, statuses

def _get_kospi_top20_from_naver_and_yf(window=5):
    try:
        top20 = _fetch_naver_kospi_top20()
    except Exception as e:
        return pd.DataFrame(columns=KOSPI_COLUMNS), e
    if not top20:
        return pd.DataFrame(columns=KOSPI_COLUMNS), RuntimeError("Naver KOSPI top list is empty.")

    rows = []
    for name, code in top20:
        yf_symbol = f"{code}.KS"
        try:
            data = yf.download(
                tickers=yf_symbol,
                period="1mo",
                interval="1d",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            close = _extract_close_series(data)
            if len(close) < window:
                continue
            week = close.tail(window)
            start = float(week.iloc[0])
            end = float(week.iloc[-1])
            change_pct = (end - start) / start * 100.0
            start_date = pd.to_datetime(week.index[0]).strftime("%Y-%m-%d")
            end_date = pd.to_datetime(week.index[-1]).strftime("%Y-%m-%d")
            rows.append((name, code, start_date, end_date, start, end, change_pct))
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=KOSPI_COLUMNS)
    return _finalize_report_df(df), None

def _get_kospi_top20_from_static_and_yf(window=5):
    rows = []
    for name, code in DEFAULT_KOSPI20:
        yf_symbol = f"{code}.KS"
        try:
            data = yf.download(
                tickers=yf_symbol,
                period="1mo",
                interval="1d",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            close = _extract_close_series(data)
            if len(close) < window:
                continue
            week = close.tail(window)
            start = float(week.iloc[0])
            end = float(week.iloc[-1])
            change_pct = (end - start) / start * 100.0
            start_date = pd.to_datetime(week.index[0]).strftime("%Y-%m-%d")
            end_date = pd.to_datetime(week.index[-1]).strftime("%Y-%m-%d")
            rows.append((name, code, start_date, end_date, start, end, change_pct))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=KOSPI_COLUMNS)
    return _finalize_report_df(df)

def _clean_env(value, default=""):
    v = value if value is not None else default
    # Normalize common .env formatting mistakes (quotes/whitespace)
    return str(v).strip().strip('"').strip("'").strip()

def load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    smtp_host = _clean_env(os.getenv("SMTP_HOST", ""))
    smtp_user = _clean_env(os.getenv("SMTP_USER", ""))
    smtp_pass = _clean_env(os.getenv("SMTP_PASS", ""))
    email_to = _clean_env(os.getenv("EMAIL_TO", ""))
    email_from = _clean_env(os.getenv("EMAIL_FROM"), default=smtp_user) or smtp_user
    tz = _clean_env(os.getenv("REPORT_TIMEZONE", "Asia/Seoul"), default="Asia/Seoul")
    cfg = {
        "smtp_host": smtp_host,
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": smtp_user,
        "smtp_pass": smtp_pass,
        "email_to": email_to,
        "email_from": email_from,
        "tz": tz,
    }
    missing = [k for k, v in cfg.items() if k in ["smtp_host", "smtp_user", "smtp_pass", "email_to"] and not v]
    if missing:
        raise RuntimeError("Missing env: " + ", ".join(missing))
    if "gmail.com" in cfg["smtp_host"].lower():
        pw_len = len(cfg["smtp_pass"])
        if pw_len != 16:
            raise RuntimeError(
                f"SMTP_PASS length is {pw_len}. Gmail SMTP requires a 16-character App Password."
            )
    return cfg
def get_sp500_tickers():
    if _is_offline_mode():
        cached = _safe_json_load(SP500_CACHE_PATH) or {}
        return cached.get("tickers", [])
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=10).text
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    return [_normalize_sp500_symbol(symbol) for symbol in df["Symbol"].tolist()]
def get_sp500_top20():
    cached_payload = None
    if SP500_CACHE_PATH.exists():
        try:
            cached_payload = json.loads(SP500_CACHE_PATH.read_text())
            cached_date = cached_payload.get("date")
            if cached_date:
                age_days = (dt.datetime.now() - dt.datetime.fromisoformat(cached_date)).days
                if age_days <= 7:
                    return cached_payload.get("tickers", [])
        except Exception:
            cached_payload = None
    try:
        tickers = get_sp500_tickers()
    except Exception:
        if cached_payload and cached_payload.get("tickers"):
            return cached_payload.get("tickers", [])
        return []
    caps = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            cap = info.get("marketCap", None)
            if cap:
                caps.append((t, cap))
        except Exception:
            continue
    caps.sort(key=lambda x: x[1], reverse=True)
    top20 = [t for t, _ in caps[:20]]
    if not top20 and cached_payload and cached_payload.get("tickers"):
        return cached_payload.get("tickers", [])
    # save cache
    try:
        SP500_CACHE_PATH.write_text(json.dumps({
            "date": dt.datetime.now().isoformat(),
            "tickers": top20
        }))
    except Exception:
        pass
    return top20
def get_sp500_weekly_change(top20, window=5):
    if not top20:
        cached_df, _ = _load_sp500_report_cache_df()
        return cached_df
    if _is_offline_mode():
        cached_df, _ = _load_sp500_report_cache_df()
        return cached_df
    try:
        data = yf.download(
            tickers=" ".join(top20),
            period="1mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        cached_df, _ = _load_sp500_report_cache_df()
        return cached_df
    rows = []
    for t in top20:
        try:
            series = _extract_symbol_close_series(data, t)
            if len(series) < window:
                continue
            week = series.tail(window)
            start = week.iloc[0]
            end = week.iloc[-1]
            change_pct = (end - start) / start * 100.0
            start_date = week.index[0].strftime("%Y-%m-%d")
            end_date = week.index[-1].strftime("%Y-%m-%d")
            rows.append((t, start_date, end_date, float(start), float(end), float(change_pct)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=SP500_COLUMNS)
    df = _finalize_report_df(df)
    if not df.empty:
        _save_sp500_report_cache_df(df)
        return df
    cached_df, _ = _load_sp500_report_cache_df()
    return cached_df
def get_kospi_top10_and_change(window=5):
    if _is_offline_mode():
        cached_df, cached_note = _load_kospi_cache_df()
        if not cached_df.empty:
            note = cached_note or "Offline mode enabled; used cached KOSPI data."
            return cached_df, note
        return pd.DataFrame(columns=KOSPI_COLUMNS), "Offline mode enabled; no cached KOSPI data available."
    end_date, caps, caps_error = _find_latest_kospi_caps(max_lookback_days=14)
    if caps_error:
        naver_df, naver_err = _get_kospi_top20_from_naver_and_yf(window=window)
        if not naver_df.empty:
            _save_kospi_cache_df(naver_df, _get_df_latest_end_date(naver_df, dt.datetime.now().strftime("%Y%m%d")))
            return naver_df, (
                "Live KOSPI fetch failed (KRX). Used Naver+yfinance fallback."
            )
        static_df = _get_kospi_top20_from_static_and_yf(window=window)
        if not static_df.empty:
            _save_kospi_cache_df(static_df, _get_df_latest_end_date(static_df, dt.datetime.now().strftime("%Y%m%d")))
            return static_df, (
                "Live KOSPI fetch failed (KRX), and Naver fallback failed. "
                "Used static KOSPI20+yfinance fallback."
            )
        cached_df, cached_note = _load_kospi_cache_df()
        if not cached_df.empty:
            return cached_df, (
                "Live KOSPI fetch failed, and live fallbacks failed, so cached data was used. "
                f"{cached_note}"
            )
        if naver_err:
            return (
                pd.DataFrame(columns=KOSPI_COLUMNS),
                f"Live KOSPI fetch failed: {caps_error}; fallback failed: {naver_err}",
            )
        return pd.DataFrame(columns=KOSPI_COLUMNS), f"Live KOSPI fetch failed: {caps_error}"

    lookup_start = (dt.datetime.now() - dt.timedelta(days=45)).strftime("%Y%m%d")
    caps = caps.sort_values("시가총액", ascending=False).head(20)
    tickers = caps.index.tolist()
    rows = []
    for t in tickers:
        try:
            ohlcv, ohlcv_err = _retry_pykrx(
                lambda: stock.get_market_ohlcv_by_date(lookup_start, end_date, t)
            )
            if ohlcv_err:
                continue
            if ohlcv.empty:
                continue
            close = ohlcv["종가"].dropna()
            if len(close) < window:
                continue
            week = close.tail(window)
            start = float(week.iloc[0])
            end = float(week.iloc[-1])
            change_pct = (end - start) / start * 100.0
            name, name_err = _retry_pykrx(lambda: stock.get_market_ticker_name(t))
            if name_err or not name:
                name = t
            start_date = pd.to_datetime(week.index[0]).strftime("%Y-%m-%d")
            end_date_row = pd.to_datetime(week.index[-1]).strftime("%Y-%m-%d")
            rows.append((name, t, start_date, end_date_row, start, end, change_pct))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=KOSPI_COLUMNS)
    df = _finalize_report_df(df)
    if df.empty:
        naver_df, naver_err = _get_kospi_top20_from_naver_and_yf(window=window)
        if not naver_df.empty:
            _save_kospi_cache_df(naver_df, _get_df_latest_end_date(naver_df, dt.datetime.now().strftime("%Y%m%d")))
            return naver_df, "Live KOSPI rows were empty after fetch. Used Naver+yfinance fallback."
        static_df = _get_kospi_top20_from_static_and_yf(window=window)
        if not static_df.empty:
            _save_kospi_cache_df(static_df, _get_df_latest_end_date(static_df, dt.datetime.now().strftime("%Y%m%d")))
            return static_df, (
                "Live KOSPI rows were empty after fetch, and Naver fallback failed. "
                "Used static KOSPI20+yfinance fallback."
            )
        cached_df, cached_note = _load_kospi_cache_df()
        if not cached_df.empty:
            return cached_df, (
                "Live KOSPI rows were empty after fetch, and live fallbacks failed, so cached data was used. "
                f"{cached_note}"
            )
        if naver_err:
            return df, f"Live KOSPI rows were empty after fetch. Fallback failed: {naver_err}"
        return df, "Live KOSPI rows were empty after fetch."
    _save_kospi_cache_df(df, end_date)
    return df, ""
def collect_report_inputs():
    kospi_df, kospi_status = get_kospi_top10_and_change()
    sp_top20 = get_sp500_top20()
    sp_df = get_sp500_weekly_change(sp_top20)
    alert_results, alert_statuses = get_market_momentum_alerts()
    return kospi_df, kospi_status, sp_df, alert_results, alert_statuses


def render_report(kospi_df, kospi_status, sp_df, alert_results, alert_statuses, report_date=None):
    now = report_date or dt.datetime.now().strftime("%Y-%m-%d")
    subject = f"Weekly Market Report - {now}"
    kospi_notice_html = ""
    kospi_notice_text = ""
    alert_sections_html = []
    alert_sections_text = []
    summary_html, summary_text = _build_summary_sections(
        kospi_df, kospi_status, sp_df, alert_results, alert_statuses
    )
    if kospi_df.empty and not kospi_status:
        kospi_notice_html = (
            "<p><strong>KOSPI data is unavailable.</strong> "
            "Source/network issue (for example DNS/connection failure to KRX) may have occurred.</p>"
        )
        kospi_notice_text = (
            "[KOSPI] Data unavailable. Source/network issue "
            "(for example DNS/connection failure to KRX) may have occurred.\n"
        )
    elif kospi_status:
        kospi_notice_html = f"<p><strong>KOSPI notice:</strong> {kospi_status}</p>"
        kospi_notice_text = f"[KOSPI] {kospi_status}\n"
    status_map = {config_path.name: status for config_path, status in alert_statuses}
    for config_path, alerts_df in alert_results:
        section_title = _safe_config_name(config_path)
        triggered_alerts_df = (
            alerts_df[alerts_df["triggered"]].copy()
            if alerts_df is not None and not alerts_df.empty and "triggered" in alerts_df.columns
            else pd.DataFrame(columns=ALERT_COLUMNS)
        )
        section_status = status_map.get(config_path.name, "")
        section_notice_html = ""
        section_notice_text = ""
        if section_status:
            section_notice_html = f"<p><strong>Momentum notice:</strong> {section_status}</p>"
            section_notice_text = f"[Momentum] {section_status}\n"
        elif triggered_alerts_df.empty:
            section_notice_html = "<p><strong>Momentum notice:</strong> No symbols met the two-week alert criteria.</p>"
            section_notice_text = "[Momentum] No symbols met the two-week alert criteria.\n"
        section_html = _to_html_table(triggered_alerts_df, "No triggered alerts.")
        alert_sections_html.append(
            f"<h3>{section_title} Two-Week Momentum Alerts</h3>\n{section_notice_html}\n{section_html}"
        )
        section_text = f"\n\n{section_title} Two-Week Momentum Alerts:\n"
        section_text += section_notice_text
        if triggered_alerts_df.empty:
            section_text += "No triggered alerts."
        else:
            section_text += _to_text_table(triggered_alerts_df, "No triggered alerts.")
        alert_sections_text.append(section_text)
    kospi_html = _to_html_table(kospi_df, "No KOSPI rows available.")
    sp_html = _to_html_table(sp_df, "No S&P 500 rows available.")
    html = f"""
    <html>
    <body>
    <h2>Weekly Market Report ({now})</h2>
    {summary_html}
    <h3>KOSPI Top 20 by Market Cap</h3>
    {kospi_notice_html}
    {kospi_html}
    <h3>S&P 500 Top 20 by Market Cap</h3>
    {sp_html}
    {''.join(alert_sections_html)}
    </body>
    </html>
    """
    text = "Weekly Market Report\n\n"
    text += summary_text
    text += "\n\nKOSPI Top 20:\n"
    text += kospi_notice_text
    text += _to_text_table(kospi_df, "No KOSPI rows available.")
    text += "\n\nS&P 500 Top 20:\n"
    text += _to_text_table(sp_df, "No S&P 500 rows available.")
    text += "".join(alert_sections_text)
    return subject, text, html


def build_report():
    kospi_df, kospi_status, sp_df, alert_results, alert_statuses = collect_report_inputs()
    return render_report(kospi_df, kospi_status, sp_df, alert_results, alert_statuses)
def send_email(cfg, subject, text, html):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["email_from"]
    msg["To"] = cfg["email_to"]
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(cfg["smtp_user"], cfg["smtp_pass"])
        recipients = [addr.strip() for addr in cfg["email_to"].split(",") if addr.strip()]
        server.send_message(msg, from_addr=cfg["email_from"], to_addrs=recipients)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate and optionally send the weekly market report.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the report and save local artifacts without sending email.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where report artifacts (.txt/.html/.json) are saved.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live network fetches and build the report from cached data only.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    set_offline_mode(args.offline)
    kospi_df, kospi_status, sp_df, alert_results, alert_statuses = collect_report_inputs()
    report_date = dt.datetime.now().strftime("%Y-%m-%d")
    history_paths = save_market_history(report_date, kospi_df, sp_df, alert_results)
    subject, text, html = render_report(
        kospi_df,
        kospi_status,
        sp_df,
        alert_results,
        alert_statuses,
        report_date=report_date,
    )
    if args.dry_run:
        artifacts = save_report_artifacts(
            subject,
            text,
            html,
            output_dir=output_dir,
            metadata={
                "delivery_status": "dry_run",
                "history_kospi_path": str(history_paths["kospi"]),
                "history_sp500_path": str(history_paths["sp500"]),
                "history_alerts_path": str(history_paths["alerts"]),
            },
        )
        print(f"Dry run complete: {artifacts['html']}")
        return

    cfg = load_env()
    try:
        send_email(cfg, subject, text, html)
    except Exception as e:
        artifacts = save_report_artifacts(
            subject,
            text,
            html,
            output_dir=output_dir,
            metadata={
                "delivery_status": "send_failed",
                "delivery_error": str(e),
                "history_kospi_path": str(history_paths["kospi"]),
                "history_sp500_path": str(history_paths["sp500"]),
                "history_alerts_path": str(history_paths["alerts"]),
            },
        )
        print(f"Email send failed. Saved report locally: {artifacts['html']}")
        raise

    artifacts = save_report_artifacts(
        subject,
        text,
        html,
        output_dir=output_dir,
        metadata={
            "delivery_status": "sent",
            "history_kospi_path": str(history_paths["kospi"]),
            "history_sp500_path": str(history_paths["sp500"]),
            "history_alerts_path": str(history_paths["alerts"]),
        },
    )
    print(f"Email sent: {subject}")
    print(f"Saved report locally: {artifacts['html']}")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
