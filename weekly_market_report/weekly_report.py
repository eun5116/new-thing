import os
import sys
import smtplib
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf
from pykrx import stock
from pathlib import Path
import json

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SP500_CACHE_PATH = CACHE_DIR / "sp500_top20.json"

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
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=10).text
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    return df["Symbol"].tolist()
def get_sp500_top20():
    # 1) cache (1주 유지)
    if SP500_CACHE_PATH.exists():
        try:
            cached = json.loads(SP500_CACHE_PATH.read_text())
            cached_date = cached.get("date")
            if cached_date:
                age_days = (dt.datetime.now() - dt.datetime.fromisoformat(cached_date)).days
                if age_days <= 7:
                    return cached.get("tickers", [])
        except Exception:
            pass
    # 2) fallback: calculate from Yahoo (slow)
    tickers = get_sp500_tickers()
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
    data = yf.download(
        tickers=" ".join(top20),
        period="1mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    rows = []
    for t in top20:
        try:
            series = data[t]["Close"].dropna()
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
    df = pd.DataFrame(rows, columns=["Ticker", "StartDate", "EndDate", "Start", "End", "ChangePct"])
    df["ChangePct"] = df["ChangePct"].round(2)
    return df
def get_kospi_top10_and_change(window=5):
    today = dt.datetime.now().strftime("%Y%m%d")
    # latest business day for top-10 market cap selection
    end_date = stock.get_nearest_business_day_in_a_week(today)
    lookup_start = (dt.datetime.now() - dt.timedelta(days=45)).strftime("%Y%m%d")
    # market caps for top 20
    caps = stock.get_market_cap_by_ticker(end_date, market="KOSPI")
    caps = caps.sort_values("시가총액", ascending=False).head(20)
    tickers = caps.index.tolist()
    rows = []
    for t in tickers:
        try:
            ohlcv = stock.get_market_ohlcv_by_date(lookup_start, end_date, t)
            if ohlcv.empty:
                continue
            close = ohlcv["종가"].dropna()
            if len(close) < window:
                continue
            week = close.tail(window)
            start = float(week.iloc[0])
            end = float(week.iloc[-1])
            change_pct = (end - start) / start * 100.0
            name = stock.get_market_ticker_name(t)
            start_date = pd.to_datetime(week.index[0]).strftime("%Y-%m-%d")
            end_date_row = pd.to_datetime(week.index[-1]).strftime("%Y-%m-%d")
            rows.append((name, t, start_date, end_date_row, start, end, change_pct))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["Name", "Ticker", "StartDate", "EndDate", "Start", "End", "ChangePct"])
    df["ChangePct"] = df["ChangePct"].round(2)
    return df
def build_report():
    kospi_df = get_kospi_top10_and_change()
    sp_top20 = get_sp500_top20()
    sp_df = get_sp500_weekly_change(sp_top20)
    now = dt.datetime.now().strftime("%Y-%m-%d")
    subject = f"Weekly Market Report - {now}"
    kospi_html = kospi_df.to_html(index=False)
    sp_html = sp_df.to_html(index=False)
    html = f"""
    <html>
    <body>
    <h2>Weekly Market Report ({now})</h2>
    <h3>KOSPI Top 20 by Market Cap</h3>
    {kospi_html}
    <h3>S&P 500 Top 20 by Market Cap</h3>
    {sp_html}
    </body>
    </html>
    """
    text = "Weekly Market Report\n\nKOSPI Top 20:\n"
    text += kospi_df.to_string(index=False)
    text += "\n\nS&P 500 Top 20:\n"
    text += sp_df.to_string(index=False)
    return subject, text, html
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
def main():
    cfg = load_env()
    subject, text, html = build_report()
    send_email(cfg, subject, text, html)
    print("Email sent:", subject)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
