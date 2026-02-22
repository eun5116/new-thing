  import os
  import sys
  import smtplib
  import datetime as dt
  from email.mime.multipart import MIMEMultipart
  from email.mime.text import MIMEText

  import pandas as pd
  import requests
  from dotenv import load_dotenv

  import yfinance as yf
  from pykrx import stock


  def load_env():
      load_dotenv()
      cfg = {
          "smtp_host": os.getenv("SMTP_HOST", ""),
          "smtp_port": int(os.getenv("SMTP_PORT", "587")),
          "smtp_user": os.getenv("SMTP_USER", ""),
          "smtp_pass": os.getenv("SMTP_PASS", ""),
          "email_to": os.getenv("EMAIL_TO", ""),
          "email_from": os.getenv("EMAIL_FROM") or os.getenv("SMTP_USER", ""),
          "tz": os.getenv("REPORT_TIMEZONE", "Asia/Seoul"),
      }
      missing = [k for k, v in cfg.items() if k in ["smtp_host", "smtp_user", "smtp_pass", "email_to"] and not v]
      if missing:
          raise RuntimeError("Missing env: " + ", ".join(missing))
      return cfg


  def get_sp500_tickers():
      url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
      tables = pd.read_html(url)
      df = tables[0]
      tickers = df["Symbol"].tolist()
      return tickers


  def get_sp500_top20():
      tickers = get_sp500_tickers()

      # Fetch market caps (slow but weekly ok)
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
      return top20


  def get_sp500_weekly_change(top20):
      data = yf.download(
          tickers=" ".join(top20),
          period="10d",
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
              if len(series) < 2:
                  continue
              start = series.iloc[0]
              end = series.iloc[-1]
              change_pct = (end - start) / start * 100.0
              rows.append((t, float(start), float(end), float(change_pct)))
          except Exception:
              continue

      df = pd.DataFrame(rows, columns=["Ticker", "Start", "End", "ChangePct"])
      df["ChangePct"] = df["ChangePct"].round(2)
      return df


  def get_kospi_top10_and_change():
      today = dt.datetime.now().strftime("%Y%m%d")

      # nearest business days
      end_date = stock.get_nearest_business_day_in_a_week(today)
      start_date = stock.get_nearest_business_day_in_a_week(today, prev=True)

      # market caps for top 10
      caps = stock.get_market_cap_by_ticker(end_date, market="KOSPI")
      caps = caps.sort_values("시가총액", ascending=False).head(10)
      tickers = caps.index.tolist()

      rows = []
      for t in tickers:
          try:
              ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, t)
              if ohlcv.empty:
                  continue
              start = float(ohlcv["종가"].iloc[0])
              end = float(ohlcv["종가"].iloc[-1])
              change_pct = (end - start) / start * 100.0
              name = stock.get_market_ticker_name(t)
              rows.append((name, t, start, end, change_pct))
          except Exception:
              continue

      df = pd.DataFrame(rows, columns=["Name", "Ticker", "Start", "End", "ChangePct"])
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
      <h3>KOSPI Top 10 by Market Cap</h3>
      {kospi_html}
      <h3>S&P 500 Top 20 by Market Cap</h3>
      {sp_html}
      </body>
      </html>
      """

      text = "Weekly Market Report\n\nKOSPI Top 10:\n"
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
          server.starttls()


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