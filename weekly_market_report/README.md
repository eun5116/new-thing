 # Weekly Market Report

  Weekly email report:
  - KOSPI top 10 by market cap
  - S&P 500 top 20 by market cap
  - Weekly price change (%)

  ## Setup (WSL)
  1) Create venv:
     python3 -m venv ~/venvs/weekly_market_report
     source ~/venvs/weekly_market_report/bin/activate

  2) Install deps:
     pip install -r requirements.txt

  3) Configure env:
     cp .env.example .env
     edit .env

  4) Run once:
     python scripts/weekly_report.py

  ## Cron (weekly)
  Example: every Monday 08:00
  crontab -e

  Add:
  0 8 * * 1 /home/jack/trail\ and\ error/weekly_market_report/scripts/run_weekly.sh >> /home/jack/trail\ and\ error/weekly_market_report/weekly.log
  2>&1

  ## Notes
  - KOSPI uses pykrx market data.
  - S&P 500 uses Wikipedia tickers + yfinance for market cap and prices.

  ———