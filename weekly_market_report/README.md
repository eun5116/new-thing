# Weekly Market Report

Weekly email report:
- KOSPI top 20 by market cap
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
   python weekly_report.py

## Cron (weekly, fixed time)
Example: every Monday 08:00 (Asia/Seoul)

1) Open crontab:
   crontab -e

2) Add:
   CRON_TZ=Asia/Seoul
   0 8 * * 1 /home/jack/trail\ and\ error/weekly_market_report/run_weekly.sh

## Notes
- KOSPI uses pykrx market data.
- S&P 500 uses Wikipedia tickers + yfinance for market cap and prices.
- Cron does not run missed jobs while your PC is off.
