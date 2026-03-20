# Weekly Market Report

Weekly email report:
- KOSPI top 20 by market cap
- S&P 500 top 20 by market cap
- Weekly price change (%)
- Two-week momentum alert summary
- Local artifact save (`outputs/*.txt`, `outputs/*.html`, `outputs/*.json`)
- Time-series market history (`history/*.csv`)

## Setup (WSL)
1) Create venv:
   python3 -m venv /home/jack/trail\ and\ error/.venv
   source /home/jack/trail\ and\ error/.venv/bin/activate

2) Install deps:
   pip install -r requirements.txt

3) Configure env:
   cp .env.example .env
   edit .env

4) Run once:
   python weekly_report.py

5) Dry run without email send:
   python weekly_report.py --dry-run

6) Offline dry run from cache only:
   python weekly_report.py --dry-run --offline

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
- `run_weekly.sh` prefers `/home/jack/trail and error/.venv/bin/python` and falls back to `~/venvs/weekly_market_report/bin/python`.
- Every run saves local report artifacts under `weekly_market_report/outputs/`.
- Every run appends market history under `weekly_market_report/history/`.
- If SMTP send fails, the script still saves the generated report locally and exits with an error.
- `--offline` uses cached market data only, so it is the fastest way to verify report rendering when DNS or internet access is broken.

## History Files
- `history/kospi_top20_history.csv`: 실행 시점별 KOSPI top 20 종목의 주간 시작가, 종가, 변동률 누적
- `history/sp500_top20_history.csv`: 실행 시점별 S&P 500 top 20 종목의 주간 시작가, 종가, 변동률 누적
- `history/momentum_alert_history.csv`: 실행 시점별 momentum alert 평가 결과 누적
- 각 파일에는 `captured_at`, `report_date`가 포함되어 있어 비정기 실행이어도 시간 흐름대로 추적할 수 있다.

## Troubleshooting
- DNS or network failure:
  `Temporary failure in name resolution`, `Could not resolve host`, `DNSError`가 보이면 외부 데이터 소스나 SMTP 서버 DNS 해석이 막힌 상태다.
- Safe verification:
  메일 송신 없이 포맷과 fallback 동작만 확인하려면 `python weekly_report.py --dry-run`을 사용한다.
- Offline verification:
  외부 DNS가 깨졌다면 `python weekly_report.py --dry-run --offline`으로 캐시 기반 산출물만 즉시 생성할 수 있다.
- Output inspection:
  최근 생성된 `outputs/*.json`에서 `delivery_status`와 저장된 HTML/TXT 경로를 확인할 수 있다.
- History inspection:
  누적 주가 흐름은 `history/*.csv`를 열면 되고, 같은 종목을 `symbol` 기준으로 필터링하면 실행 시점별 변동률 변화를 볼 수 있다.
