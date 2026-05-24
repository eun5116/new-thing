# stock_rl_project

한국 주식/ETF의 일봉 가격, 기술지표, 정책/뉴스 이벤트, 거시지표를 거래일 기준으로 정렬해 강화학습용 데이터셋과 매매 환경을 만드는 프로젝트입니다.

전체 운용 절차는 [Stock RL 운용 사용 설명서](docs/operations_guide.md)를 기준으로 확인합니다.

## 1차 목표

첫 버전은 뉴스 텍스트를 직접 읽히지 않습니다. 먼저 가격 기반 feature와 단순 이벤트 더미를 합쳐 누수 없는 일봉 테이블을 만들고, Gymnasium 환경에서 매수/보유/매도 행동을 검증합니다.

```text
raw prices -> daily features -> train/valid/test parquet -> trading env -> PPO/A2C/DQN
```

## 폴더 구조

```text
data/
  raw/
    prices/          # yfinance/KRX 등 원천 가격 CSV
    events/          # 정책/뉴스/공시 이벤트 CSV
    macro/           # ECOS/FRED 등 거시 CSV
    disclosures/     # DART 등 공시 원천
  processed/         # daily_features, train, valid, test parquet
src/stock_rl/
  collect_prices.py  # yfinance/KRX OpenAPI 기반 가격 수집
  build_features.py  # 가격 feature + 이벤트 더미 결합
  trading_env.py     # Gymnasium 매매 환경
  evaluate.py        # Buy & Hold 및 전략 성과 지표
  train_rl.py        # Stable-Baselines3 PPO 학습 진입점
configs/config.yaml
tests/
```

## 빠른 시작

```bash
cd /home/jack/stock_rl_project
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m stock_rl.collect_prices --config configs/config.yaml
python -m stock_rl.build_features --config configs/config.yaml
python -m stock_rl.evaluate --features data/processed/test.parquet --ticker SPY
```

KRX OpenAPI 승인 키가 있으면 한국 종목은 공식 KRX 일별매매정보로 수집할 수 있습니다.

```bash
export KRX_AUTH_KEY=your_krx_openapi_auth_key
python -m stock_rl.collect_prices --config configs/krx_kospi.yaml
python -m stock_rl.build_features --config configs/krx_kospi.yaml
```

PPO 학습은 CPU용 PyTorch와 `stable-baselines3`를 추가 설치한 뒤 실행합니다. GPU는 필요 없습니다.

```bash
pip install -r requirements-rl-cpu.txt
python -m stock_rl.train_rl --config configs/config.yaml
```

## 운용표 갱신

현재 기본 운용 후보는 E032 PPO와 `strong_trend_full_else070` cap rule입니다. 최신 KRX 가격과 지수를 증분 확인하고, feature/target/trading sheet를 한 번에 갱신하려면 아래 명령을 사용합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.update_daily_targets \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --rule strong_trend_full_else070
```

수집 시작일은 기존 `daily_features.parquet`의 최신일 다음 날로 자동 추정합니다. 필요하면 `--start YYYY-MM-DD`로 직접 지정할 수 있습니다. 출력의 `index_latest`, `stale`, `max_lag`를 보면 지수/종목 최신일이 어긋났는지 바로 확인할 수 있습니다.

최신 target이 생성되면 직전 target 파일과 비교한 리밸런싱 변화표도 함께 생성됩니다.

```text
reports/target_changes_YYYYMMDD_strong_trend_full_else070.csv
reports/target_changes_YYYYMMDD_strong_trend_full_else070.md
reports/target_changes_YYYYMMDD_strong_trend_full_else070.png
```

KRX가 아직 특정 일자의 데이터를 공개하지 않아 빈 응답을 반환하면 `data_krx/raw/collection_state.json`에 기록합니다. 같은 빈 응답은 기본 60분 동안 재조회하지 않아, 같은 날 반복 실행할 때 API 호출을 줄입니다.

## 포트폴리오 백테스트

종목별 target을 실제 계좌형 basket으로 바꿔 검증하려면 allocator 백테스트를 실행합니다. 기본 비교는 전체 universe Buy & Hold, MA20/60 상위 basket, E032 target 상위 basket입니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.backtest_portfolio_allocator \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --splits valid test \
  --top-n 12 \
  --gross-cap 0.90 \
  --max-weight 0.20 \
  --transaction-cost-pct 0.0015 \
  --rebalance-frequency weekly
```

## 실제 보유 포지션 반영

카카오페이증권 등에서 가져온 보유 내역은 아래 CSV 포맷으로 저장합니다.

```text
data_krx/raw/positions/current_positions.csv
```

최소 입력은 보유 종목과 수량만 있으면 됩니다.

```csv
ticker,quantity
005930,5
NVDA,2.415
```

`name`, `current_price`, `market_value`는 가능한 경우 로컬 가격 데이터의 최신 종가와 KRX reference에서 자동 보강됩니다. KRX ETF/ETN은 보유 분석 실행 시 KRX 증권상품 일별매매정보로 가격을 수집합니다. `avg_price`가 없으면 손익률은 0으로 계산됩니다.

리밸런싱 주문 후보표는 아래 명령으로 생성합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_rebalance_orders \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --positions data_krx/raw/positions/current_positions.csv \
  --rule strong_trend_full_else070 \
  --top-n 12 \
  --gross-cap 0.90 \
  --max-weight 0.20 \
  --min-order-amount 5000 \
  --cash 0
```

```text
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.csv
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.md
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.png
```

현재 모델 universe 밖 자산은 `out_of_universe`로 표시되며 target weight는 0으로 계산됩니다. 이는 자동 매도 지시가 아니라 모델이 평가하지 않는 자산이라는 의미입니다.

현재 보유종목 자체의 손익, 비중, 모델 universe 여부, 한국 주식 추세를 보려면 아래를 실행합니다.
미국 주식/글로벌 ETF는 yfinance 가격으로 추세와 drawdown을 붙입니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.analyze_positions \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --positions data_krx/raw/positions/current_positions.csv \
  --rule strong_trend_full_else070 \
  --krx-start 2025-01-01
```

```text
reports/current_position_analysis_YYYYMMDD.csv
reports/current_position_analysis_YYYYMMDD.md
reports/current_position_analysis_YYYYMMDD.png
```

분석 결과와 리밸런싱 후보를 합친 한 장짜리 의사결정표는 아래로 생성합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_portfolio_decision_sheet \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml
```

```text
reports/portfolio_decision_sheet_YYYYMMDD.csv
reports/portfolio_decision_sheet_YYYYMMDD.md
reports/portfolio_decision_sheet_YYYYMMDD.png
```

보고서 묶음 대시보드는 아래 명령으로 생성합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_report_dashboard \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml
```

```text
reports/report_dashboard_YYYYMMDD.png
```

## 주간 시장 리포트

기존 `weekly_market_report` 기능은 `stock_rl` 모듈로 통합했습니다. KOSPI 시가총액 상위 20종목, S&P 500 시가총액 상위 20종목, 2주 모멘텀 알림, PDF/HTML/TXT/JSON 산출물을 한 번에 생성합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.weekly_market_report --dry-run
```

외부 네트워크가 불안정하거나 캐시 기반 렌더링만 확인하려면 오프라인 모드로 실행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.weekly_market_report --dry-run --offline
```

간단히 실행하려면 스크립트를 사용할 수 있습니다.

```bash
scripts/run_weekly_market_report.sh --dry-run
```

메일 발송까지 사용하려면 루트의 `.env.example`을 참고해 `.env`를 채웁니다. Gmail SMTP를 쓰는 경우 `SMTP_PASS`는 일반 비밀번호가 아니라 16자리 앱 비밀번호여야 합니다.

주간 리포트 산출물은 아래에 저장됩니다.

```text
reports/weekly_market/
```

캐시와 누적 history는 모델 입력 후보로 재사용하기 쉽도록 아래에 저장됩니다.

```text
data_weekly_market/cache/
data_weekly_market/history/
```

모멘텀 알림 설정은 아래 파일에서 관리합니다.

```text
configs/weekly_market/kospi_alerts.yaml
configs/weekly_market/us_equities_alerts.yaml
```

이 리포트는 바로 매매 지시가 아니라, 주간 시장 흐름과 후보군을 RL 운용 판단에 붙이는 보조 입력입니다. 이후 `reports/research`, `reports/macro`, `reports/retrain`과 함께 확인합니다.

## 이벤트 CSV 스키마

`data/raw/events/events.csv`는 아래 컬럼을 사용합니다.

```csv
event_date,effective_date,ticker,event_id,event_type,event_score,source,title
2023-11-06,2023-11-06,KOSPI,short_ban_202311,policy,1.0,FSC,short selling ban
```

- `event_date`: 발표일 또는 최초 관측일
- `effective_date`: 모델 state에 반영할 거래일
- `ticker`: 특정 종목/ETF/지수. 전체 시장 이벤트는 `ALL`
- `event_type`: `policy`, `macro`, `disclosure`, `news`, `crisis` 등
- `event_score`: 숫자화한 강도. 단순 더미는 `1.0`

핵심 원칙은 `state_t`에는 t일 장 시작 전에 관측 가능한 정보만 넣고, 보상은 t 이후 가격 변화로 계산하는 것입니다.
