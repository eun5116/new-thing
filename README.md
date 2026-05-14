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
