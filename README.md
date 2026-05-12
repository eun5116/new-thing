# stock_rl_project

한국 주식/ETF의 일봉 가격, 기술지표, 정책/뉴스 이벤트, 거시지표를 거래일 기준으로 정렬해 강화학습용 데이터셋과 매매 환경을 만드는 프로젝트입니다.

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
