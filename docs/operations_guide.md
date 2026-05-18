# Stock RL 운용 사용 설명서

이 문서는 `stock_rl_project`를 실제로 어떻게 쓰는지 정리한 운용 가이드다. 현재 기본 전략은 E032 PPO 모델과 `strong_trend_full_else070` risk cap rule이다.

## 1. 현재 상태

현재 프로젝트는 아래 흐름까지 동작한다.

```text
KRX 가격/지수 증분 수집
-> feature build
-> E032 PPO target ratio 생성
-> 운용용 trading sheet 생성
-> 직전 target 대비 변화표 생성
-> 포트폴리오 allocator 백테스트
```

현재 기본 운용 후보:

- model: `models/ppo_KRX_E032_liquid48_long_trend_min_exposure.zip`
- config: `configs/KRX_E032_liquid48_long_trend_min_exposure.yaml`
- rule: `strong_trend_full_else070`
- universe: KOSPI/KOSDAQ 48종목

주의할 점:

- 이 프로젝트는 매수/매도 추천 확정 시스템이 아니라, target ratio와 포트폴리오 후보를 계산하는 연구/운용 보조 도구다.
- `target_pct`는 상승 확률이 아니다. 모델과 risk cap이 허용하는 목표 노출이다.
- 최종 계좌 비중은 포트폴리오 allocator와 실제 보유 현황을 함께 봐야 한다.

## 2. 매일 장마감 후 실행

장마감 후에는 아래 명령 하나를 실행한다.

```bash
cd /home/jack/stock_rl_project
PYTHONPATH=src .venv/bin/python -m stock_rl.update_daily_targets \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --rule strong_trend_full_else070
```

이 명령이 하는 일:

1. KRX 종목 가격을 시장별로 증분 수집한다.
2. KOSPI/KOSDAQ 지수를 시장별로 증분 수집한다.
3. `daily_features`, `train`, `valid`, `test` parquet을 다시 만든다.
4. 최신 feature 기준 target ratio를 만든다.
5. 운용용 trading sheet를 만든다.
6. 직전 target과 비교한 target change report를 만든다.

정상 출력 예:

```text
collection: 2026-05-14..today
stock_starts: KOSDAQ=2026-05-14 KOSPI=2026-05-14
index_starts: KOSDAQ=2026-05-14 KOSPI=2026-05-14
price_files: 48
index_files: 2
index_latest: KOSDAQ=2026-05-13 KOSPI=2026-05-13
target: /home/jack/stock_rl_project/reports/current_targets_20260513_strong_trend_full_else070.csv
sheet_csv: /home/jack/stock_rl_project/reports/trading_sheet_20260513_strong_trend_full_else070.csv
sheet_markdown: /home/jack/stock_rl_project/reports/trading_sheet_20260513_strong_trend_full_else070.md
changes_csv: /home/jack/stock_rl_project/reports/target_changes_20260513_strong_trend_full_else070.csv
changes_markdown: /home/jack/stock_rl_project/reports/target_changes_20260513_strong_trend_full_else070.md
summary: as_of=2026-05-13 tickers=48 avg_target=83.8% full=13 capped=20 stale=0 max_lag=0d
```

## 3. 출력 해석

### `collection`

실제 수집을 시도한 전체 시작일이다. 내부적으로는 KOSPI/KOSDAQ별 시작일을 따로 쓴다.

### `stock_starts`

종목 가격 수집 시작일이다. 각 시장의 raw 가격 최신일 다음 날로 자동 계산된다.

예:

```text
stock_starts: KOSDAQ=2026-05-14 KOSPI=2026-05-14
```

### `index_starts`

지수 수집 시작일이다. 각 시장의 raw index 최신일 다음 날로 자동 계산된다.

### `index_latest`

수집 후 KOSPI/KOSDAQ 지수 파일의 최신일이다. 종목 가격 최신일과 지수 최신일이 크게 어긋나면 feature 해석에 주의한다.

### `stale`, `max_lag`

최신 target 생성 시 stale feature row가 있었는지 알려준다.

- `stale=0`, `max_lag=0d`: 48종목 모두 같은 최신일 feature를 사용했다.
- `stale>0`: 일부 종목은 최신일보다 과거 feature를 사용했다. 거래정지, 데이터 지연, API 누락 가능성이 있다.

## 4. 매일 확인할 파일

### 운용표

```text
reports/trading_sheet_YYYYMMDD_strong_trend_full_else070.md
reports/trading_sheet_YYYYMMDD_strong_trend_full_else070.csv
```

우선 Markdown을 읽으면 된다.

주요 섹션:

- `Target Summary`: 100%, 88%, 70% target 분포
- `Top Targets`: target과 momentum 기준 상위 후보
- `Capped Names`: strong trend 조건을 만족하지 못해 70%로 제한된 종목
- `Stale Feature Rows`: 최신 feature가 아닌 종목

### target 원본

```text
reports/current_targets_YYYYMMDD_strong_trend_full_else070.csv
```

모델 진단 컬럼까지 모두 포함한다. 사람이 바로 보기에는 trading sheet가 더 편하다.

중요 컬럼:

- `target_ratio`: cap 적용 후 목표 노출
- `raw_target_ratio`: PPO가 고른 원래 목표 노출
- `cap`: regime cap
- `cap_reason`: `strong_trend` 또는 `none`
- `return_20d`, `return_60d`, `drawdown_60d`

### 리밸런싱 변화표

```text
reports/target_changes_YYYYMMDD_strong_trend_full_else070.md
reports/target_changes_YYYYMMDD_strong_trend_full_else070.csv
```

직전 target과 최신 target을 비교한다.

주요 컬럼:

- `previous_target_pct`: 직전 목표 노출
- `target_pct`: 최신 목표 노출
- `target_delta_pct`: 변화폭
- `rebalance_action`: `increase`, `reduce`, `hold`

기본 기준은 5%p 이상 변하면 increase/reduce다.

## 5. 실제 운용 판단 흐름

현재 권장 흐름은 이렇다.

1. 장마감 후 `update_daily_targets` 실행
2. `trading_sheet_YYYYMMDD_*.md`에서 상위 후보 확인
3. `target_changes_YYYYMMDD_*.md`에서 늘릴 종목과 줄일 종목 확인
4. 포트폴리오 allocator 기준을 함께 참고
5. 실제 계좌의 보유비중과 비교해 최종 주문 결정

현재 백테스트 기준 allocator 설정:

- 상위 12종목
- 종목당 최대 20%
- 총 주식 노출 90%
- 거래비용 편도 0.15%
- 주 1회 리밸런싱

즉 trading sheet의 target이 100%라고 해서 계좌 100%를 한 종목에 넣는다는 뜻이 아니다. 계좌 allocator에서는 종목당 최대 20%, 전체 주식 노출 90%가 상한이다.

## 6. 포트폴리오 백테스트

종목별 target을 계좌형 basket으로 바꿔 검증하려면 아래를 실행한다.

```bash
cd /home/jack/stock_rl_project
PYTHONPATH=src .venv/bin/python -m stock_rl.backtest_portfolio_allocator \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --splits valid test \
  --top-n 12 \
  --gross-cap 0.90 \
  --max-weight 0.20 \
  --transaction-cost-pct 0.0015 \
  --rebalance-frequency weekly
```

생성 파일:

```text
reports/portfolio_allocator_valid_metrics.csv
reports/portfolio_allocator_valid_trace.csv
reports/portfolio_allocator_valid_allocations.csv
reports/portfolio_allocator_valid_report.md
reports/portfolio_allocator_test_metrics.csv
reports/portfolio_allocator_test_trace.csv
reports/portfolio_allocator_test_allocations.csv
reports/portfolio_allocator_test_report.md
```

현재 결과 요약:

| split | strategy | return | Sharpe | MDD |
| --- | --- | ---: | ---: | ---: |
| valid | E032 target basket | 21.0% | 0.76 | -24.9% |
| valid | MA20/60 basket | 14.5% | 0.53 | -29.0% |
| valid | Buy & Hold universe | 9.4% | 0.43 | -23.3% |
| test | MA20/60 basket | 922.2% | 10.69 | -22.1% |
| test | E032 target basket | 687.0% | 8.62 | -22.6% |
| test | Buy & Hold universe | 408.3% | 7.35 | -20.3% |

해석:

- valid에서는 E032 target basket이 가장 좋았다.
- test 강한 상승장에서는 MA20/60 basket이 가장 좋았다.
- E032 target basket은 Buy & Hold universe보다 좋지만, test에서는 MA20/60보다 약했다.
- 다음 연구 후보는 E032와 MA20/60 중 어느 basket을 선택할지 정하는 portfolio-level selector다.

## 7. 실제 보유 포지션 반영

카카오페이증권 등에서 가져온 보유 내역은 CSV로 저장한다.

현재 입력 파일:

```text
data_krx/raw/positions/current_positions.csv
```

필수 컬럼:

```csv
ticker,name,quantity,avg_price,current_price,market_value
```

최소로는 아래 두 컬럼만 있어도 된다.

```csv
ticker,market_value
```

현재 보유비중과 모델 allocator 목표비중을 비교하려면 아래 명령을 실행한다.

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

생성 파일:

```text
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.csv
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.md
reports/rebalance_orders_YYYYMMDD_strong_trend_full_else070.png
```

주의:

- 현재 모델 universe는 KRX 48종목이다.
- 삼성전자우, 국내 ETF, 미국 주식, 금 ETF 등 universe 밖 자산은 `out_of_universe`로 표시된다.
- `out_of_universe` 자산은 모델 기준 target weight가 0이다. 이것은 자동 매도 지시가 아니라, 현재 모델이 평가하지 않는 자산이라는 뜻이다.
- 실제 주문은 세금, 환전, 계좌 목적, 장기 보유 의도 등을 별도로 고려해야 한다.

현재 카카오페이 입력 기준 요약:

- 총 평가금액: 약 `2,518,650원`
- 모델 universe 보유: 삼성전자 1종목
- universe 밖 보유: 삼성전자우, 국내 ETF, 미국 주식, GLD 등
- allocator 기준 상위 KRX 종목을 사려면 universe 밖 자산을 줄여야 한다는 형태로 보고서가 나온다.

### 보유종목 자체 분석

현재 보유종목을 모델 universe 여부와 별개로 분석하려면 아래 명령을 실행한다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.analyze_positions \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --positions data_krx/raw/positions/current_positions.csv \
  --rule strong_trend_full_else070 \
  --krx-start 2025-01-01
```

생성 파일:

```text
reports/current_position_analysis_YYYYMMDD.csv
reports/current_position_analysis_YYYYMMDD.md
reports/current_position_analysis_YYYYMMDD.png
```

이 분석은 아래를 포함한다.

- 현재 보유비중
- 평가손익률
- 모델 universe 여부
- 한국 주식의 KRX reference 정보
- 한국 주식의 MA20/60 추세, 20일/60일 수익률, 60일 drawdown
- 미국/글로벌 자산의 yfinance 기반 MA20/60 추세, 20일/60일/120일 수익률, 60일 drawdown, 20일 변동성
- 모델 target이 있는 종목은 target ratio

현재 보유 한국 주식 분석:

| ticker | name | scope | pnl | trend | return_20d | drawdown_60d | model target |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 005930 | 삼성전자 | model universe | 34.9% | uptrend | 41.3% | -0.5% | 100.0% |
| 005935 | 삼성전자우 | KRX stock outside model | 6.1% | uptrend | 36.5% | -3.0% | 없음 |
| 001510 | SK증권 | KRX stock outside model | -19.7% | uptrend | 140.2% | -20.7% | 없음 |

현재 보유 미국/글로벌 자산 분석:

| ticker | name | weight | pnl | trend | return_20d | drawdown_60d | volatility_20d |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| AMD | AMD | 26.4% | 2.0% | uptrend | 62.1% | -1.7% | 98.2% |
| GLD | SPDR Gold Shares | 25.5% | 2.8% | downtrend | -2.3% | -12.3% | 21.7% |
| NVDA | 엔비디아 | 13.4% | 16.6% | uptrend | 19.0% | 0.0% | 40.6% |
| COP | 코노코필립스 | 7.0% | -10.6% | downtrend | -1.8% | -10.8% | 37.0% |
| IONQ | 아이온큐 | 3.3% | 2.3% | uptrend | 27.4% | 0.0% | 91.1% |
| DVN | 데번 에너지 | 2.8% | -0.1% | uptrend | 2.4% | -10.0% | 43.5% |
| QUBT | 퀀텀 컴퓨팅 | 0.7% | -18.4% | uptrend | 26.5% | 0.0% | 101.0% |

미국/글로벌 자산은 E032 target을 받지 않는다. 현재는 trend/risk 분석만 한다.

### 한 장짜리 의사결정표

보유종목 분석과 리밸런싱 주문 후보를 합쳐 action label을 보려면 아래를 실행한다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_portfolio_decision_sheet \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml
```

생성 파일:

```text
reports/portfolio_decision_sheet_YYYYMMDD.csv
reports/portfolio_decision_sheet_YYYYMMDD.md
```

현재 label 의미:

- `trim_candidate`: 비중이 크거나 downtrend/손실/고변동성 때문에 축소 검토
- `trim_to_allocator`: 모델 allocator 목표비중보다 현재 비중이 높음
- `keep_watch`: 유지하되 관찰
- `speculative_watch`: 고변동성 관찰
- `manual_review`: ETF/unmapped 등 모델 밖이라 수동 검토
- `watch_or_trim`: 추세는 살아 있지만 drawdown/손실이 커서 관찰 또는 축소 검토

### 포트폴리오 정책 시트

KRX 48종목 모델 밖의 미국 주식, ETF, 우선주, 테마 종목은 별도 정책으로 비중을 점검한다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_portfolio_policy_sheet \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --policy configs/portfolio_policy.yaml
```

생성 파일:

```text
reports/portfolio_policy_sheet_YYYYMMDD.csv
reports/portfolio_policy_sheet_YYYYMMDD.md
```

이 시트는 자동 주문표가 아니라 자산군별 총 비중과 종목별 최대 비중을 확인하는 risk overlay다. E032 target은 KRX 모델 종목에만 적용하고, 미국 주식/ETF/모델 밖 KRX 종목은 `configs/portfolio_policy.yaml`의 그룹 한도로 관리한다.

## 8. 데이터 수집 구조

KRX OpenAPI는 일별 `basDd` 기준으로 데이터를 준다. 프로젝트는 API 호출을 줄이기 위해 아래를 적용한다.

- raw 종목 가격 parquet의 시장별 최신일을 읽는다.
- raw 지수 parquet의 시장별 최신일을 읽는다.
- KOSPI/KOSDAQ 각각 최신일 다음 날부터만 수집한다.
- KRX 빈 응답은 `data_krx/raw/collection_state.json`에 기록한다.
- 같은 `stock/index + market + date` 빈 응답은 기본 60분 동안 재조회하지 않는다.

상태 파일:

```text
data_krx/raw/collection_state.json
```

예:

```json
{
  "empty_responses": {
    "stock:KOSPI:20260514": "2026-05-14T14:32:26.819220+00:00",
    "index:KOSDAQ:20260514": "2026-05-14T14:32:27.083629+00:00"
  }
}
```

## 9. 수동 실행 명령

### 가격만 증분 수집

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.collect_prices \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --start 2026-05-14
```

보통은 직접 실행할 필요가 없다. `update_daily_targets`가 내부에서 처리한다.

### feature만 재생성

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_features \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml
```

### target만 재생성

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.generate_current_targets \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --rule strong_trend_full_else070
```

### trading sheet만 재생성

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_trading_sheet \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --rule strong_trend_full_else070
```

### target 변화표만 재생성

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.build_target_change_report \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml \
  --rule strong_trend_full_else070
```

## 10. 학습과 재학습

현재 E032 모델은 이미 학습된 상태다.

```text
models/ppo_KRX_E032_liquid48_long_trend_min_exposure.zip
```

새로 학습하려면 아래 명령을 쓴다.

```bash
PYTHONPATH=src .venv/bin/python -m stock_rl.train_rl \
  --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml
```

다만 지금 단계에서는 무조건 재학습보다 포트폴리오 allocator와 selector 실험이 우선이다.

재학습을 검토할 때 필요한 것:

- train/valid/test 기간 재정의
- 기존 E032와 새 모델의 동일 조건 비교
- portfolio allocator 기준 재평가
- test 성과뿐 아니라 valid/test 일관성 확인

## 11. 테스트

코드 변경 후에는 항상 아래를 실행한다.

```bash
cd /home/jack/stock_rl_project
PYTHONPATH=src .venv/bin/python -m pytest -q
```

현재 기준:

```text
24 passed
```

## 12. 문제 해결

### 장마감 후에도 최신일이 안 바뀐다

가능한 원인:

- KRX API가 아직 해당 일자 데이터를 공개하지 않았다.
- 해당 일자가 휴장일이다.
- 빈 응답 TTL 때문에 같은 날 재조회가 skip됐다.

확인할 것:

```text
stock_starts
index_starts
index_latest
summary: as_of, stale, max_lag
data_krx/raw/collection_state.json
```

TTL을 무시하고 확인하고 싶으면 `data_krx/raw/collection_state.json`의 해당 empty key를 지운 뒤 다시 실행한다.

### `stale > 0`이 나온다

일부 종목의 최신 feature 날짜가 뒤처진 것이다.

확인할 파일:

```text
reports/trading_sheet_YYYYMMDD_strong_trend_full_else070.md
```

`Stale Feature Rows` 섹션에서 어떤 종목인지 확인한다.

### target은 높은데 allocator에서는 비중이 낮다

정상이다. `target_pct`는 종목 단위 신호이고, allocator는 아래 제약을 다시 적용한다.

- 상위 12종목
- 종목당 최대 20%
- 총 주식 노출 90%

### MA20/60이 E032보다 좋아 보인다

현재 test 구간에서는 맞다. 강한 상승장에서는 MA20/60 basket이 E032보다 강했다. 하지만 valid에서는 E032가 좋았다. 따라서 다음 연구 과제는 “어느 regime에서 어느 basket을 쓸지”를 정하는 selector다.

### 미국 주식과 ETF가 전부 매도로 나온다

현재 모델 universe가 KRX 48종목이기 때문이다. `out_of_universe`는 “모델이 목표비중을 계산하지 않는 자산”이라는 의미다. 미국 주식/ETF까지 함께 운용하려면 별도 universe와 데이터 파이프라인을 추가해야 한다.

## 13. 다음 개발 후보

우선순위는 아래 순서가 좋다.

1. 실제 보유 포지션 CSV 입력
2. 현재 보유비중 대비 주문 필요 비중 산출
3. E032 vs MA20/60 portfolio-level selector
4. 리밸런싱 주기와 top_n grid search
5. 재학습 walk-forward 실험
