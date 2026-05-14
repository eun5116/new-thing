# Patchnote

이 문서는 코드와 실험 접근의 변경 사항을 개발 관점에서 추적하기 위한 기록이다. 읽기 쉬운 진행 기록은 `reports/daily_log.md`를 기준으로 한다.

## 2026-05-12

### KRX 데이터 파이프라인

- KRX OpenAPI 기반 종목 일별매매정보 수집 흐름을 구축했다.
- KOSPI/KOSDAQ 종목 기본정보, 가격, 거래대금, 시가총액, 지수 데이터를 feature build 파이프라인에 연결했다.
- KRX raw cache와 processed parquet 분리를 유지해 재수집 비용을 줄이는 구조로 정리했다.
- train/valid/test split을 parquet 산출물로 고정해 이후 실험들이 같은 데이터 경계를 쓰도록 했다.

### Feature Engineering

- 기본 가격 feature:
  - `return_1d`, `return_5d`, `return_20d`
  - 이동평균 괴리율
  - 변동성
  - RSI
  - MACD
  - drawdown
  - 거래량 z-score
- KRX 추가 feature:
  - 거래대금 변화
  - 거래대금 z-score
  - 시가총액 변화
  - 시가총액 MA 괴리율
  - 거래대금/시가총액 비율
- 시장 feature:
  - KOSPI/KOSDAQ 시장 수익률
  - 시장 MA 괴리율
  - 시장 변동성
  - 시장 급락 flag
- 상대강도/추세 feature:
  - 시장 대비 초과수익률
  - 20일 상대강도
  - 60일 시장 대비 drawdown 차이
  - MA20/60 신호와 포지션
- 이벤트 feature:
  - 수동 이벤트 CSV를 `effective_date` 기준으로 결합
  - `event_recent_5d`, `event_recent_20d`로 이벤트 이후 반응 구간 표현

### Trading Environment

- `target_position` action mode를 실험의 중심으로 사용했다.
- MA20/60 기준 포지션에 PPO가 overlay를 더하는 `ma20_60_overlay` 접근을 추가해 비교했다.
- reward mode를 여러 방향으로 비교했다.
  - absolute return
  - excess return
  - risk-adjusted
  - drawdown budget
  - MA20/60 relative
  - MA20/60 drawdown hybrid

### Experiment Batch

- E014~E016:
  - MA20/60 relative reward를 단일 종목 중심으로 실험했다.
  - MA20/60 기준선을 reward에 직접 넣는 접근을 시작했다.
- E017:
  - multi-ticker 학습으로 전환했다.
  - 여러 종목을 하나의 PPO 정책으로 학습하는 구조를 검증했다.
- E018:
  - MA20/60 overlay action을 실험했다.
  - 평균 성과는 기준선과 비슷했지만 행동이 중립에 치우쳤다.
- E019:
  - overlay reward에 drawdown penalty를 섞었다.
  - 위험 구간에서 overlay를 낮추는 행동은 생겼지만 수익률이 낮았다.
- E020:
  - 이벤트 이후 반응 구간과 regime feature를 추가했다.
  - action이 다시 중립 overlay에 수렴했다.
- E021~E025:
  - drawdown penalty 완화, soft relative reward, overlay 폭 확대, target-position 전환, entropy 증가를 각각 실험했다.
  - overlay 계수 조정만으로는 충분하지 않다는 결론을 얻었다.

### Universe Expansion

- `select_krx_universe` 흐름을 이용해 KOSPI 30개 + KOSDAQ 20개 universe를 구성했다.
- 데이터 길이가 부족한 2개 종목을 제외하고 E027용 48종목 universe를 확정했다.
- KOSDAQ 종목을 포함하면서 KOSDAQ 지수 데이터를 2020년부터 추가 수집했다.

## 2026-05-13

### E027 48종목 Target-position PPO

- 추가 config:
  - `configs/KRX_E027_liquid48_target_hybrid.yaml`
- 모델:
  - `models/ppo_KRX_E027_liquid48_target_hybrid.zip`
- 접근:
  - 48종목 multi-ticker 환경에서 target-position action과 MA20/60 drawdown hybrid reward를 사용했다.
  - 150,000 timesteps 학습했다.
- 결과:
  - test 평균 수익률 `242.6%`
  - test 평균 Sharpe `2.52`
  - test 평균 MDD `-28.5%`
- 개발 판단:
  - E027은 공격형 정책이 아니라 중간 위험형 PPO 후보로 분류했다.

### E028 공격형 PPO

- 추가 config:
  - `configs/KRX_E028_liquid48_target_hybrid_aggressive.yaml`
- 모델:
  - `models/ppo_KRX_E028_liquid48_target_hybrid_aggressive.zip`
- 접근:
  - E027 대비 drawdown penalty를 낮추고 MA20/60 대비 성과 보상을 강화했다.
  - `drawdown_penalty`: `0.25` -> `0.10`
  - `ma_underperformance_penalty`: `1.0` -> `0.50`
  - `ma_outperformance_bonus`: `0.25` -> `0.50`
- 결과:
  - test 평균 수익률 `326.6%`
  - test 평균 Sharpe `3.25`
  - test 평균 MDD `-28.9%`
  - test 평균 target ratio `0.788`
- 개발 판단:
  - 현재 best PPO 후보로 채택했다.

### Policy Action Analysis

- PPO action trace를 날짜별로 재생해 context별 평균 target ratio를 산출했다.
- 생성 산출물:
  - `reports/ppo_*_action_trace.csv`
  - `reports/ppo_*_action_context_summary.csv`
  - `reports/ppo_*_action_context_aggregate.csv`
- 활용:
  - 단순 성과지표뿐 아니라 이벤트일, 시장 급락일, 상대강도 하위 구간, drawdown 구간에서 정책이 실제로 어떤 노출을 택하는지 확인했다.

### Gated Policy Evaluator

- 추가 파일:
  - `src/stock_rl/evaluate_gated_policy.py`
- 접근:
  - E027/E028 action trace를 candidate ratio로 재생했다.
  - Buy & Hold, MA20/60, E027, E028을 후보로 두고 rule-based gated policy를 평가했다.
  - 거래비용을 반영한 ratio replay 방식으로 비교했다.
- 산출물:
  - `reports/gated_policy_{train,valid,test}_metrics.csv`
  - `reports/gated_policy_{train,valid,test}_summary.csv`
  - `reports/gated_policy_{train,valid,test}_trace.csv`
  - `reports/regime_candidate_comparison_{train,valid,test}.csv`
- 결론:
  - 단순 gated rule은 E028 단독을 검증 기준으로 안정적으로 이기지 못했다.

### E029 추가 PPO

- 추가 config:
  - `configs/KRX_E029_liquid48_target_hybrid_more_aggressive.yaml`
- 모델:
  - `models/ppo_KRX_E029_liquid48_target_hybrid_more_aggressive.zip`
- 접근:
  - E028보다 reward를 더 공격적으로 조정했다.
  - `drawdown_penalty`: `0.05`
  - `ma_underperformance_penalty`: `0.25`
  - `ma_outperformance_bonus`: `0.75`
- 결과:
  - test 평균 수익률 `139.4%`
  - test 평균 MDD `-21.7%`
  - 평균 target ratio `0.519`
- 개발 판단:
  - reward를 공격적으로 바꿔도 정책은 더 방어적으로 수렴할 수 있음을 확인했다.

### Meta-policy Input Dataset

- 추가 파일:
  - `src/stock_rl/analyze_meta_policy_inputs.py`
  - `src/stock_rl/evaluate_monthly_meta_rules.py`
- 접근:
  - Buy & Hold, MA20/60, E028의 월별·종목별 candidate return dataset을 만들었다.
  - 월초 feature와 월별 winner를 연결해 meta-policy 입력 구조를 만들었다.
  - rule grid로 간단한 월별 선택 규칙을 검증했다.
- 산출물:
  - `reports/meta_policy_monthly_strategy_returns_{train,valid,test}.csv`
  - `reports/meta_policy_monthly_winner_summary_{train,valid,test}.csv`
  - `reports/meta_policy_feature_signal_{train,valid,test}.csv`
  - `reports/meta_policy_e028_failure_cases_{train,valid,test}.csv`
  - `reports/meta_policy_monthly_rule_*`
- 결론:
  - E028은 특정 추세/상대강도 구간에서만 유리하다.
  - meta-policy의 기본 후보는 Buy & Hold가 되어야 한다.

### Softmax Meta-policy

- 추가 파일:
  - `src/stock_rl/train_monthly_meta_policy.py`
- 접근:
  - 월초 feature로 Buy & Hold, MA20/60, E028 중 하나를 선택하는 softmax model을 구현했다.
  - label 기반 softmax와 return 기반 softmax를 둘 다 비교했다.
  - L2 정규화와 entropy 계수를 grid로 평가했다.
- 산출물:
  - `reports/meta_policy_softmax_candidate_valid.csv`
  - `reports/meta_policy_softmax_candidate_test.csv`
  - `reports/meta_policy_softmax_selected_predictions.csv`
  - `reports/meta_policy_softmax_selected_summary.csv`
- 결론:
  - valid에서는 개선됐지만 test에서는 Buy & Hold보다 낮아 채택하지 않았다.

### Walk-forward Meta-policy

- 추가 파일:
  - `src/stock_rl/evaluate_walk_forward_meta_policy.py`
- 접근:
  - train 월별 candidate return dataset까지 만든 뒤 walk-forward 구조로 평가했다.
  - 2020~2023 train 학습, 2024 valid 선택, 2025~2026 test 고정 평가를 적용했다.
- 산출물:
  - `reports/meta_policy_walk_forward_candidate_valid.csv`
  - `reports/meta_policy_walk_forward_candidate_test.csv`
  - `reports/meta_policy_walk_forward_selected_predictions.csv`
  - `reports/meta_policy_walk_forward_selected_summary.csv`
- 결론:
  - 선택 모델은 사실상 Buy & Hold에 수렴했다.
  - 현재 feature와 softmax 구조로는 E028 선택 구간을 안정적으로 잡지 못했다.

### Stop-loss / Re-entry Evaluator

- 추가 파일:
  - `src/stock_rl/evaluate_stop_reentry.py`
- 접근:
  - 15% 또는 20% 손절 후, MA20/60·상대강도·시장조건을 만족하면 재진입하는 rule-based 전략을 구현했다.
  - 거래비용 `0.1%`를 반영했다.
- 산출물:
  - `reports/stop_reentry_{train,valid,test}_metrics.csv`
  - `reports/stop_reentry_{train,valid,test}_summary.csv`
  - `reports/stop_reentry_{train,valid,test}_trace.csv`
- 결론:
  - 고정 손절은 단독 전략으로 채택하지 않는다.
  - 포트폴리오 MDD를 손절 비율로 직접 제한하지 못한다.
  - 추후에는 PPO target ratio를 줄이는 risk overlay로 쓰는 편이 적절하다.

### Long Trend Feature Patch

- 수정 파일:
  - `src/stock_rl/build_features.py`
  - `tests/test_features_and_env.py`
- 추가 feature:
  - `return_60d`
  - `return_120d`
  - `ma60_gap`
  - `ma120_gap`
  - `market_return_60d`
  - `market_return_120d`
  - `market_ma60_gap`
  - `market_ma120_gap`
- 구현 내용:
  - `FEATURE_COLUMNS`에 장기 feature 8개를 추가했다.
  - `MARKET_FEATURE_DEFAULTS`에도 시장 장기 feature 기본값을 추가했다.
  - `add_price_features`에서 종목 60/120일 수익률과 MA 괴리율을 계산했다.
  - `add_market_features`에서 시장 60/120일 수익률과 MA 괴리율을 계산했다.
  - 테스트 fixture 기간을 90영업일에서 180영업일로 늘려 120일 rolling feature 검증이 가능하게 했다.
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `14 passed`

### Price File Filtering Patch

- 수정 파일:
  - `src/stock_rl/build_features.py`
- 구현 내용:
  - `read_price_files(price_dir, tickers=None)` 형태로 ticker filter를 지원하게 했다.
  - config에 지정된 universe만 feature build 대상이 되도록 했다.
  - raw price directory에 이전 실험 종목 파일이 남아 있어도 현재 config universe로 feature가 제한된다.
- 이유:
  - KRX 확장 universe 실험을 반복하면서 raw price directory에 여러 실험의 가격 파일이 공존할 수 있다.
  - config universe와 feature build 대상이 어긋나면 학습 종목 수와 평가 결과가 섞일 수 있으므로 명시적으로 차단했다.

### KRX Daily Cache Patch

- 수정 파일:
  - `src/stock_rl/krx_openapi.py`
- 구현 내용:
  - KRX 일별 cache를 저장할 때 요청 ticker subset이 아니라 해당 market/day 전체 normalized frame을 저장하도록 바꿨다.
  - cache read 후 요청 ticker가 누락되어 있으면 해당 날짜 raw API를 다시 받아 cache를 갱신한다.
  - 최종 반환 직전에 wanted ticker set으로 필터링한다.
- 이유:
  - 날짜별 cache가 특정 요청 subset으로 저장되면, 이후 다른 universe를 요청할 때 cache hit가 발생해도 필요한 ticker가 빠질 수 있다.
  - 전체 normalized cache + 반환 시 필터링 구조가 universe 변경에 더 안전하다.

### E030 Long Trend PPO

- 추가 config:
  - `configs/KRX_E030_liquid48_long_trend_aggressive.yaml`
- 모델:
  - `models/ppo_KRX_E030_liquid48_long_trend_aggressive.zip`
- 접근:
  - E028 설정을 유지하되, 장기 trend feature 8개가 포함된 observation으로 새로 학습했다.
  - observation shape이 바뀌었으므로 기존 E028 model 재사용 없이 새 모델을 학습했다.
- 결과:
  - test 평균 수익률 `273.3%`
  - test 평균 Sharpe `3.00`
  - test 평균 MDD `-26.4%`
  - 평균 target ratio `0.680`
- 결론:
  - 장기 feature는 MDD를 낮추는 신호로 작동했다.
  - 그러나 상승장 노출이 줄어 E028을 대체하지 못했다.

### E031 Long Trend More Aggressive PPO

- 추가 config:
  - `configs/KRX_E031_liquid48_long_trend_more_aggressive.yaml`
- 모델:
  - `models/ppo_KRX_E031_liquid48_long_trend_more_aggressive.zip`
- 접근:
  - E030과 같은 feature set에서 reward만 더 공격적으로 조정했다.
  - `drawdown_penalty`: `0.10` -> `0.02`
  - `ma_underperformance_penalty`: `0.50` -> `0.25`
  - `ma_outperformance_bonus`: `0.50` -> `1.00`
- 결과:
  - test 평균 수익률 `138.6%`
  - test 평균 Sharpe `2.57`
  - test 평균 MDD `-23.2%`
  - 평균 target ratio `0.527`
- 결론:
  - 보상을 더 공격적으로 바꿨지만 실제 정책은 더 방어적으로 수렴했다.
  - 단순 reward weight 조정만으로 평균 노출을 원하는 수준으로 만들기 어렵다.

### Current Technical Direction

- 현재 best PPO 후보:
  - `ppo_KRX_E028_liquid48_target_hybrid_aggressive`
- 유지할 feature:
  - 장기 trend feature 8개는 정보가 있으므로 유지할 가치가 있다.
- 보류할 접근:
  - feature를 무작정 더 늘리기
  - reward weight만 더 극단적으로 조정하기
  - 단순 hand-written gated rule을 계속 미세조정하기
- 다음 후보 접근:
  - 평균 target ratio 하한
  - action mask 또는 action prior
  - E028을 baseline으로 두고 risk overlay만 별도 제어
  - meta-policy는 더 강한 regime feature가 생긴 뒤 재검토

### Verification

- 테스트 명령:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
- 최종 결과:
  - `14 passed`

## 2026-05-13 Regime Exposure Cap Grid

- 수정 파일:
  - `src/stock_rl/evaluate_regime_exposure_cap.py`
- 구현 내용:
  - E032 replay regime cap 후보에 `strong_trend_full_else080`, `strong_trend_full_else075`, `cap_weak80_drop85`, `cap_weak75_drop85`를 추가했다.
  - valid/test summary, metrics, trace를 재생성했다.
- 결과:
  - valid 최선은 `strong_trend_full_else070`
  - test `strong_trend_full_else070`: 수익률 `380.0%`, Sharpe `3.39`, MDD `-29.2%`
  - test `strong_trend_full_else080`: 수익률 `420.6%`, Sharpe `3.56`, MDD `-30.9%`
- 결론:
  - 기본 후보는 `strong_trend_full_else070`
  - 공격 후보는 `strong_trend_full_else080`

### Regime Cap Breakdown And Stock Exception

- 추가 파일:
  - `src/stock_rl/analyze_regime_exposure_cap.py`
- 수정 파일:
  - `src/stock_rl/evaluate_regime_exposure_cap.py`
- 구현 내용:
  - regime cap trace/metrics에서 월별 summary, 후보-vs-baseline 월별 비교, 종목별 비교 CSV를 생성하는 분석기를 추가했다.
  - 강한 종목 예외 rule인 `strong_trend_or_stock_full_else070/080`, `strong_trend_or_very_strong_stock_full_else070/080`을 추가했다.
- 결과:
  - `strong_trend_full_else070`은 test에서 47/48 종목의 MDD를 개선하지만 수익률은 uncapped보다 낮다.
  - `strong_trend_or_very_strong_stock_full_else070`은 test 수익률을 `380.0%`에서 `383.4%`로 소폭 올렸지만 MDD가 `-29.2%`에서 `-29.4%`로 악화됐다.
- 결론:
  - 기본 후보는 계속 `strong_trend_full_else070`
  - stock-level 예외는 개선 폭 대비 복잡도가 높아 채택 보류

### Candidate Scorecard

- 추가 파일:
  - `src/stock_rl/build_candidate_scorecard.py`
- 구현 내용:
  - 기존 PPO 후보 CSV와 regime exposure cap summary를 합쳐 valid/test scorecard를 생성한다.
  - Buy & Hold, MA20/60, E028/E030/E032/E034 PPO, E032 replay cap 후보를 한 표에서 비교한다.
- 산출물:
  - `reports/candidate_scorecard_valid.csv`
  - `reports/candidate_scorecard_test.csv`
  - `reports/candidate_scorecard_valid_test.csv`
- 결론:
  - valid 1위는 `E032_replay_strong_trend_full_else070`
  - 기본 후보를 `strong_trend_full_else070`으로 유지한다.

### Current Target Generator

- 추가 파일:
  - `src/stock_rl/generate_current_targets.py`
- 구현 내용:
  - 저장된 E032 PPO 모델과 선택된 regime cap rule로 최신 feature 행의 종목별 target ratio를 산출한다.
  - optional positions CSV가 있으면 현재 position ratio를 observation에 반영할 수 있다.
- 산출물:
  - `reports/current_targets_20260511_strong_trend_full_else070.csv`
- 결과:
  - raw 가격은 2026-05-11까지 수집됐다.
  - `daily_features`는 추론용 최신 행을 보존하고, train/valid/test만 `target_return_1d`가 있는 행으로 제한한다.
  - 평균 target ratio는 `0.835`이다.

### Trading Sheet

- 추가 파일:
  - `src/stock_rl/build_trading_sheet.py`
- 구현 내용:
  - 최신 `current_targets` 파일에 KRX 종목명, 시장, 현재가, 거래대금, 시총, momentum/drawdown 정보를 붙인다.
  - 운용자가 보기 쉬운 CSV와 Markdown 요약을 생성한다.
- 산출물:
  - `reports/trading_sheet_20260511_strong_trend_full_else070.csv`
  - `reports/trading_sheet_20260511_strong_trend_full_else070.md`

## 2026-05-14 Incremental Price Collection

- 수정 파일:
  - `src/stock_rl/collect_prices.py`
  - `daily_log.md`
- 구현 내용:
  - `collect_prices(config_path, start=None, end=None)` 형태로 수집 기간 override를 지원한다.
  - CLI에 `--start`, `--end` 옵션을 추가했다.
  - 부분 기간 수집 결과를 기존 ticker별 parquet/CSV와 병합하고, `ticker/date` 기준 중복은 최신 행으로 정리한다.
  - KRX 증분 기간에 신규 행이 없으면 기존 가격 파일이 모두 있을 때 정상 종료한다.
- 이유:
  - 운용표 갱신은 최신 며칠만 확인하면 되는데 기존 수집기는 config `market.start`부터 전체 기간을 다시 순회했다.
  - 매일 target ratio를 갱신할 때 전체 수집 비용을 줄이고, 아직 KRX가 최신 일자를 반환하지 않는 경우도 실패로 보지 않게 했다.
- 확인:
  - 2026-05-12~2026-05-14 증분 수집에서는 KRX 신규 행이 없었다.
  - `daily_features.parquet` 최신 유효 거래일은 `2026-05-11`로 유지됐다.
  - `reports/current_targets_20260511_strong_trend_full_else070.csv`와 `reports/trading_sheet_20260511_strong_trend_full_else070.md`를 재생성했다.
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `16 passed`

### Daily Target Update Entrypoint

- 추가 파일:
  - `src/stock_rl/update_daily_targets.py`
- 수정 파일:
  - `src/stock_rl/collect_krx_reference.py`
  - `src/stock_rl/generate_current_targets.py`
  - `tests/test_features_and_env.py`
  - `README.md`
  - `daily_log.md`
- 구현 내용:
  - `update_daily_targets` entrypoint를 추가해 증분 가격 수집, feature build, current target 생성, trading sheet 생성을 한 번에 실행하게 했다.
  - 기존 `daily_features.parquet` 최신일 다음 날을 증분 수집 시작일로 자동 추정한다.
  - KOSPI/KOSDAQ 지수 증분 수집도 daily update에 포함했다.
  - 지수 부분 수집 결과는 기존 parquet와 병합하고, 신규 행이 없으면 기존 파일을 유지한다.
  - `--start`, `--end`, `--skip-collect`, `--skip-indices`, `--skip-features` 옵션을 지원한다.
  - 실행 요약에 `stale_count`, `max_feature_lag_days`를 포함해 최신 feature가 없는 종목을 바로 확인할 수 있게 했다.
  - 실행 요약에 `index_latest`를 포함해 지수 최신일도 확인할 수 있게 했다.
  - `generate_current_targets` 실행 전 `MPLCONFIGDIR` 기본값을 `/tmp/stock_rl_matplotlib`로 설정해 matplotlib cache 경고를 제거했다.
- 사용 예:
  - `PYTHONPATH=src .venv/bin/python -m stock_rl.update_daily_targets --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml --rule strong_trend_full_else070`
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `18 passed`
  - `PYTHONPATH=src .venv/bin/python -m stock_rl.update_daily_targets --config configs/KRX_E032_liquid48_long_trend_min_exposure.yaml --rule strong_trend_full_else070 --skip-collect --skip-features`
  - 추가 테스트 후 결과: `19 passed`

### Recent Empty Cache Refresh

- 수정 파일:
  - `src/stock_rl/krx_openapi.py`
  - `daily_log.md`
- 구현 내용:
  - `fetch_stock_prices`와 `fetch_index_history`가 최근 7일 안의 empty daily cache를 만나면 KRX API를 다시 조회하도록 했다.
  - 장중/장마감 전 빈 응답이 cache에 저장된 뒤, 장마감 후에도 빈 cache 때문에 신규 가격을 놓치는 문제를 막는다.
- 확인:
  - 장마감 후 daily update 재실행으로 `2026-05-13` 종목 가격과 지수 데이터를 수집했다.
  - `reports/current_targets_20260513_strong_trend_full_else070.csv`
  - `reports/trading_sheet_20260513_strong_trend_full_else070.md`
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `19 passed`

### Target Change Report

- 추가 파일:
  - `src/stock_rl/build_target_change_report.py`
- 수정 파일:
  - `src/stock_rl/update_daily_targets.py`
  - `tests/test_features_and_env.py`
  - `README.md`
  - `daily_log.md`
- 구현 내용:
  - 최신 `current_targets`와 직전 `current_targets`를 비교해 target exposure 변화표를 생성한다.
  - `target_delta_pct`와 `rebalance_action`을 산출한다.
  - 기본 threshold는 `5.0%p`이며, 이 이상 증가하면 `increase`, 이 이하 감소하면 `reduce`, 나머지는 `hold`로 분류한다.
  - daily update 실행 시 target/sheet와 함께 change report도 자동 생성한다.
- 산출물:
  - `reports/target_changes_20260513_strong_trend_full_else070.csv`
  - `reports/target_changes_20260513_strong_trend_full_else070.md`
- 결과:
  - 2026-05-11 대비 2026-05-13: increase 11종목, reduce 7종목, hold 30종목
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `20 passed`

### Collection State And Market-specific Starts

- 추가 파일:
  - `src/stock_rl/collection_state.py`
- 수정 파일:
  - `src/stock_rl/collect_prices.py`
  - `src/stock_rl/collect_krx_reference.py`
  - `src/stock_rl/krx_openapi.py`
  - `src/stock_rl/update_daily_targets.py`
  - `tests/test_features_and_env.py`
  - `README.md`
  - `daily_log.md`
- 구현 내용:
  - raw 종목 가격 parquet의 시장별 최신일을 읽고 KOSPI/KOSDAQ 각각 `latest + 1`부터 수집한다.
  - raw 지수 parquet도 시장별 최신일을 읽고 각각 `latest + 1`부터 수집한다.
  - KRX empty response manifest를 `data_krx/raw/collection_state.json`에 기록한다.
  - 같은 `kind/market/date` empty response는 기본 60분 TTL 안에서 재조회하지 않는다.
  - daily update 출력에 `stock_starts`, `index_starts`를 추가했다.
- 확인:
  - 2026-05-13 데이터 수집 이후 daily update 시작일은 `KOSPI=2026-05-14`, `KOSDAQ=2026-05-14`로 잡힌다.
  - 2026-05-14 빈 응답은 manifest에 기록된다.
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `23 passed`

### Portfolio Allocator Backtest

- 추가 파일:
  - `src/stock_rl/backtest_portfolio_allocator.py`
- 수정 파일:
  - `tests/test_features_and_env.py`
  - `README.md`
  - `daily_log.md`
- 구현 내용:
  - 종목별 E032 target ratio를 계좌형 basket으로 변환해 일자별 portfolio return을 재생한다.
  - 공통 설정은 상위 12종목, 종목당 최대 20%, 총 주식 노출 90%, 편도 거래비용 0.15%, 주 1회 리밸런싱이다.
  - 비교 전략은 전체 universe Buy & Hold, MA20/60 basket, E032 target basket이다.
  - turnover, total cost, holdings, gross exposure를 함께 기록한다.
- 산출물:
  - `reports/portfolio_allocator_valid_metrics.csv`
  - `reports/portfolio_allocator_valid_trace.csv`
  - `reports/portfolio_allocator_valid_allocations.csv`
  - `reports/portfolio_allocator_valid_report.md`
  - `reports/portfolio_allocator_test_metrics.csv`
  - `reports/portfolio_allocator_test_trace.csv`
  - `reports/portfolio_allocator_test_allocations.csv`
  - `reports/portfolio_allocator_test_report.md`
- 결과:
  - valid 1위: E032 target basket, 수익률 21.0%, Sharpe 0.76, MDD -24.9%
  - test 1위: MA20/60 basket, 수익률 922.2%, Sharpe 10.69, MDD -22.1%
  - test E032 target basket: 수익률 687.0%, Sharpe 8.62, MDD -22.6%
- 검증:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `24 passed`

### Operations Guide

- 추가 파일:
  - `docs/operations_guide.md`
- 수정 파일:
  - `README.md`
  - `daily_log.md`
- 구현 내용:
  - 매일 장마감 후 실행하는 daily update 흐름을 문서화했다.
  - `trading_sheet`, `current_targets`, `target_changes` 해석법을 정리했다.
  - portfolio allocator backtest 실행법과 현재 결과를 정리했다.
  - KRX API 수집 상태, empty response TTL, 수동 재생성 명령, 문제 해결 절차를 추가했다.
