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
