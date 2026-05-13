# Daily Log

이 문서는 프로젝트 진행을 날짜별로 읽기 쉽게 남기는 로그다. 세부 숫자와 CSV 산출물은 각 실험 보고서를 기준으로 하고, 여기서는 하루 동안 어떤 판단을 했고 왜 다음 방향이 바뀌었는지에 집중한다.

## 2026-05-12

### 출발점

처음 목표는 KRX 데이터를 이용해 한국 주식 RL 매매 환경을 만들고, 단순히 수익률만 높은 모델이 아니라 drawdown budget을 의식하는 전략을 찾는 것이었다.

초기 기준은 대략 이랬다.

- 허용 가능한 MDD는 대략 `-20%` 안팎으로 보고 싶다.
- 하지만 MDD만 낮추면 현금 보유 전략으로 무너질 수 있으니, 수익률과 Sharpe도 같이 봐야 한다.
- 단일 종목에서 시작하되, 최종적으로는 여러 종목에 일반화되는 정책이 필요하다.

### KRX 데이터 파이프라인 정리

KRX OpenAPI 기반 수집 흐름을 정리했다. 단순 yfinance 데이터가 아니라 KRX 공식 일별 매매정보와 지수 데이터를 기반으로 feature를 만들 수 있게 했다.

구성한 데이터는 크게 네 가지다.

- 종목 가격/거래량 데이터
- 거래대금, 시가총액 등 KRX 추가 필드
- KOSPI/KOSDAQ 지수 데이터
- 외부 시장 이벤트 데이터

이후 train/valid/test parquet을 만들고, Gymnasium 환경에서 PPO가 같은 feature set을 관측하도록 연결했다.

### 시장 feature와 이벤트 feature 추가

처음에는 가격 기반 기술지표만으로 학습했지만, 시장 전체 흐름과 사건 정보를 반영하지 못하면 한국 주식의 regime 변화를 놓칠 가능성이 컸다.

그래서 다음 feature들을 추가했다.

- 시장 수익률
- 시장 이동평균 괴리율
- 시장 변동성
- 시장 급락 여부
- 시장 대비 초과수익률
- 상대강도
- 이벤트 발생 이후 5일/20일 반응 구간

외부 이벤트는 뉴스 텍스트를 직접 넣는 방식은 아니고, 사람이 정리한 이벤트 테이블을 거래일 기준으로 반영했다. 핵심 원칙은 `state_t`에는 t일 장 시작 전에 알 수 있는 정보만 들어가야 한다는 점이었다.

### MA20/60 기준선 발견

단순 Buy & Hold만 기준선으로 두면 PPO가 잘하는지 판단하기 어렵다. 그래서 MA20/60 추세 전략을 기준선으로 추가했다.

중요한 발견은 MA20/60이 꽤 강한 기준선이라는 점이었다.

- 테스트 구간에서 Buy & Hold보다 수익률이 높았다.
- Sharpe도 더 높았다.
- MDD도 더 낮았다.

이후 실험의 방향은 “PPO가 MA20/60을 이기는가?”보다 더 구체적으로 바뀌었다. PPO가 MA20/60의 장점을 흡수하면서, 특정 위험 구간에서 더 나은 판단을 할 수 있는지가 핵심이 됐다.

### Drawdown budget 실험

사용자가 관심을 둔 핵심 제약은 MDD였다. 그래서 `drawdown_budget` reward와 `target_position` action을 섞은 실험들을 돌렸다.

이 과정에서 확인한 점:

- drawdown penalty를 강하게 주면 PPO가 노출을 줄여 MDD는 낮춘다.
- 하지만 수익률도 같이 크게 줄어든다.
- 손실을 피하는 행동은 배우지만, 상승장 재진입을 충분히 잘하지 못한다.

즉 MDD를 낮추는 것 자체는 가능했지만, “낮은 MDD + 높은 수익률” 조합은 단순 penalty만으로 나오지 않았다.

### Multi-ticker 학습과 overlay 실험

단일 종목에서 얻은 결과가 특정 종목에만 맞는지 확인하기 위해 여러 종목을 함께 학습하는 multi-ticker 환경으로 확장했다.

이후 두 가지 action 구조를 비교했다.

- target-position: action이 직접 목표 보유비중을 정한다.
- MA20/60 overlay: MA20/60 기준 포지션 위에 PPO가 노출을 조금 더하거나 줄인다.

overlay 방식은 해석하기 좋았지만, 실제로는 action이 중립값에 붕괴하는 문제가 있었다. E018/E020/E021/E022/E023/E025 여러 실험에서 대부분 action `2`, 즉 overlay `0.0` 근처로 수렴했다.

반대로 E019는 drawdown penalty 덕분에 위험 구간에서 노출을 낮추는 행동이 보였다. 다만 수익률이 낮았다.

이때 결론은 명확했다.

- overlay reward 계수만 계속 조정하는 것은 효율이 낮다.
- 공격형 후보, 방어형 후보, 기준선 전략을 따로 만든 뒤 regime별로 고르는 방식이 더 적절하다.

### 확장 universe 준비

다음 단계로 KRX 거래대금 기준 확장 universe를 만들었다.

- KOSPI 30개
- KOSDAQ 20개
- 총 50개 후보
- 데이터 길이가 짧은 2개를 제외하고 48개 학습 universe 확정

KOSDAQ 종목도 포함되면서 KOSDAQ 지수 데이터도 추가 수집했다. 이로써 KOSPI/KOSDAQ이 섞인 48종목 multi-ticker 학습 준비가 끝났다.

이날 마지막 결론은 E027 48종목 target-position hybrid PPO를 학습하는 것이었다.

관련 보고서:

- `reports/krx_pipeline_status_2026-05-12.md`
- `reports/krx_market_feature_update_2026-05-12.md`
- `reports/external_event_feature_update_2026-05-12.md`
- `reports/krx_drawdown_budget_result_2026-05-12.md`
- `reports/krx_ma20_60_relative_result_2026-05-12.md`
- `reports/krx_exploration_batch_result_2026-05-12.md`
- `reports/krx_expanded_universe_data_collection_2026-05-12.md`

## 2026-05-13

### E027: 48종목 첫 target-position 후보

E027은 48종목 universe에서 처음 학습한 target-position hybrid PPO였다.

결과는 방어형에 가까웠다.

- test 평균 수익률: `242.6%`
- test 평균 Sharpe: `2.52`
- test 평균 MDD: `-28.5%`

Buy & Hold보다 수익률이 높은 종목은 `1/48`뿐이었지만, MDD는 대부분 종목에서 더 낮았다. 따라서 E027은 최종 공격형 전략이 아니라, 중간 위험형 PPO 후보로 분류했다.

### E028: 더 공격적인 48종목 후보

E027이 너무 방어적이었기 때문에 E028을 만들었다.

주요 변경은 drawdown penalty를 낮추고, MA20/60 대비 성과 보상 쪽을 강화하는 것이었다.

결과는 좋아졌다.

- test 평균 수익률: `326.6%`
- test 평균 Sharpe: `3.25`
- test 평균 MDD: `-28.9%`
- test 평균 target ratio: `0.788`

E028은 E027보다 훨씬 공격적으로 상승장에 참여했다. MA20/60보다 수익률이 높은 종목도 E027 `17/48`에서 E028 `29/48`로 늘었다.

다만 Buy & Hold와 MA20/60의 평균 수익률을 완전히 넘지는 못했다. 그래서 E028은 현재 best PPO 후보이지만, 단독 최종 전략이라기보다 regime 선택 후보로 두는 것이 맞다고 판단했다.

### Gated policy와 E029

다음으로 `Buy & Hold`, `MA20/60`, `E027`, `E028`을 regime별로 고르는 규칙형 gated policy를 평가했다.

아이디어는 단순했다.

- 추세가 좋으면 E028 또는 MA20/60
- drawdown이 깊거나 시장 대비 부진하면 E027
- 이벤트일 또는 시장 급락 구간에서는 방어형 후보

하지만 결과는 좋지 않았다. 검증 구간에서 gated policy가 E028 단독을 안정적으로 이기지 못했다. 테스트에서 일부 규칙은 수익률이 약간 높았지만, valid에서 약했고 MDD도 커서 채택하기 어려웠다.

동시에 더 공격적인 reward를 준 E029도 학습했다. 그런데 의도와 달리 E029는 더 방어적으로 수렴했다.

- test 평균 수익률: `139.4%`
- test 평균 MDD: `-21.7%`
- 평균 target ratio: `0.519`

이 결과로 단순히 reward weight를 더 공격적으로 바꾼다고 실제 정책이 더 공격적으로 되는 것은 아니라는 점을 확인했다.

### Meta-policy 입력 분석

규칙형 gated policy가 충분하지 않았기 때문에, 월별·종목별로 어떤 전략이 이기는지 분석했다.

후보는 세 가지로 잡았다.

- Buy & Hold
- MA20/60
- E028

월별 winner 비중은 다음과 같았다.

- valid: MA20/60 `39.1%`, Buy & Hold `34.5%`, E028 `26.4%`
- test: Buy & Hold `51.3%`, MA20/60 `24.8%`, E028 `23.9%`

E028이 잘되는 구간은 대체로 상대강도와 추세가 좋은 구간이었다.

- `relative_strength_20d`가 높다.
- `relative_strength_regime`이 좋다.
- `ma20_60_position == 1`
- `ma20_60_gap`이 높다.

반대로 실패 구간은 시장 급락, 이벤트 직후, 깊은 drawdown이 많았다.

이 분석으로 meta-policy는 E028 중심이 아니라 Buy & Hold를 기본으로 두고 일부 구간에서 MA20/60 또는 E028로 바꾸는 구조가 더 맞다는 결론을 얻었다.

### Softmax meta-policy

월초 feature로 세 후보 중 하나를 고르는 softmax meta-policy를 구현했다.

두 가지 학습 방식을 비교했다.

- 월별 winner label을 맞추는 label softmax
- 선택 전략의 월수익률을 직접 최대화하는 return softmax

valid에서는 좋아 보였다.

- selected softmax valid 월평균 수익률: `1.05%`
- Buy & Hold valid 월평균 수익률: `0.28%`

하지만 test에서는 실패했다.

- selected softmax test 월평균 수익률: `8.91%`
- Buy & Hold test 월평균 수익률: `12.06%`
- E028 test 월평균 수익률: `9.31%`

2024년 valid regime에서 고른 모델이 2025~2026 상승장 regime으로 잘 일반화되지 않았다.

### Walk-forward meta-policy

softmax meta-policy의 과적합 가능성을 줄이기 위해 train 구간까지 월별 candidate return dataset을 만들고 walk-forward 방식으로 다시 평가했다.

절차는 다음과 같았다.

1. 2020~2023 train으로 softmax 후보 학습
2. 2024 valid에서 best 후보 선택
3. 2025~2026 test에 고정 적용
4. 선택된 hyperparameter를 train+valid로 refit해 test 확인

결과적으로 walk-forward meta-policy는 Buy & Hold에 수렴했다.

- test 월평균 수익률: `12.06%`
- Buy & Hold test 월평균 수익률: `12.06%`
- oracle best test 월평균 수익률: `13.40%`

즉 현재 feature와 softmax 구조로는 E028을 선택해야 하는 구간을 충분히 학습하지 못했다.

### 15% 손절 후 재진입 아이디어 평가

사용자가 제안한 “일정 수준, 예를 들어 15% 떨어지면 손절하고 반등 가능성이 보이면 재진입” 아이디어를 룰 기반으로 구현했다.

테스트한 조건:

- `stop15_reenter_ma_rs`
- `stop20_reenter_ma_rs`
- `stop15_reenter_ma_market_rs`
- `stop20_reenter_ma_market_rs`

결과는 단독 전략으로는 좋지 않았다.

- `stop15_reenter_ma_rs`: test 평균 수익률 `255.1%`, MDD `-37.8%`
- `stop20_reenter_ma_rs`: test 평균 수익률 `292.1%`, MDD `-36.5%`
- E028 replay: test 평균 수익률 `326.6%`, MDD `-28.9%`

핵심 해석은 이렇다.

- 15% 손절이 포트폴리오 MDD를 15%로 제한하지 않는다.
- 갭 하락, 반복 손절, 늦은 재진입 때문에 누적 equity MDD는 더 커질 수 있다.
- 손절은 단독 전략보다 PPO target ratio를 줄이는 risk overlay로 쓰는 편이 더 적절하다.

### 필요한 feature만 최소 확장

Meta-policy와 walk-forward 결과를 보면, 단순 classifier 튜닝보다 장기 시장 trend와 장기 종목 momentum 정보가 부족해 보였다.

다만 feature를 너무 많이 늘리면 모델이 산으로 갈 수 있으니 8개만 추가했다.

- `return_60d`
- `return_120d`
- `ma60_gap`
- `ma120_gap`
- `market_return_60d`
- `market_return_120d`
- `market_ma60_gap`
- `market_ma120_gap`

이 feature set으로 E030을 학습했다.

E030 결과:

- test 평균 수익률: `273.3%`
- test 평균 Sharpe: `3.00`
- test 평균 MDD: `-26.4%`
- 평균 target ratio: `0.680`

E030은 E028보다 MDD를 낮췄지만, 수익률과 Sharpe를 포기했다. 장기 trend feature는 위험을 줄이는 신호로 작동했지만 정책이 너무 방어적으로 반응했다.

### E031: 공격 보상 강화 재시도

E030이 너무 방어적이었기 때문에 같은 feature set에서 reward를 더 공격적으로 조정한 E031을 학습했다.

변경:

- `drawdown_penalty`: `0.10` -> `0.02`
- `ma_underperformance_penalty`: `0.50` -> `0.25`
- `ma_outperformance_bonus`: `0.50` -> `1.00`

하지만 결과는 더 방어적이었다.

- test 평균 수익률: `138.6%`
- test 평균 Sharpe: `2.57`
- test 평균 MDD: `-23.2%`
- 평균 target ratio: `0.527`

보상을 더 공격적으로 바꿨는데도 실제 정책은 평균 노출을 더 낮췄다. 따라서 단순 reward weight 조정만으로는 원하는 노출을 안정적으로 만들기 어렵다는 결론을 얻었다.

### 현재 결론

2026-05-13 기준 best PPO 후보는 E028이다.

비교:

| 후보 | test 평균 수익률 | Sharpe | MDD | 평균 target ratio |
| --- | ---: | ---: | ---: | ---: |
| E028 | 326.6% | 3.25 | -28.9% | 0.788 |
| E030 | 273.3% | 3.00 | -26.4% | 0.680 |
| E031 | 138.6% | 2.57 | -23.2% | 0.527 |

장기 trend feature는 유지할 가치가 있지만, 지금 방식에서는 정책이 방어적으로 무너지는 경향이 있다. 다음에는 feature를 더 늘리기보다 평균 노출 하한, action mask, 또는 구조적으로 방어 과잉을 막는 방식이 더 적절하다.

관련 보고서:

- `reports/krx_e027_liquid48_training_result_2026-05-13.md`
- `reports/krx_e028_liquid48_aggressive_result_2026-05-13.md`
- `reports/krx_gated_policy_and_e029_result_2026-05-13.md`
- `reports/krx_meta_policy_input_analysis_2026-05-13.md`
- `reports/krx_softmax_meta_policy_result_2026-05-13.md`
- `reports/krx_walk_forward_meta_policy_result_2026-05-13.md`
- `reports/krx_stop_reentry_strategy_result_2026-05-13.md`
- `reports/krx_e030_long_trend_feature_result_2026-05-13.md`
- `reports/krx_e031_long_trend_more_aggressive_result_2026-05-13.md`

### Regime cap grid 추가 결과

E032 replay + regime exposure cap 후보를 `0.70/0.75/0.80/0.85`로 촘촘히 비교했다.

- valid 기준 최선: `strong_trend_full_else070`
- test `strong_trend_full_else070`: 수익률 `380.0%`, Sharpe `3.39`, MDD `-29.2%`
- test `strong_trend_full_else080`: 수익률 `420.6%`, Sharpe `3.56`, MDD `-30.9%`

현재 기본 후보는 E028보다 수익률이 높고 MDD가 거의 같은 `strong_trend_full_else070`이다. MDD `~31%`를 허용하는 공격 후보는 `strong_trend_full_else080`이다.

관련 보고서:

- `reports/krx_regime_exposure_cap_grid_result_2026-05-13.md`
- `reports/krx_regime_exposure_cap_breakdown_2026-05-13.md`

### Stock-level cap 예외 평가

강한 종목은 market strong trend가 아니어도 full exposure를 허용하는 조건을 추가했다.

- `strong_trend_or_very_strong_stock_full_else070` test: 수익률 `383.4%`, Sharpe `3.40`, MDD `-29.4%`
- 기존 `strong_trend_full_else070` test: 수익률 `380.0%`, Sharpe `3.39`, MDD `-29.2%`

test 개선은 작고 valid/MDD가 악화돼 기본 후보는 `strong_trend_full_else070`으로 유지한다.

### Candidate scorecard 정리

기존 PPO 후보, Buy & Hold, MA20/60, E032 replay cap 후보를 같은 표로 비교했다.

- valid 1위: `E032_replay_strong_trend_full_else070`
- test `strong_trend_full_else070`: 수익률 `380.0%`, Sharpe `3.39`, MDD `-29.2%`
- test E028 PPO: 수익률 `326.6%`, Sharpe `3.25`, MDD `-28.9%`

기본 후보는 `strong_trend_full_else070`으로 유지한다. 다음 단계는 이 rule을 최종 overlay target ratio 산출기로 고정하는 것이다.

관련 보고서:

- `reports/krx_candidate_scorecard_2026-05-13.md`

### Current target ratio 산출

`src/stock_rl/generate_current_targets.py`를 추가해 최신 feature 행 기준 종목별 목표비중을 생성했다.

- 산출물: `reports/current_targets_20260511_strong_trend_full_else070.csv`
- raw 가격 최신일: `2026-05-11`
- daily feature 최신일: `2026-05-11`
- 48종목 전체 산출
- 47종목은 2026-05-11 feature 사용, 1종목은 2026-05-08 feature 사용
- 평균 target ratio: `0.835`
- 분포: `1.00` 6종목, `0.88` 26종목, `0.70` 16종목

### 운용용 후보표 생성

`current_targets` CSV는 모델 진단 컬럼이 많아서 사람이 바로 보기 어렵다. 종목명, 시장, 현재가, 목표비중, cap 이유, 20/60일 momentum, drawdown만 추린 운용용 후보표를 추가했다.

- 구현: `src/stock_rl/build_trading_sheet.py`
- CSV: `reports/trading_sheet_20260511_strong_trend_full_else070.csv`
- Markdown: `reports/trading_sheet_20260511_strong_trend_full_else070.md`
- 2026-05-11 기준 평균 target ratio: `83.5%`
- full target: 6종목
- capped target: 16종목
