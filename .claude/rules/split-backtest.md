# Split And Backtest

## split 규칙

1. split은 시간 기준으로만 수행한다.
2. 기본은 `train/val/test` 날짜 구간 분리다.
3. 대안으로 walk-forward validation 또는 expanding window를 허용한다.
4. 랜덤 샘플 분할을 금지한다.
5. scaler, selector, threshold, model fit은 train 구간만 사용한다.

## 백테스트 규칙

1. 백테스트는 label horizon과 decision horizon을 명확히 구분한다.
2. 거래 비용과 슬리피지와 체결 가정을 명시한다.
3. survivorship bias 여부를 명시한다. 가능하면 point-in-time universe를 사용한다.
4. look-ahead bias 여부를 점검한다.
5. 결과에는 기준선 baseline을 반드시 포함한다.

## 최소 보고 지표

- RMSE 또는 MAE
- correlation 또는 rank correlation
- calibration 또는 bucketed error
- backtest return
- max drawdown
- turnover

## 필수 확인 질문

- 예측 시점에 실제로 알 수 있는 데이터만 사용했는가?
- 비용 반영 전후 결과를 모두 기록했는가?
- random split이 숨어 있지 않은가?
