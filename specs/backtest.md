# Backtest Spec

## 목적

예측 성능과 전략 성능을 구분해서 평가한다.

## 평가 구분

1. 예측 지표
2. 의사결정 지표
3. 전략 지표

## 예측 지표

- RMSE
- MAE
- Pearson correlation
- Spearman correlation
- bucketed calibration

## 전략 지표

- cumulative return
- annualized return
- volatility
- Sharpe
- max drawdown
- turnover

## 필수 가정

- 거래 비용
- 슬리피지
- 체결 시점
- 리밸런싱 주기

## 금지

- 비용 없는 결과만 보고하는 것
- look-ahead bias가 있는 진입 시점 사용
