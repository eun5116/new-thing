# Feature Spec

## 목적

모델 입력 피처의 의미를 고정한다.

## 기본 범주

- price returns
- rolling volatility
- range based features
- volume features
- calendar features
- optional market regime features

## 필수 규칙

1. feature는 `configs/feature_schema.yaml` 순서를 따른다.
2. 각 feature는 생성 시점까지의 데이터만 사용한다.
3. lookback 길이와 입력 price 기준을 명시한다.
4. adjusted/raw 기준을 혼용하지 않는다.

## 예시

- log return 1
- log return 5
- rolling std 20
- Parkinson volatility 20
- volume zscore 20
- day of week
