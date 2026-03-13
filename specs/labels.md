# Label Spec

## 목적

변동성 예측 라벨의 정의를 고정한다.

## 기본 라벨

### 1. next-day realized volatility

- 기준: 다음 거래일 수익률 기반 realized volatility
- 입력 시점: `t`
- 라벨 시점: `t+1`

### 2. next-k-bar realized volatility

- 기준: `t+1`부터 `t+k`까지의 수익률로 계산
- `k`는 dataset config에 명시한다.

### 3. volatility regime classification

- 기준: 미래 realized volatility가 train 구간 threshold 이상인지 여부

## 계산 규칙

- 수익률 정의: 기본은 log return
- annualization 여부를 명시한다.
- threshold가 필요하면 train split 기준으로만 추정한다.

## 금지

- test 구간 통계로 threshold 설정
- horizon과 label alignment 불일치
