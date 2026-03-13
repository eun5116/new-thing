# Label Contract

## 목적

변동성 예측 라벨의 의미를 고정하고, 학습과 평가와 추론에서 같은 목표를 사용하도록 한다.

## 지원 범위

- next-day realized volatility
- next-k-bar realized volatility
- volatility regime classification

## 필수 규칙

1. label 정의는 `specs/labels.md`에 먼저 적고 구현한다.
2. horizon은 `configs/dataset/*.yaml`과 `specs/labels.md`에 동시에 명시한다.
3. realized volatility 계산식은 return 정의와 annualization 여부를 함께 명시한다.
4. 회귀 라벨과 분류 라벨은 혼용하지 않는다. 실험별 목표를 명확히 구분한다.
5. target transform이 있으면 train split에 대해서만 fit하거나 추정한다.

## 예시

- 회귀: `y_t = realized_vol(t+1 ... t+h)`
- 분류: `y_t = 1 if realized_vol_{t+h} >= threshold_train else 0`

## 금지

- future leakage가 있는 label alignment
- test 통계를 이용해 threshold를 정하는 행위
- 백테스트용 수익률 타깃과 변동성 타깃을 동일 실험에서 암묵적으로 섞는 행위
