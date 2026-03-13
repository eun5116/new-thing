# Alert Spec

## 목적

단기 강한 상승 추세를 보이는 종목을 별도 체크 대상으로 표시한다.

## 기본 알림

### two_week_momentum_watch

- 의미: 최근 약 2주 동안 지속적으로 상승했고, 누적 상승률이 10% 이상인 종목을 체크 대상으로 표시
- 기본 기간: 최근 10 거래일
- 기본 임계값: 누적 수익률 10% 이상

## 판정 규칙

1. 입력 가격 기준은 `adjusted_close`를 기본으로 한다.
2. 판정 시점 `t`에서 최근 10 거래일 구간 `t-9 ... t`만 사용한다.
3. 누적 상승률은 아래 식으로 계산한다.
   - `total_return = close_t / close_{t-9} - 1`
4. 지속적 상승은 아래 둘 중 하나로 정의하고, 프로젝트에서 하나를 고정해 사용한다.
   - strict: 10거래일 동안 일별 수익률이 모두 0 초과
   - tolerant: 10거래일 중 상승 마감이 8일 이상이며, 3거래일 연속 하락이 없음
5. 기본 정책은 `tolerant`로 한다.

## 출력 필드

- `alert_name`
- `symbol`
- `as_of_date`
- `window_trading_days`
- `total_return`
- `up_days`
- `max_consecutive_down_days`
- `triggered`

## 금지

- 휴장일 포함 달력 일수로 2주를 계산하는 행위
- adjusted/raw 기준을 혼용하는 행위
- 당일 미확정 종가를 사용한 판정
