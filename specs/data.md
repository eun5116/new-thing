# Data Spec

## 목적

시장 데이터의 저장 형식과 시간 의미를 고정한다.

## 원천 데이터

- OHLCV
- corporate actions
- 뉴스 또는 이벤트 데이터
- 선택적 거시 지표

## 필수 정의

- provider
- symbol universe
- interval
- timezone
- trading calendar
- adjusted/raw 기준

## 정렬 규칙

- 저장은 UTC를 권장한다.
- 해석은 거래소 현지 시간 기준을 명시한다.
- 이벤트성 데이터는 공개 시각 이후 바에만 결합한다.

## 금지

- 미래 시점 이벤트 병합
- 임의의 전일 종가 복원
- split/dividend 반영 기준 혼용
