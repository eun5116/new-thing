# Data Contract

## 목적

시장 데이터 수집과 저장과 정렬에서 의미가 바뀌지 않도록 공통 계약을 고정한다.

## 필수 규칙

1. 데이터 provider 이름과 버전 또는 SDK 버전을 기록한다.
2. symbol universe를 명시한다.
3. interval을 명시한다. 예: `1m`, `5m`, `1d`
4. timezone을 명시한다. 저장은 UTC를 권장하고 해석은 거래소 현지 시간 기준을 명시한다.
5. trading calendar를 명시한다. 휴장일, 조기 종료, DST 변화를 거래소 캘린더 기준으로 처리한다.
6. adjusted/raw 기준을 명시한다. split, dividend 반영 여부를 코드와 문서에서 일치시킨다.
7. 결측과 휴장과 수집 실패 처리 규칙을 문서화한다.

## 금지

- 공개 시각 이전의 뉴스, 실적, 경제지표를 과거 시점에 병합하는 행위
- 당일 확정 전 종가를 종가로 사용하는 행위
- 미래 바를 사용한 보간 또는 fill

## point-in-time join

- feature timestamp는 항상 prediction timestamp 이하여야 한다.
- 발표 시각이 있는 데이터는 실제 공개 시각 이후 바부터만 사용한다.

## 최소 메타데이터

`dataset_meta.yaml` 또는 동등한 로그에 아래를 남긴다.

- provider
- symbols
- interval
- timezone
- calendar
- adjusted/raw
- collected_at
- notes
