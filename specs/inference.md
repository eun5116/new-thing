# Inference Spec

## 목적

온라인 서비스에서 수행하는 작업 범위를 고정한다.

## 온라인에서 허용되는 일

- 최신 시세 또는 캐시 데이터 조회
- feature 생성
- 저장된 모델로 추론
- 상승 추세 알림 조건 판정
- safety gate 적용
- 결과 반환 또는 기록

## 온라인에서 금지되는 일

- model fitting
- optimizer step
- checkpoint 저장
- train 통계 재계산

## 입력

- symbol
- timestamp 또는 latest 요청
- 필요한 경우 interval

## 출력 예시

- predicted_volatility
- confidence
- momentum_watch_triggered
- data_freshness_ms 또는 bar_age
- fallback_used
