# Safety Spec

## 목적

추론 결과를 과신하거나 오래된 데이터로 잘못 사용하는 것을 방지한다.

## 최소 규칙

1. stale data guard
2. missing feature fallback
3. confidence threshold
4. risk limit

## 운영 절차

1. 신규 모델은 shadow mode부터 시작한다.
2. shadow mode 결과를 오프라인 재현과 비교한다.
3. 품질이 확인되면 제한된 범위에서 적용한다.

## 로그

- request timestamp
- data timestamp
- freshness
- model version
- confidence
- gate decision
