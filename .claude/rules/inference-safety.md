# Inference Safety

## 목적

온라인 조회와 추론과 신호 사용에서 잘못된 입력과 과신을 억제한다.

## 최소 safety gate

1. stale data guard
2. missing feature fallback
3. confidence threshold
4. risk gate

## stale data guard

- 데이터 freshness 임계값을 config에 둔다.
- 임계값 초과 시 예측을 무효 처리하거나 fallback을 사용한다.

## missing feature fallback

- 필수 feature가 비어 있으면 추론을 강행하지 않는다.
- fallback 응답은 명시적으로 표시한다.

## confidence threshold

- 모델 confidence 또는 uncertainty proxy가 낮으면 보수적으로 처리한다.

## risk gate

- 출력 score를 그대로 거래 의사결정에 연결하지 않는다.
- 포지션 크기, 알림 강도, 추천 노출 수준에 상한을 둔다.

## shadow mode

- 신규 모델 또는 gate 변경 시 먼저 shadow mode로 기록만 수행한다.
- shadow mode 결과와 오프라인 재현 결과를 비교한다.
