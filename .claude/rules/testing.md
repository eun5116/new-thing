# Testing

## 필수 원칙

1. 기능 추가 시 테스트를 함께 추가한다.
2. 버그 수정 시 회귀 테스트를 우선 추가한다.
3. 테스트 이름은 의도를 드러내야 한다.
4. 스펙 계약은 테스트로 고정한다.

## 오프라인 필수 테스트

- `test_no_future_leakage`
- `test_split_by_time_only`
- `test_scaler_fit_train_only`
- `test_label_alignment`
- `test_feature_ordering`
- `test_export_parity`
- `test_two_week_momentum_watch`

## 온라인 권장 테스트

- `test_stale_data_guard`
- `test_missing_feature_fallback`
- `test_api_schema_stable`

## Done 조건

- 관련 테스트 추가
- 기존 테스트 통과
- 새 규칙이면 문서와 테스트를 함께 갱신
