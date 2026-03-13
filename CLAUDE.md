# Market Data + Volatility Forecasting Platform

시세 조회, 피처 생성, 변동성 예측, 백테스트, 단기 강한 상승 종목 알림을 포함하는 리포지토리다.
온라인 서비스는 조회와 추론만 수행하고, 학습과 재학습과 평가는 오프라인에서만 수행한다.

---

## Source of Truth(SoT) 우선순위

상충 시 상위가 우선한다. 코드 생성과 수정 전에 반드시 SoT를 먼저 읽고, 불일치 시 BLOCKER로 보고한다.

1. `specs/*.md`
2. `configs/feature_schema.yaml`
3. `configs/dataset/*.yaml`
4. `configs/model/*.yaml`
5. 본 `CLAUDE.md` + `.claude/rules/`
6. 코드 주석/README

---

## Non-negotiables

1. 오프라인 학습만 허용한다. 온라인 API, 배치, 서비스에 optimizer, gradient, online fitting을 넣지 않는다.
2. 데이터 누수를 금지한다. 미래 가격, 거래량, 이벤트, 지표를 참조하지 않는다.
3. placeholder 구현을 금지한다. 완성 불가 시 즉시 BLOCKER를 보고한다.
4. 공개 인터페이스를 무단 변경하지 않는다. feature schema, label, horizon, API schema 변경은 Contract change다.
5. 모든 run은 재현 가능해야 한다. git hash, command, seed, config, metrics를 기록한다.
6. 온라인 추론에는 stale data guard, confidence threshold, risk gate를 둔다.
7. 파일 변경 시 `daily_notes/<YYYY>/<MM>/<YYMMDD>.md`를 갱신한다.
8. 기능 추가 시 테스트를 반드시 함께 추가한다.

상세 규칙은 `.claude/rules/`를 따른다.

---

## 환경

- Ubuntu 22.04
- Python 3.10+
- PyTorch, numpy, pandas, scikit-learn
- ONNX 또는 TorchScript export
- 버전 고정: `requirements.txt` 또는 `pyproject.toml`

---

## 리포 구조

- `specs/`: 데이터, 피처, 라벨, 백테스트, 추론, 알림, 안전 규칙
- `configs/`: feature schema, dataset config, model config
- `ml/`: 오프라인 학습, 평가, export
- `app/`: API, batch job, inference service
- `runs/`: 실험 산출물
- `daily_notes/`: 작업 일지

---

## 변경 권한 정책

| 타입 | 범위 | 권한 |
|---|---|---|
| Doc-only | 일반 문서 | 자율 |
| Code | `*.py`, `*.yaml`, `*.json`, 테스트/CI | 사전 승인 필요 |
| Contract | specs, feature_schema, label, horizon, API schema | 승인 필수 |

애매하면 더 엄격한 쪽으로 분류한다.

---

## 코드 작성 절차

1. 변경 타입을 분류한다.
2. daily note를 먼저 기록한다.
3. SoT와 프리플라이트를 확인한다.
4. 테스트를 먼저 만든다.
5. 구현한다.
6. 검증 후 daily note를 갱신한다.

---

## Claude Code 사용 지침

- 작업 시작 전 `specs/*.md`, `configs/feature_schema.yaml`, `configs/dataset/*.yaml`를 읽는다.
- `/preflight`로 SoT 충돌, split, horizon, safety를 점검한다.
- `/daily-note`로 일지를 생성하거나 갱신한다.
- `/new-run`으로 run 디렉터리 스캐폴딩을 만든다.
- `/propose-change`로 제안서를 만든다.

---

## 핵심 규칙 요약

- 데이터: adjusted/raw 기준, timezone, market calendar, point-in-time join을 명시한다.
- 피처: `feature_schema.yaml`의 순서와 단위와 결측 처리 의미를 유지한다.
- 라벨: `specs/labels.md` 정의만 사용한다.
- 알림: `specs/alerts.md`의 조건과 계산 창을 따른다.
- split: 시간 기준 train/val/test 또는 walk-forward만 허용한다.
- 정규화: scaler는 train split로만 fit한다.
- 모델: export 후 parity test를 통과해야 한다.
- 온라인: 조회, 피처 생성, 추론만 수행한다.
- 백테스트: look-ahead bias, survivorship bias, 비용 가정을 명시한다.

---

## 언어 정책

- 기본: 한국어
- 영어: 사용자가 요청한 경우만
- 톤: 공식적이고 전문적인 문체
