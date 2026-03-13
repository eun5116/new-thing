# /new-run

새 실험 run 디렉터리를 만든다.

## 목적

- 재현 가능한 실험 스캐폴딩 생성

## 생성 항목

- `runs/<run_id>/run_config.yaml`
- `runs/<run_id>/stdout.log`
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/figures/`

## 확인 사항

- dataset config
- model config
- feature schema
- seed
- split 경계
- label과 horizon
