# Run Directory

## 목적

실험 하나를 나중에 다시 재현할 수 있도록 최소 산출물 구조를 강제한다.

## 경로

- `runs/<run_id>/`

## 필수 파일

- `run_config.yaml`
- `metrics.json`
- `stdout.log`
- `scaler.json` 또는 동등 산출물
- `model.onnx` 또는 `model.pt`
- `predictions.parquet` 또는 `predictions.csv`

## run_config 최소 항목

- run_id
- git_commit
- dirty
- command
- seed
- dataset_config
- feature_schema
- model_config
- symbols 또는 universe
- interval
- timezone
- split 경계
- label_name
- horizon

## 권장 항목

- data fingerprint
- provider version
- feature list hash
- export 옵션
