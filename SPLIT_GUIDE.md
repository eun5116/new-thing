# Weekly Market Report 분리 안내

이 폴더는 기존 `trail and error` 저장소에서 주간 시장 리포트 실행에 필요한 코드와 설정만 추린 clean copy다.

## 포함한 항목

- `weekly_market_report/weekly_report.py`
- `weekly_market_report/run_weekly.sh`
- `weekly_market_report/requirements.txt`
- `weekly_market_report/.env.example`
- `weekly_market_report/README.md`
- `configs/dataset/kospi_alerts.yaml`
- `configs/dataset/us_equities_alerts.yaml`
- 로컬 agent 지침 파일

## 제외한 항목

- `.env`
- `.venv`
- `__pycache__/`
- `weekly_market_report/cache/`
- `weekly_market_report/data/`
- `weekly_market_report/history/`
- `weekly_market_report/outputs/`
- `weekly_market_report/weekly.log`

## 새 저장소로 시작하기

```bash
cd /home/jack/project_splits_20260512/weekly_market_report_repo
git init
git add .
git status
git commit -m "Initial weekly market report import"
```

## GitHub에 올리기

commit, push, 직전 commit에 합치기, force-with-lease 사용법은 `GITHUB_WORKFLOW.md`를 참고한다.
