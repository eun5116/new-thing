# Exodus Chat Game 분리 안내

이 폴더는 기존 `trail and error` 저장소의 `online_room_escape` 브랜치에서 `exodus_chat_game/`만 추출한 clean copy다.

## 제외한 항목

- `.env`
- `.venv`
- `node_modules/`
- 캐시와 빌드 산출물
- weekly market report 관련 파일

## 새 저장소로 시작하기

```bash
cd /home/jack/project_splits_20260512/exodus_chat_game
git init
git add .
git status
git commit -m "Initial Exodus chat game import"
```

