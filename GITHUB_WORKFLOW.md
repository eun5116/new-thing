# GitHub Workflow

이 문서는 이 저장소에서 변경사항을 GitHub에 올릴 때 자주 쓰는 명령을 정리한다.

## 현재 상태 확인

```bash
cd /home/jack/project_splits_20260512/weekly_market_report_repo
git status -sb
git diff
```

- `git status -sb`: 변경된 파일 목록을 짧게 본다.
- `git diff`: 아직 commit하지 않은 상세 변경 내용을 본다.

## 한번에 올리기

수정한 내용을 새 commit으로 만들고 현재 branch에 push한다.

```bash
cd /home/jack/project_splits_20260512/weekly_market_report_repo
git status -sb
git add .
git commit -m "Update weekly market report"
git push
```

새 branch를 처음 push하는 경우에는 upstream을 같이 지정한다.

```bash
git push -u origin HEAD
```

## 일부 파일만 올리기

원하는 파일만 골라 commit한다.

```bash
git status -sb
git add weekly_market_report/weekly_report.py configs/dataset/kospi_alerts.yaml
git commit -m "Update KOSPI momentum alerts"
git push
```

## 직전 commit에 합쳐서 진행하기

아직 GitHub에 push하지 않은 직전 commit에 현재 변경사항을 합친다.

```bash
git status -sb
git add .
git commit --amend
git push
```

commit 메시지를 바꾸지 않고 바로 합치려면:

```bash
git add .
git commit --amend --no-edit
git push
```

이미 GitHub에 push한 직전 commit을 고쳤다면 push가 거절될 수 있다. 이때는 원격 branch를 안전하게 갱신한다.

```bash
git push --force-with-lease
```

`--force-with-lease`는 내가 마지막으로 본 원격 상태가 그대로일 때만 강제 push한다. 협업 중이면 사용 전에 원격 변경이 있는지 확인한다.

### amend 후 non-fast-forward rejected가 뜰 때

`git commit --amend`를 하면 commit 내용이 같아 보여도 commit 해시가 바뀐다. 이미 GitHub에 올라간 commit을 amend했다면 일반 `git push`는 아래처럼 거절될 수 있다.

```text
! [rejected] weekly-market-report -> weekly-market-report (non-fast-forward)
```

내가 방금 amend한 commit으로 GitHub의 기존 commit을 교체하려는 상황이면:

```bash
git status -sb
git log --oneline HEAD..origin/weekly-market-report
git log --oneline origin/weekly-market-report..HEAD
git push --force-with-lease origin weekly-market-report
```

현재 작업 중인 파일까지 직전 commit에 같이 넣고 올리려면:

```bash
git add .
git commit --amend --no-edit
git push --force-with-lease origin weekly-market-report
```

원격에 내가 모르는 새 commit이 있으면 `--force-with-lease`가 실패한다. 그때는 먼저 `git fetch origin` 후 원격 commit 내용을 확인한다.

## GitHub HTTPS 인증

GitHub는 Git push에서 계정 비밀번호 인증을 지원하지 않는다. HTTPS remote를 쓸 때 password 입력칸에는 GitHub 계정 비밀번호가 아니라 Personal Access Token을 넣어야 한다.

```bash
git remote -v
git push
```

처음 인증할 때:

- Username: GitHub username 또는 이메일
- Password: GitHub Personal Access Token

토큰을 계속 입력하기 싫으면 credential helper를 켠다.

```bash
git config --global credential.helper store
```

보안이 더 중요한 환경이면 `store` 대신 OS credential manager를 사용한다.

## 최근 여러 commit을 하나로 합치기

최근 3개 commit을 정리하려면:

```bash
git log --oneline -5
git rebase -i HEAD~3
git push --force-with-lease
```

interactive 화면에서 남길 commit은 `pick`, 합칠 commit은 `squash` 또는 `fixup`으로 바꾼다.

## 원격 변경 먼저 가져오기

push 전에 GitHub 쪽 변경을 먼저 반영한다.

```bash
git fetch origin
git status -sb
git pull --rebase
```

충돌이 나면 파일을 고친 뒤:

```bash
git add .
git rebase --continue
git push
```

## 되돌리기 전에 확인할 명령

실수로 staging한 파일을 내리려면:

```bash
git restore --staged path/to/file
```

아직 commit하지 않은 변경을 버리는 명령은 복구가 어렵다. 실행 전 `git diff`로 반드시 확인한다.

```bash
git restore path/to/file
```

## 자주 쓰는 확인 명령

```bash
git branch --show-current
git remote -v
git log --oneline -10
```
