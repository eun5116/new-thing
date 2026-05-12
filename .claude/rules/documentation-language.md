# Documentation Language

## 목적

README와 같은 Markdown 문서의 설명 언어를 일관되게 유지한다.

## 필수 규칙

1. `*.md` 파일의 본문 설명, 주석, 작업 메모, 변경 요약은 기본적으로 한글로 작성한다.
2. 사용자가 영어 작성을 명시적으로 요청한 경우에만 영어 본문을 허용한다.
3. 명령어, 코드 블록, 파일 경로, 환경 변수, 설정 키, API 필드명, 패키지명, 고유명사는 원문을 유지한다.
4. 외부 문서 제목이나 에러 메시지를 인용할 때는 원문을 유지하되, 필요한 설명은 한글로 덧붙인다.
5. 기존 Markdown 문서를 수정할 때 새로 추가하는 설명은 한글로 작성하고, 주변 문맥이 영어라면 필요한 범위에서 한글로 정리한다.

## 예시

권장:

````markdown
## 실행 방법

아래 명령으로 주간 리포트를 생성한다.

```bash
python weekly_report.py --dry-run
```
````

허용:

```markdown
`SMTP_HOST`, `EMAIL_TO`, `reportlab` 값은 원문 이름을 유지한다.
```

피해야 할 것:

```markdown
This section explains how to run the weekly report.
```
