# /preflight

작업 전 아래를 순서대로 확인한다.

1. `specs/*.md` 읽기
2. `configs/feature_schema.yaml` 읽기
3. `configs/dataset/*.yaml` 읽기
4. 변경 타입 분류: Doc-only, Code, Contract
5. 데이터 누수 위험 확인
6. split과 horizon과 timezone과 adjusted/raw 기준 확인
7. 필요한 테스트 목록 작성
8. BLOCKER 여부 판단

출력 형식:

```markdown
## Preflight
- Change type:
- SoT checked:
- Risks:
- Required tests:
- Blockers:
```
