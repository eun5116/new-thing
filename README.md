# Exodus Chat Escape

선택형, 정렬형, 단답형 문제를 섞은 출애굽 방탈출 게임입니다.

## Local Run

```bash
npm run dev
```

브라우저에서 엽니다.

```text
http://127.0.0.1:4174/
```

이 로컬 서버는 정적 파일을 제공합니다. 현재 게임은 비용 없는 정적 NPC 대화와 장면별 퍼즐을 사용합니다.

## Interaction Mode

API 없이 동작하도록 모든 진행을 브라우저 안의 정적 데이터로 처리합니다.

- 1장 불붙은 떨기나무: NPC 대화 뒤에 제시되는 선택지를 골라 진행합니다.
- 2장 바로의 궁전: 10가지 재앙을 드래그해서 올바른 순서로 정렬합니다.
- 3장 이후: 선택지 없이 정답을 직접 입력하는 단답형 문제로 진행합니다.
- 단서 요청: 현재 장면의 힌트를 NPC 말풍선으로 표시합니다.

## Animated Avatars

큰 NPC 영역은 4장의 개별 프레임 이미지를 자동으로 순환 재생합니다. 프레임 로딩에 실패하면 각 인물의 첫 번째 프레임을 정지 이미지로 사용합니다.

필요한 파일명:

```text
assets/moses1.png
assets/moses2.png
assets/moses3.png
assets/moses4.png
assets/aaron1.png
assets/aaron2.png
assets/aaron3.png
assets/aaron4.png
assets/hebrew-family1.png
assets/hebrew-family2.png
assets/hebrew-family3.png
assets/hebrew-family4.png
assets/egypt-guard1.png
assets/egypt-guard2.png
assets/egypt-guard3.png
assets/egypt-guard4.png
```

각 프레임은 같은 크기여야 합니다. 투명 배경 PNG를 권장합니다. 평소에는 느리게 재생되고, NPC가 말할 때는 더 빠르게 재생됩니다.

## Netlify Deploy

정적 게임이므로 OpenAI 설정은 필요 없습니다.
