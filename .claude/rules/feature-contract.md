# Feature Contract

## 목적

온라인과 오프라인에서 동일한 피처 의미와 ordering을 보장한다.

## 필수 규칙

1. `configs/feature_schema.yaml`의 feature 순서와 이름과 단위를 변경하지 않는다.
2. 온라인과 오프라인은 동일한 ordering을 사용한다.
3. 각 feature는 생성 시점보다 미래의 데이터를 참조하지 않는다.
4. 결측 처리와 fill 정책은 schema에 정의하고 코드에서 일관되게 적용한다.
5. 파생 지표는 lookback 길이와 입력 price 기준을 명시한다.

## 금지

- 모델 코드 안에서 feature ordering을 암묵적으로 하드코딩하는 행위
- adjusted 가격과 raw 가격을 섞어서 feature를 만드는 행위
- fit이 필요한 변환을 train 외 구간에 다시 맞추는 행위

## 변경 분류

- feature 추가, 삭제, 순서 변경, 단위 변경은 Contract change다.
