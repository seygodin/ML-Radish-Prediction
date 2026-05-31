---
name: radish-data-pipeline
description: 무(radish) 질병 데이터셋용 PyTorch 데이터 파이프라인을 만들 때 반드시 사용. 분류(정상 vs disease_3/4)·detection(단일 거친 bbox) Dataset/DataLoader, split/manifest, 전처리·증강, 불균형 샘플링, EXIF·해상도·zero-dim 라벨 처리를 다룬다. "데이터로더 만들어", "전처리 파이프라인", "dataset 코드", "manifest 생성", "증강 추가", "불균형 샘플러" 요청 시 트리거. 모델 학습/평가 코드 자체는 experiment-runner 스킬 소관.
---

# radish-data-pipeline

무 질병 데이터셋을 모델이 바로 학습 가능한 **재현 가능한 PyTorch 데이터 파이프라인**으로 만든다.
데이터 사실은 `CLAUDE.md`와 `report/REPORT.md`가 단일 출처다 — 먼저 읽어 수치를 확정하라.

## 왜 이 파이프라인이 까다로운가 (검증된 함정)
이 데이터는 그대로 학습에 넣으면 조용히 틀어진다. 아래는 분석으로 **확인된 사실**이며 코드로 방어해야 한다.

- **zero-dim 라벨 43건**: 질병 JSON 43개의 `description.width/height`가 0이다. JSON 차원을 믿으면 0 나눗셈·bbox 정규화 오류가 난다 → 크기는 **실제 이미지 파일에서 읽어라**.
- **클래스 불균형 13–16:1**: 정상이 압도적이다 → 정확도는 무의미. baseline 프로토콜은 **normal 다운샘플링으로 type 간 표본 수를 균등화**한다(아래 "클래스 균형" 참조). 데이터 레이어는 다운샘플 + (보조)가중 샘플러/클래스 가중치를 모두 지원한다.
- **해상도·방향 혼재**(720×960 ~ 6000×4000, 세로/가로): EXIF 회전 미보정 시 라벨 좌표와 어긋난다 → 로딩 시 `ImageOps.exif_transpose` 후 resize.
- **확장자 혼재**(.jpg/.jpeg/.JPG): 이미지 탐색은 대소문자 무시.
- **bbox는 이미지당 1개, 거친 영역 박스**(면적 중앙값 ≈50%, 중앙 집중). 정상에도 박스가 있으므로 **박스 존재로 정상/질병을 나누지 말 것**.
- **valid는 데이터셋 제공 split을 그대로** 쓴다(train에서 재분할 금지). 클래스 stratify가 필요하면 train 내부에서만.

## 데이터 위치
- 라벨/이미지: `data/{train,valid}/[라벨·원천]무_{0.정상,1.질병}` (라벨 `<img>.json`, 1:1 매칭 검증됨).
- 질병 종류별 분리(심링크): `data/by_disease/<split>/disease_{3,4}/{images,labels,image_w_bbox}/`.
- 라벨 스키마와 코드 의미는 `CLAUDE.md` 참조(disease 0=정상/3·4=질병, risk 심각도, points=bbox).

## 강건한 이미지+라벨 로더 (모든 Dataset의 기반)
EXIF·zero-dim·확장자 문제를 한 곳에서 해결한다. 새 코드는 이 패턴을 재사용하라.

```python
import os, json
from PIL import Image, ImageOps

def load_image(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return im  # 회전 보정된 RGB, 실제 크기는 im.size

def parse_label(json_path, image_path):
    a = json.load(open(json_path))["annotations"]
    w, h = Image.open(image_path).size          # JSON이 아니라 실제 파일에서
    disease = a["disease"]                        # 0 정상 / 3,4 질병
    boxes = [[p["xtl"], p["ytl"], p["xbr"], p["ybr"]] for p in (a.get("points") or [])]
    return dict(disease=disease, risk=a["risk"], boxes=boxes, w=w, h=h)

def find_image(stem, image_dir, _idx={}):       # 대소문자 무시 + 캐시
    key = image_dir
    if key not in _idx:
        _idx[key] = {f.lower(): f for f in os.listdir(image_dir)}
    return _idx[key].get(stem.lower())          # stem 예: "V006_..._1.jpg"
```

## 산출물 (계약)
1. **`src/data/` 모듈** — import 가능한 Dataset/DataLoader. 최소 공개 API:
   - `build_classification_loaders(disease_code, img_size, batch_size, ...) -> (train_loader, valid_loader, class_weights)`
   - `build_detection_loaders(img_size, batch_size, ...) -> (train_loader, valid_loader)` (torchvision detection 형식: `image, {"boxes","labels"}`)
   - 파일 하단에 `if __name__ == "__main__":` 스모크 테스트(한 배치 로딩 → shape/라벨/박스 출력).
2. **`_workspace/data/manifest_{task}.csv`** — 행: `split, image_path, label_path, klass, disease_code, risk, x0,y0,x1,y1` (좌표는 실제 크기 기준).
3. **`_workspace/data/data_card_{task}.md`** — 클래스 분포, 입력 변환, 샘플러/가중치, 스킵된 항목 수와 사유, **공개 API 시그니처**.

## 작업 절차
1. `CLAUDE.md`·`report/REPORT.md`로 사실 확정 → 필요하면 `data/verify_pairs.py`로 매칭 재검증.
2. ml-researcher 사양(`_workspace/specs/`)이 있으면 입력 크기·정규화·가중치 요구를 반영. 없으면 합리적 기본값(ImageNet 정규화, img_size=320 분류, resize+pad 유지 detection).
3. manifest를 고정 seed로 생성·동결(실제 크기 기준 좌표).
4. Dataset/DataLoader 구현 + 가중 샘플러/클래스 가중치 + 증강(분류: flip/color jitter/random resized crop; detection: 박스 동기 변환).
5. **스모크 테스트 실행**으로 shape·라벨·박스 정합 확인 → data_card 작성.

## 분류 vs detection 라벨 구성
- **분류 (확정 3세팅)**: (1) normal vs disease_3, (2) normal vs disease_4, (3) normal vs d3 vs d4(3-class). 정상은 `[라벨]무_0.정상`, 질병은 `data/by_disease/<split>/disease_{3,4}` 사용.
- **detection**: 질병 이미지의 단일 거친 bbox. `labels`는 단일 클래스(질병) 또는 disease_code별. 박스는 1개. 좌표는 resize 스케일에 맞춰 변환.

## 클래스 균형 — normal 다운샘플링 (확정)
type 간 표본 수를 맞춰 불균형을 제거한다. **train에만 적용**하고 valid는 제공 split을 그대로 둔다(현실 분포 평가). 다운샘플은 **고정 seed**로 manifest에 동결해 재현한다.
- 2-class: `n_normal = n_disease_k` (해당 disease_code 수만큼 normal을 무작위 추출).
- 3-class: 세 type을 동일 수로 — `n = min(n_normal, n_d3, n_d4)` 기준으로 각 type을 다운샘플(보통 disease_4가 최소).
- 다운샘플 후 실제 표본 수를 data_card에 명시한다(소표본이므로 평가 해석에 중요).

## 성능 — 디코드 1회 + RAM 캐시 (필수)
원본 JPEG는 최대 6000×4000(수 MB)이다. `__getitem__`에서 매 접근마다 디코드하면 **GPU가 CPU 디코드를 기다리며 굶는다**(실측: GPU util 0%, nextvit 104s/epoch). 데이터셋이 작으므로(train 수백 / valid ~1400) **로더 생성 시 디코드+다운스케일을 1회만 하고 RAM에 캐시**하라:
- 분류: 짧은변 = `round(img_size*256/224)`로 리사이즈한 PIL을 리스트로 보관 → `__getitem__`은 작은 이미지에 augmentation만.
- detection: img_size 정사각형으로 리사이즈 + 박스 사전 스케일을 캐시(`DetectionTransform(pre_resized=True)`로 resize 생략).
- DataLoader 워커는 캐시 빌드 **후** fork되므로 copy-on-write로 공유(재디코드 0, 워커별 중복 없음). ThreadPoolExecutor로 캐시 빌드 병렬화.
- 효과(실측): epoch 104s→~1.8s(약 50배), GPU util 0%→55–87%. 비용은 run당 1회 캐시 빌드(~30–60s).

## 검증 체크리스트
- [ ] zero-dim 라벨에서 크기를 실제 파일로 읽는가
- [ ] EXIF 회전 보정 후 박스 좌표가 이미지와 일치하는가(시각 확인 1~2장 권장)
- [ ] valid를 재분할하지 않았는가
- [ ] 클래스 가중치/샘플러가 manifest 분포와 일치하는가
- [ ] 스모크 테스트가 통과하고 data_card에 공개 API가 적혀 있는가
