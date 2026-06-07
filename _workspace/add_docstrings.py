#!/usr/bin/env python3
"""주석 보강: 대상 함수/클래스에 docstring이 없으면 첫 본문 라인 앞에 삽입.
로직은 절대 변경하지 않고 docstring 라인만 추가한다. 같은 이름이 여러 개면 모두에 동일 텍스트."""
import ast, io

# (파일 -> {함수/클래스명: docstring 한 줄}) — docstring은 따옴표/개행 없이 작성
DOC = {
 "src/data/core.py": {
   "Sample": "한 샘플의 메타데이터(경로·클래스·disease 코드·risk·실제 크기 기준 bbox). manifest 행과 1:1.",
   "_clip_box": "bbox(xyxy)를 이미지 경계 [0,w]/[0,h]로 클립하고 좌표 역전 시 정렬해 반환.",
   "_scan_normal": "split의 정상 라벨/이미지를 스캔해 Sample 목록 생성(매칭 실패·zero-size는 skip 로그).",
   "_scan_disease": "split·disease_code의 질병 라벨/이미지를 스캔(빈/퇴화 bbox는 detection 안전성 위해 skip).",
   "_catalog_from_manifest": "동결된 manifest CSV에서 Catalog를 복원(이미지 재스캔 없이 sub-second). 누락/무효 시 None.",
 },
 "src/data/loaders.py": {
   "_detection_collate": "detection 배치 collate: 이미지/타깃을 가변 길이 리스트로 묶음(박스 수가 달라 stack 불가).",
   "_det_counts": "detection 표본의 클래스별(normal/d3/d4) 개수 dict.",
   "__init__": "데이터셋 생성: 이미지를 1회 디코드·리사이즈해 RAM 캐시(fork COW 공유).",
   "__len__": "캐시된 샘플 수.",
   "__getitem__": "캐시 이미지에 transform을 적용해 (텐서, 라벨/타깃) 반환.",
 },
 "src/data/transforms.py": {
   "classification_train_transform": "분류 train 기본 증강(RandomResizedCrop+HFlip+약한 ColorJitter)+ImageNet 정규화.",
   "classification_eval_transform": "분류 eval 변환(Resize→CenterCrop)+ImageNet 정규화(결정적).",
   "DetectionTransform": "박스 동기 detection 변환. pre_resized=True면 캐시에서 이미 정사각 리사이즈·박스 스케일됨.",
   "__init__": "변환 파라미터 설정(img_size, train 여부, pre_resized).",
   "__call__": "이미지·박스에 (필요시 resize+스케일)·flip·정규화를 동기 적용해 반환.",
 },
 "src/data/build_manifest.py": {
   "_rel": "repo 루트 기준 상대경로 문자열로 변환(manifest 이식성).",
   "_row_base": "Sample 하나를 manifest 공통 컬럼 dict로 직렬화(실제 크기 기준 좌표).",
   "write_classification_manifest": "전 split·클래스 샘플을 분류 manifest CSV로 기록(전체 풀스캔).",
   "write_detection_manifest": "질병 샘플을 detection manifest CSV로 기록.",
   "write_data_card": "세팅별 분포·변환·스킵·공개 API를 요약한 data_card.md 작성.",
   "main": "전체 스캔으로 manifest/ data_card를 재생성하는 진입점.",
 },
 "src/models/detector.py": {
   "__init__": "동결 백본 + 단일박스 회귀(+objectness) 헤드 구성(헤드만 학습 대상).",
   "forward": "백본 풀링특징 → 박스[B,4] xyxy∈[0,1] (+ with_objectness면 objectness logit) 반환.",
 },
 "src/models/dinov3.py": {
   "build_dinov3_classifier": "arch/variant에 맞는 DINOv3 frozen+2-layer head 분류기 생성 헬퍼.",
   "__init__": "DINOv3 백본(pretrained, frozen) + 2-layer MLP 헤드 구성(헤드만 trainable).",
   "forward": "frozen 백본 풀링특징 → head → logits(labels 주면 (loss, logits)).",
 },
 "src/models/timm_backbone.py": {
   "TimmForImageClassification": "timm 백본(num_classes=0 풀링특징) + 표준 분류 헤드 래퍼(baseline 인터페이스).",
   "build_timm_backbone": "timm 모델명을 받아 풀링특징 백본을 생성(detector가 forward_features로 사용).",
   "forward_features": "입력을 백본에 통과시켜 풀링된 (B,C) 특징 반환.",
   "__init__": "timm 백본 + 분류 헤드 구성.",
   "forward": "특징 추출 → 헤드 → logits(labels 주면 (loss, logits)).",
 },
 "src/models/mamba_vision.py": {
   "__init__": "Vision-Mamba(mambapy) 구성: patch-embed → mamba 블록 → 헤드.",
   "_interp_pos": "입력 토큰 수에 맞춰 위치 임베딩을 보간.",
   "forward_features": "patch 토큰 → mamba 인코더 → 평균풀링한 (B,C) 특징.",
   "forward": "특징 → 헤드 → logits(labels 주면 (loss, logits)).",
 },
 "src/train.py": {
   "set_seed": "random/numpy/torch(+cuda) 시드를 고정해 재현성 확보.",
   "setup_logger": "run별 파일+stdout 로거 구성(experiments/<name>/train.log).",
   "package_versions": "지정 패키지들의 설치 버전을 dict로 수집(config.snapshot 기록용).",
   "git_state": "현재 git HEAD 커밋 해시(없으면 표식) 반환 — 재현 추적용.",
   "cosine_warmup_lr": "warmup 후 cosine 감쇠 학습률 스케줄 값을 step에 대해 계산.",
   "resolve_outdir": "출력 디렉토리 결정. 기존 run은 덮지 않고 _vN으로 분기, smoke는 _smoke_ 격리.",
   "run_classification": "분류 spec 학습/평가 루프: 로더·모델 구성, head/trainable만 옵티마이저, epoch별 valid 평가·best 체크포인트·predictions·metrics.json.",
   "run_detection": "detection spec 학습/평가 루프: 단일박스 GIoU+L1(양성만)+objectness BCE, 이미지단위 검출+IoU 평가.",
   "main": "CLI 진입점: spec 로드 → task 분기 → 표준 산출물 생성. 실패해도 metrics.json(status:failed) 기록.",
   "_targets_to_tensors": "detection 타깃 리스트를 (gt_boxes, has_box) 텐서로 변환(빈 박스=음성→has_box 0).",
 },
 "src/losses.py": {
   "__init__": "FocalLoss 설정(gamma, 클래스 가중 weight=alpha, label_smoothing).",
   "forward": "logits/targets에 대해 FL=(1-p_t)^gamma·CE 계산(log_softmax로 수치 안정).",
   "_check": "내부 sanity check 헬퍼.",
 },
 "src/inference.py": {
   "Pipeline": "데모용 로드된 한 파이프라인(모델+메타). public()으로 JSON 직렬화.",
   "_load_one": "experiments/<name>의 config.snapshot+best.pt로 모델을 만들어 Pipeline 구성.",
   "list_pipelines": "로드된 전 파이프라인의 공개 메타 목록 반환(/api/pipelines).",
   "_resolve_ids": "요청의 pipelines 인자('all' 또는 id 리스트)를 실제 파이프라인 id로 해석.",
   "_detection_input": "PIL을 detection 입력(img_size 정사각 resize+ImageNet 정규화) 텐서로 변환.",
   "public": "Pipeline의 외부 노출용 메타(id/arch/task/지표 등) dict.",
 },
 "src/vqa.py": {
   "_device_index": "cuda 디바이스 인덱스 정수 반환(pipeline device 인자용).",
 },
 "demo/app.py": {
   "_build_valid_index": "manifest의 valid 행을 정렬해 안정적 id(0..N-1)와 메타 인덱스 구성.",
   "_startup": "서버 시작 시 추론 레지스트리(파이프라인)와 valid 인덱스를 1회 로드.",
   "api_pipelines": "GET /api/pipelines — 로드된 파이프라인 목록(arch/task/지표) 반환.",
   "api_valid_images": "GET /api/valid-images — klass 필터·페이지네이션된 valid 이미지 목록.",
   "api_valid_image_raw": "GET /api/valid-images/{id}/raw — 해당 valid 이미지 파일 응답.",
   "api_predict": "POST /api/predict — 업로드/valid 이미지에 선택 파이프라인들로 분류+detection 추론.",
   "api_vqa": "POST /api/vqa — 이미지 + (오디오 STT 또는 텍스트) 질문으로 SmolVLM VQA 답변.",
   "index": "GET / — 데모 프론트엔드(index.html) 서빙.",
 },
 "data/verify_pairs.py": {"listdir": "디렉토리 파일 목록(없으면 빈 리스트) — 라벨↔이미지 매칭 검증용."},
 "data/split_by_disease.py": {"link": "target을 src로 가리키는 상대 심링크 생성(기존 링크는 교체)."},
 "data/draw_bbox.py": {"load_font": "사용 가능한 트루타입 폰트를 size로 로드(없으면 기본 폰트)."},
 "data/analyze.py": {
   "vc": "Series의 값별 개수를 정렬된 dict로(JSON 직렬화용).",
   "save": "matplotlib figure를 tight_layout으로 PNG 저장 후 close.",
   "montage": "이미지(선택적 bbox)들을 그리드로 모아 한 장의 샘플 몽타주로 저장.",
   "collect": "split/클래스(코드)별 (이미지경로, 박스) 표본을 n개까지 결정적으로 수집.",
 },
}

def add_docstrings(path, mapping):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    inserts = []  # (lineno_1based_of_first_body_stmt, indent, text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in mapping:
            continue
        if ast.get_docstring(node) is not None:
            continue
        if not node.body:
            continue
        first = node.body[0]
        indent = " " * first.col_offset
        text = mapping[node.name].replace('"""', "'")
        inserts.append((first.lineno, indent, f'{indent}"""{text}"""\n'))
    # 같은 줄에 여러 삽입 방지: 아래에서 위로 처리
    seen = set()
    for lineno, indent, doc in sorted(inserts, key=lambda x: -x[0]):
        if lineno in seen:
            continue
        seen.add(lineno)
        lines.insert(lineno - 1, doc)
    out = "".join(lines)
    ast.parse(out)  # 검증: 삽입 후에도 파싱돼야 함
    open(path, "w", encoding="utf-8").write(out)
    return len(seen)

total = 0
for path, mapping in DOC.items():
    try:
        n = add_docstrings(path, mapping)
        total += n
        print(f"{path}: +{n} docstrings")
    except FileNotFoundError:
        print(f"{path}: MISSING (skip)")
print("total docstrings added:", total)
