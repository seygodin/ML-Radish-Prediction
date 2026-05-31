"""Generate frozen manifests and the data card.

Outputs (under _workspace/data/):
  - manifest_classification.csv : every kept sample for classification settings,
    with a column per setting indicating the assigned label (-1 = not in setting),
    plus a `train_split` column reflecting the post-downsample TRAIN membership.
  - manifest_detection.csv      : every kept disease (and optionally normal) sample.
  - data_card.md                : per-setting train/valid counts (post-downsample),
    transforms, skip log, public API signatures.

Coordinates are in REAL image pixel coordinates (read from the file, EXIF-applied).

Run:  ./.venv/bin/python -m src.data.build_manifest
"""

from __future__ import annotations

import csv
import os
from collections import Counter

from .core import REPO_ROOT, Sample, build_catalog
from .loaders import (
    CLASSIFICATION_SETTINGS,
    _downsample,
    build_classification_loaders,
    build_detection_loaders,
)

OUT_DIR = os.path.join(REPO_ROOT, "_workspace", "data")
SEED = 42

CLS_HEADER = [
    "split", "image_path", "label_path", "klass", "disease_code", "risk",
    "x0", "y0", "x1", "y1",
    "label_normal_vs_d3", "label_normal_vs_d4", "label_normal_d3_d4",
    "in_train_normal_vs_d3", "in_train_normal_vs_d4", "in_train_normal_d3_d4",
]
DET_HEADER = [
    "split", "image_path", "label_path", "klass", "disease_code", "risk",
    "x0", "y0", "x1", "y1",
]


def _rel(p: str) -> str:
    return os.path.relpath(p, REPO_ROOT)


def _row_base(s: Sample):
    return {
        "split": s.split,
        "image_path": _rel(s.image_path),
        "label_path": _rel(s.label_path),
        "klass": s.klass,
        "disease_code": s.disease_code,
        "risk": s.risk,
        "x0": round(s.x0, 2), "y0": round(s.y0, 2),
        "x1": round(s.x1, 2), "y1": round(s.y1, 2),
    }


def _train_membership():
    """Return per-setting set of TRAIN image_paths kept after downsampling, and
    counts dicts, by reusing the exact loader logic."""
    cat = build_catalog()
    membership = {}
    train_counts = {}
    valid_counts = {}

    # rebuild the same downsampled train sets the loaders use
    # normal_vs_d3
    d3_tr = list(cat.d3["train"])
    norm_d3 = _downsample(cat.normal["train"], len(d3_tr), SEED)
    membership["normal_vs_d3"] = {s.image_path for s in (norm_d3 + d3_tr)}
    # normal_vs_d4
    d4_tr = list(cat.d4["train"])
    norm_d4 = _downsample(cat.normal["train"], len(d4_tr), SEED)
    membership["normal_vs_d4"] = {s.image_path for s in (norm_d4 + d4_tr)}
    # normal_d3_d4
    n = min(len(cat.normal["train"]), len(cat.d3["train"]), len(cat.d4["train"]))
    norm3 = _downsample(cat.normal["train"], n, SEED)
    d3_3 = _downsample(cat.d3["train"], n, SEED)
    d4_3 = _downsample(cat.d4["train"], n, SEED)
    membership["normal_d3_d4"] = {s.image_path for s in (norm3 + d3_3 + d4_3)}

    # counts straight from the loaders (authoritative)
    for setting in CLASSIFICATION_SETTINGS:
        _, _, meta = build_classification_loaders(
            setting, num_workers=0, seed=SEED)
        train_counts[setting] = meta["train_counts"]
        valid_counts[setting] = meta["valid_counts"]
    return membership, train_counts, valid_counts


CLS_LABEL_MAPS = {
    "normal_vs_d3": {"normal": 0, "disease_3": 1},
    "normal_vs_d4": {"normal": 0, "disease_4": 1},
    "normal_d3_d4": {"normal": 0, "disease_3": 1, "disease_4": 2},
}


def write_classification_manifest(membership) -> int:
    cat = build_catalog()
    all_samples = []
    for split in ("train", "valid"):
        all_samples += cat.normal[split] + cat.d3[split] + cat.d4[split]
    all_samples.sort(key=lambda s: (s.split, s.image_path))

    path = os.path.join(OUT_DIR, "manifest_classification.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CLS_HEADER)
        w.writeheader()
        for s in all_samples:
            row = _row_base(s)
            for setting in CLASSIFICATION_SETTINGS:
                lm = CLS_LABEL_MAPS[setting]
                row[f"label_{setting}"] = lm.get(s.klass, -1)
                in_train = (
                    s.split == "train"
                    and s.klass in lm
                    and s.image_path in membership[setting]
                )
                row[f"in_train_{setting}"] = int(in_train)
            w.writerow(row)
    return len(all_samples)


def write_detection_manifest() -> int:
    cat = build_catalog()
    samples = []
    for split in ("train", "valid"):
        samples += cat.d3[split] + cat.d4[split]
    samples.sort(key=lambda s: (s.split, s.image_path))

    path = os.path.join(OUT_DIR, "manifest_detection.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DET_HEADER)
        w.writeheader()
        for s in samples:
            w.writerow(_row_base(s))
    return len(samples)


def write_data_card(membership, train_counts, valid_counts, n_cls, n_det):
    cat = build_catalog()
    skip = Counter(reason for _, reason in cat.skipped)

    raw = {
        "train": {
            "normal": len(cat.normal["train"]),
            "disease_3": len(cat.d3["train"]),
            "disease_4": len(cat.d4["train"]),
        },
        "valid": {
            "normal": len(cat.normal["valid"]),
            "disease_3": len(cat.d3["valid"]),
            "disease_4": len(cat.d4["valid"]),
        },
    }

    lines = []
    A = lines.append
    A("# Radish disease — data card\n")
    A("재현 가능한 PyTorch 데이터 파이프라인 산출 요약. "
      "데이터 사실은 `CLAUDE.md` / `report/REPORT.md` 단일 출처를 따른다.\n")
    A(f"- seed: **{SEED}** (downsample 동결), valid는 제공 split 그대로(재분할/다운샘플 없음).")
    A("- 이미지 크기는 **JSON이 아니라 실제 파일에서** 읽음(zero-dim 라벨 43건 방어). "
      "EXIF `exif_transpose` 후 RGB. 확장자 대소문자 무시 매칭. bbox는 실제 크기 기준 픽셀 xyxy.\n")

    A("## Raw kept samples (스캔 후, 다운샘플 전)\n")
    A("| split | normal | disease_3 | disease_4 |")
    A("|-------|-------:|----------:|----------:|")
    for sp in ("train", "valid"):
        r = raw[sp]
        A(f"| {sp} | {r['normal']} | {r['disease_3']} | {r['disease_4']} |")
    A("")

    A("## Classification settings — counts (다운샘플 후)\n")
    for setting in CLASSIFICATION_SETTINGS:
        A(f"### {setting}\n")
        A("| split | " + " | ".join(train_counts[setting].keys()) + " |")
        A("|-------|" + "|".join(["------:"] * len(train_counts[setting])) + "|")
        A("| train | " + " | ".join(str(v) for v in train_counts[setting].values()) + " |")
        A("| valid | " + " | ".join(str(v) for v in valid_counts[setting].values()) + " |")
        A("")
        A("> train normal은 disease 수에 맞춰 다운샘플(seed 고정). valid는 원 분포.\n")

    A("## Detection — counts\n")
    A("torchvision detection 형식. 기본 disease만(질병 단일 클래스 label=1). "
      "`include_normal=True`면 train에서 normal을 disease 수만큼 다운샘플해 추가, valid엔 normal 전체 추가.\n")
    A("| split | disease_3 | disease_4 |")
    A("|-------|----------:|----------:|")
    A(f"| train | {raw['train']['disease_3']} | {raw['train']['disease_4']} |")
    A(f"| valid | {raw['valid']['disease_3']} | {raw['valid']['disease_4']} |")
    A("")

    A("## Transforms\n")
    A("- **classification train**: RandomResizedCrop(img_size, scale 0.6–1.0) + "
      "RandomHorizontalFlip(0.5) + ColorJitter(b/c/s=0.2, h=0.02) + ToTensor + ImageNet Normalize.")
    A("- **classification eval**: Resize(round(img_size*256/224)) + CenterCrop(img_size) + "
      "ToTensor + ImageNet Normalize.")
    A("- **detection train**: Resize→(img_size,img_size) + box scale, HFlip(0.5, box-sync), "
      "brightness jitter, ToTensor + ImageNet Normalize, box clamp.")
    A("- **detection eval**: Resize→(img_size,img_size) + box scale, ToTensor + ImageNet Normalize.")
    A("- ImageNet mean/std = (0.485,0.456,0.406)/(0.229,0.224,0.225).\n")

    A("## Skipped items (조용한 누락 없음)\n")
    if skip:
        A("| reason | count |")
        A("|--------|------:|")
        for reason, c in sorted(skip.items()):
            A(f"| {reason} | {c} |")
    else:
        A("스킵 0건 (모든 라벨이 실제 이미지와 매칭되고 크기/박스 유효).")
    A("")

    A("## Manifests\n")
    A(f"- `_workspace/data/manifest_classification.csv` — {n_cls} rows "
      "(전 split의 normal/d3/d4 전체; 세팅별 label과 `in_train_*` 멤버십 컬럼 포함, "
      "좌표는 실제 크기 기준 xyxy).")
    A(f"- `_workspace/data/manifest_detection.csv` — {n_det} rows (전 split disease만).\n")

    A("## Public API (experiment-runner가 import)\n")
    A("```python")
    A("from src.data import build_classification_loaders, build_detection_loaders")
    A("")
    A("def build_classification_loaders(setting, img_size=224, batch_size=32,")
    A("                                 num_workers=8, seed=42):")
    A('    # setting in {"normal_vs_d3","normal_vs_d4","normal_d3_d4"}')
    A("    # -> (train_loader, valid_loader, meta)")
    A("    #   loaders yield (images FloatTensor[B,3,H,W], labels LongTensor[B])")
    A("    #   meta: {num_classes, class_names, train_counts, valid_counts, class_weights}")
    A("    #   labels: normal=0; 2-class disease=1; 3-class normal=0/d3=1/d4=2")
    A("")
    A("def build_detection_loaders(img_size=512, batch_size=8, num_workers=8,")
    A("                            seed=42, include_normal=False):")
    A("    # -> (train_loader, valid_loader, meta)")
    A("    #   loaders yield (images list[FloatTensor[3,H,W]],")
    A("    #                  targets list[dict(boxes FloatTensor[N,4] xyxy, labels LongTensor[N])])")
    A("    #   disease only by default; single class (1 = disease).")
    A("```")
    A("")
    A("## Known limitations\n")
    A("- valid disease_4 = 24장 → 해당 지표 신뢰구간 넓게 해석.")
    A("- bbox는 거친 단일 박스(면적 중앙값 ≈50%, 중앙 집중) — 병변 핀포인트 아님. "
      "정상에도 박스 존재(분류는 박스 미사용).")
    A("- 단일 시즌(2020-10~2021-01) — 외부 일반화 한계.")
    A("- normal_d3_d4 3-class는 disease_4(train 227)에 맞춰 전 type을 동일 수로 다운샘플 → 표본 작음.")

    path = os.path.join(OUT_DIR, "data_card.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Regenerate from source truth: force a full image scan (never read a
    # possibly-stale manifest), so the skip log and real sizes are authoritative.
    build_catalog(force=True, use_manifest=False)
    membership, train_counts, valid_counts = _train_membership()
    n_cls = write_classification_manifest(membership)
    n_det = write_detection_manifest()
    write_data_card(membership, train_counts, valid_counts, n_cls, n_det)
    print(f"wrote manifest_classification.csv ({n_cls} rows)")
    print(f"wrote manifest_detection.csv ({n_det} rows)")
    print("wrote data_card.md")
    cat = build_catalog()
    print(f"skipped: {len(cat.skipped)} items")


if __name__ == "__main__":
    main()
