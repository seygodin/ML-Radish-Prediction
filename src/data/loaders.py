"""Public data-loader API for the radish disease baselines.

Contract (callers depend on these exact signatures):

    build_classification_loaders(setting, img_size=224, batch_size=32,
                                 num_workers=8, seed=42)
        setting in {"normal_vs_d3","normal_vs_d4","normal_d3_d4"}
        -> (train_loader, valid_loader, meta)
        loaders yield (images FloatTensor[B,3,H,W], labels LongTensor[B])
        labels: normal=0; 2-class disease=1; 3-class normal=0/d3=1/d4=2
        meta: {num_classes, class_names, train_counts, valid_counts, class_weights}

    build_detection_loaders(img_size=512, batch_size=8, num_workers=8,
                            seed=42, include_normal=False)
        -> (train_loader, valid_loader, meta)
        loaders yield (images list[FloatTensor[3,H,W]],
                       targets list[dict(boxes FloatTensor[N,4] xyxy, labels LongTensor[N])])
        disease only by default; single class (1 = disease).
        include_normal=True: NORMAL images are supplied as NEGATIVES -- empty
        boxes (shape [0,4]) and empty labels (shape [0]) so objectness gets true
        negative targets. (Without this the baseline collapsed to objectness~=1.0
        everywhere because every image carried a positive box+label.)

Class balance: NORMAL is downsampled in TRAIN ONLY with a fixed seed.
VALID is the dataset-provided split, never downsampled.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import core
from .core import Sample, build_catalog, load_image
from .transforms import (
    DetectionTransform,
    classification_eval_transform,
    classification_train_transform,
    classification_train_transform_strong,
)

CLASSIFICATION_SETTINGS = ("normal_vs_d3", "normal_vs_d4", "normal_d3_d4")


# ----------------------------------------------------------------------------
# Downsampling helpers (train only, deterministic)
# ----------------------------------------------------------------------------
def _downsample(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """`samples`에서 `n`개를 seed 고정으로 무작위 추출(클래스 균형/비율 축소용).

    n이 전체보다 크거나 같으면 원본을 그대로 반환. 추출 인덱스를 정렬해 반환 순서를
    결정적(reproducible)으로 유지한다 — 같은 seed면 항상 같은 부분집합.
    """
    if n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(samples)), n))
    return [samples[i] for i in idx]


# ----------------------------------------------------------------------------
# In-RAM image cache (decode + resize ONCE, reuse every epoch)
# ----------------------------------------------------------------------------
# The verified bottleneck: __getitem__ decoded the full-res JPEG (up to
# 6000x4000, multi-MB) on every access, every epoch -> the GPU starved waiting
# on CPU JPEG decode. Datasets here are small (hundreds train / ~1400 valid), so
# we decode+downscale each image once at construction and keep the small PIL in a
# list. DataLoader workers are forked AFTER this, so they inherit the cache via
# copy-on-write (no re-decode in any worker, no per-worker duplication of decode
# work). Per-access cost drops to a cheap resize/crop on a small image.

def _decode_resized(path: str, short_side: int) -> Image.Image:
    """Decode (EXIF-corrected RGB) and downscale so the shorter side == short_side
    (never upscale). Aspect ratio preserved."""
    img = load_image(path)
    w, h = img.size
    s = short_side / float(min(w, h))
    if s < 1.0:  # only downscale
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def _decode_square_with_box(path: str, img_size: int, box):
    """Decode and resize to a square img_size; scale the box into that frame.
    Returns (square_img, scaled_box_xyxy_list)."""
    img = load_image(path)
    w, h = img.size
    img = img.resize((img_size, img_size), Image.BILINEAR)
    sx, sy = img_size / float(w), img_size / float(h)
    b = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    return img, b


def _parallel(fn, items, workers=16):
    """`fn`을 `items`에 스레드풀로 병렬 적용(순서 보존). 캐시 빌드 시 이미지 디코드가
    I/O·libjpeg 바운드라 스레드로 충분히 빨라진다(GIL은 디코드 중 풀림)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    """분류용 Dataset. 생성 시 모든 이미지를 디코드+다운스케일해 RAM에 캐시(`self.cache`)하고,
    `__getitem__`은 캐시된 작은 PIL에 transform(증강·정규화)만 적용한다. 캐시는 fork된
    DataLoader 워커가 copy-on-write로 공유 → 매 epoch 재디코드 없음(위 주석 참조)."""

    def __init__(self, samples: list[Sample], label_map: dict[str, int], transform,
                 cache_short: int):
        """데이터셋 생성: 이미지를 1회 디코드·리사이즈해 RAM 캐시(fork COW 공유)."""
        self.labels = [label_map[s.klass] for s in samples]   # 정수 라벨(normal=0 등)
        self.transform = transform
        # decode+downscale once; workers inherit via fork copy-on-write
        self.cache = _parallel(lambda s: _decode_resized(s.image_path, cache_short),
                               samples)

    def __len__(self):
        """캐시된 샘플 수."""
        return len(self.cache)

    def __getitem__(self, i):
        """캐시 이미지에 transform을 적용해 (텐서, 라벨/타깃) 반환."""
        return self.transform(self.cache[i]), self.labels[i]


class DetectionDataset(Dataset):
    """Detection samples. Disease images are POSITIVES (one box, label 1); normal
    images are NEGATIVES (empty boxes/labels) so objectness has true negatives.

    Without negatives the baseline collapsed to objectness ~= 1.0 everywhere
    ("always disease"). We still decode + cache+resize normal images so they flow
    through the model, but their GT box list is empty -> objectness target 0.
    """

    def __init__(self, samples: list[Sample], transform: DetectionTransform,
                 img_size: int):
        """데이터셋 생성: 이미지를 1회 디코드·리사이즈해 RAM 캐시(fork COW 공유)."""
        self.transform = transform  # pre_resized=True
        # Still decode+resize every image (normals included) so it goes through
        # the network as a negative; the box result is simply dropped for normals.
        out = _parallel(
            lambda s: _decode_square_with_box(
                s.image_path, img_size, [s.x0, s.y0, s.x1, s.y1]),
            samples)
        self.cache = [o[0] for o in out]
        self.is_normal = [s.klass == "normal" for s in samples]
        self.boxes = [
            torch.empty((0, 4), dtype=torch.float32) if normal
            else torch.tensor([o[1]], dtype=torch.float32)
            for o, normal in zip(out, self.is_normal)
        ]

    def __len__(self):
        """캐시된 샘플 수."""
        return len(self.cache)

    def __getitem__(self, i):
        """캐시 이미지에 transform을 적용해 (텐서, 라벨/타깃) 반환."""
        img, boxes = self.transform(self.cache[i], self.boxes[i])
        # labels track box count: disease -> [1], normal (negative) -> [] (shape [0])
        labels = torch.ones((boxes.shape[0],), dtype=torch.long)  # 1 = disease
        return img, {"boxes": boxes, "labels": labels}


def _detection_collate(batch):
    """detection 배치 collate: 이미지/타깃을 가변 길이 리스트로 묶음(박스 수가 달라 stack 불가)."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# ----------------------------------------------------------------------------
# Count / weight helpers
# ----------------------------------------------------------------------------
def _counts(samples: list[Sample], label_map: dict[str, int], class_names: list[str]):
    """class_names별 표본 수 dict 반환(분포 보고·가중치 계산용)."""
    counts = {name: 0 for name in class_names}
    for s in samples:
        counts[s.klass] += 1
    return counts


def _class_weights(train_counts: dict, class_names: list[str]) -> torch.Tensor:
    """역빈도 class weight(평균 1로 정규화). train을 다운샘플로 균형 맞추면 전부 ≈1이 되어
    focal의 alpha 역할은 사실상 무력화되고 focal은 gamma(hard-example 집중)로만 작동한다."""
    n = [max(1, train_counts[name]) for name in class_names]
    total = sum(n)
    k = len(n)
    # inverse-frequency, normalized so weights average to 1
    w = [total / (k * c) for c in n]
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------------------------------------------------------
# Public API: classification
# ----------------------------------------------------------------------------
def build_classification_loaders(
    setting: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    seed: int = 42,
    balance_valid: bool = False,
    aug: str = "default",
    train_ratio: float = 1.0,
):
    """balance_valid=True downsamples the VALID split to a balanced ratio
    (1:1 for binary, 1:1:1 for 3-class) with a fixed seed -- for evaluating on a
    class-balanced valid set. Default False keeps the dataset-provided valid
    distribution (real prevalence). TRAIN is always downsampled to balance.

    aug: TRAIN augmentation strength. "default" = original (RandomResizedCrop +
    HFlip + mild ColorJitter). "strong" = heavier augmentation to mitigate
    overfitting / improve generalization (adds VFlip, rotation, strong
    ColorJitter, TrivialAugmentWide, RandomErasing). VALID transform is
    deterministic and unaffected (fair comparison).

    train_ratio: fraction (0 < train_ratio <= 1.0) of the balanced TRAIN set to
    keep, sampled stratified PER CLASS with the fixed seed (preserves the class
    ratio of the balanced train set). 1.0 (default) keeps the full balanced train
    set unchanged (no behavior change). Used for data-quantity / stability
    analysis. VALID is never affected by train_ratio."""
    if aug not in ("default", "strong"):
        raise ValueError(f"aug must be 'default' or 'strong', got {aug!r}")
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1.0], got {train_ratio!r}")
    if setting not in CLASSIFICATION_SETTINGS:
        raise ValueError(f"setting must be one of {CLASSIFICATION_SETTINGS}, got {setting!r}")

    cat = build_catalog()

    if setting == "normal_vs_d3":
        class_names = ["normal", "disease_3"]
        label_map = {"normal": 0, "disease_3": 1}
        train_disease = list(cat.d3["train"])
        valid_disease = list(cat.d3["valid"])
        n_normal_train = len(train_disease)
        train_normal = _downsample(cat.normal["train"], n_normal_train, seed)
        train_samples = train_normal + train_disease
        if balance_valid:  # 1:1 valid -> downsample normal to disease count
            valid_normal = _downsample(cat.normal["valid"], len(valid_disease), seed)
        else:
            valid_normal = cat.normal["valid"]
        valid_samples = valid_normal + valid_disease

    elif setting == "normal_vs_d4":
        class_names = ["normal", "disease_4"]
        label_map = {"normal": 0, "disease_4": 1}
        train_disease = list(cat.d4["train"])
        valid_disease = list(cat.d4["valid"])
        n_normal_train = len(train_disease)
        train_normal = _downsample(cat.normal["train"], n_normal_train, seed)
        train_samples = train_normal + train_disease
        if balance_valid:  # 1:1 valid
            valid_normal = _downsample(cat.normal["valid"], len(valid_disease), seed)
        else:
            valid_normal = cat.normal["valid"]
        valid_samples = valid_normal + valid_disease

    else:  # normal_d3_d4
        class_names = ["normal", "disease_3", "disease_4"]
        label_map = {"normal": 0, "disease_3": 1, "disease_4": 2}
        n = min(len(cat.normal["train"]), len(cat.d3["train"]), len(cat.d4["train"]))
        train_normal = _downsample(cat.normal["train"], n, seed)
        train_d3 = _downsample(cat.d3["train"], n, seed)
        train_d4 = _downsample(cat.d4["train"], n, seed)
        train_samples = train_normal + train_d3 + train_d4
        if balance_valid:  # 1:1:1 valid -> all classes to the min valid count
            nv = min(len(cat.normal["valid"]), len(cat.d3["valid"]), len(cat.d4["valid"]))
            valid_samples = (_downsample(cat.normal["valid"], nv, seed)
                             + _downsample(cat.d3["valid"], nv, seed)
                             + _downsample(cat.d4["valid"], nv, seed))
        else:
            valid_samples = cat.normal["valid"] + cat.d3["valid"] + cat.d4["valid"]

    # Stratified train subsampling (per class) for data-quantity analysis.
    # Applied AFTER class balancing so the balanced ratio is preserved.
    if train_ratio < 1.0:
        reduced: list[Sample] = []
        for name in class_names:
            cls_samples = [s for s in train_samples if s.klass == name]
            n_keep = round(len(cls_samples) * train_ratio)
            reduced += _downsample(cls_samples, n_keep, seed)
        train_samples = reduced

    # deterministic ordering before shuffle for reproducibility
    train_samples.sort(key=lambda s: s.image_path)
    valid_samples.sort(key=lambda s: s.image_path)

    train_counts = _counts(train_samples, label_map, class_names)
    valid_counts = _counts(valid_samples, label_map, class_names)
    class_weights = _class_weights(train_counts, class_names)

    cache_short = int(round(img_size * 256 / 224))  # match eval resize target
    train_tf = (classification_train_transform_strong(img_size) if aug == "strong"
                else classification_train_transform(img_size))
    train_ds = ClassificationDataset(
        train_samples, label_map, train_tf, cache_short)
    valid_ds = ClassificationDataset(
        valid_samples, label_map, classification_eval_transform(img_size), cache_short)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False, generator=g,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
    )

    meta = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_counts": train_counts,
        "valid_counts": valid_counts,
        "class_weights": class_weights,
        "train_ratio": train_ratio,
    }
    return train_loader, valid_loader, meta


# ----------------------------------------------------------------------------
# Public API: detection
# ----------------------------------------------------------------------------
def build_detection_loaders(
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 8,
    seed: int = 42,
    include_normal: bool = False,
    balance_valid: bool = False,
):
    """balance_valid=True downsamples the VALID normals to a 1:1 ratio with the
    disease (positive) images -- for evaluating image-level disease detection on a
    class-balanced valid set (affects det PR-AUC / fp_rate / mAP; presence_recall
    and positive-only IoU are unchanged). Only meaningful with include_normal=True.
    Default False keeps the full provided valid (real prevalence)."""
    cat = build_catalog()

    train_samples = list(cat.d3["train"]) + list(cat.d4["train"])
    valid_samples = list(cat.d3["valid"]) + list(cat.d4["valid"])

    if include_normal:
        # match normal count to disease count in TRAIN only (valid untouched by default)
        n_normal_train = len(train_samples)
        train_samples = train_samples + _downsample(
            cat.normal["train"], n_normal_train, seed)
        if balance_valid:  # 1:1 valid -> normals downsampled to #disease(=100)
            valid_normal = _downsample(cat.normal["valid"], len(valid_samples), seed)
        else:
            valid_normal = list(cat.normal["valid"])
        valid_samples = valid_samples + valid_normal

    train_samples.sort(key=lambda s: s.image_path)
    valid_samples.sort(key=lambda s: s.image_path)

    train_ds = DetectionDataset(
        train_samples, DetectionTransform(img_size, train=True, pre_resized=True), img_size)
    valid_ds = DetectionDataset(
        valid_samples, DetectionTransform(img_size, train=False, pre_resized=True), img_size)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, collate_fn=_detection_collate, generator=g,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_detection_collate,
        persistent_workers=num_workers > 0,
    )

    def _det_counts(samples):
        """detection 표본의 클래스별(normal/d3/d4) 개수 dict."""
        c = {"disease_3": 0, "disease_4": 0, "normal": 0}
        for s in samples:
            c[s.klass] += 1
        return c

    meta = {
        "num_classes": 2,  # background(0) + disease(1), torchvision convention
        "class_names": ["__background__", "disease"],
        "train_counts": _det_counts(train_samples),
        "valid_counts": _det_counts(valid_samples),
        "include_normal": include_normal,
        "img_size": img_size,
    }
    return train_loader, valid_loader, meta


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
def _smoke():
    print("=" * 70)
    print("SMOKE TEST: radish data pipeline")
    print("=" * 70)
    nw = 2

    for setting in CLASSIFICATION_SETTINGS:
        tl, vl, meta = build_classification_loaders(
            setting, img_size=224, batch_size=16, num_workers=nw, seed=42)
        xb, yb = next(iter(tl))
        print(f"\n[classification: {setting}]")
        print(f"  num_classes={meta['num_classes']} class_names={meta['class_names']}")
        print(f"  train_counts={meta['train_counts']}")
        print(f"  valid_counts={meta['valid_counts']}")
        print(f"  class_weights={meta['class_weights'].tolist()}")
        print(f"  batch images: shape={tuple(xb.shape)} dtype={xb.dtype} "
              f"min={xb.min():.3f} max={xb.max():.3f}")
        uniq = torch.bincount(yb, minlength=meta['num_classes']).tolist()
        print(f"  batch labels: dtype={yb.dtype} dist={uniq}")

    for inc in (False, True):
        tl, vl, meta = build_detection_loaders(
            img_size=512, batch_size=8, num_workers=nw, seed=42, include_normal=inc)
        print(f"\n[detection: include_normal={inc}]")
        print(f"  train_counts={meta['train_counts']} valid_counts={meta['valid_counts']}")
        for name, loader in (("train", tl), ("valid", vl)):
            imgs, targets = next(iter(loader))
            n_pos = sum(int(t["boxes"].shape[0] > 0) for t in targets)
            n_neg = sum(int(t["boxes"].shape[0] == 0) for t in targets)
            print(f"  [{name}] n_images={len(imgs)} img0 shape={tuple(imgs[0].shape)} "
                  f"-> positives(box)={n_pos} negatives(empty)={n_neg}")
            pos = next((t for t in targets if t["boxes"].shape[0] > 0), None)
            neg = next((t for t in targets if t["boxes"].shape[0] == 0), None)
            if pos is not None:
                print(f"    positive: boxes shape={tuple(pos['boxes'].shape)} "
                      f"labels shape={tuple(pos['labels'].shape)} "
                      f"box={[round(v, 1) for v in pos['boxes'][0].tolist()]} "
                      f"labels={pos['labels'].tolist()}")
            if neg is not None:
                print(f"    negative: boxes shape={tuple(neg['boxes'].shape)} "
                      f"labels shape={tuple(neg['labels'].shape)} "
                      f"dtypes=({neg['boxes'].dtype},{neg['labels'].dtype})")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    _smoke()
