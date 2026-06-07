"""Robust image+label loading and manifest construction for the radish disease dataset.

This module centralizes the *verified data traps* (see CLAUDE.md / report/REPORT.md):
  - 43 diseased labels have width=height=0 in their JSON -> read real size from the
    image file, never from JSON. bbox normalization is done against the real size.
  - EXIF orientation: apply ImageOps.exif_transpose before converting to RGB.
  - Mixed extensions (.jpg/.jpeg/.JPG): case-insensitive matching.
  - valid split is the dataset-provided split (never re-split).
  - exactly one bbox per image; normal images also carry a box (whole-radish crop),
    so box presence does NOT separate normal vs disease.

All paths are resolved relative to the repository root (parent of `src/`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------
# core.py is at <repo>/src/data/core.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# Normal source/label dirs per split (Korean names with brackets).
NORMAL_IMG_DIR = {
    "train": os.path.join(DATA_ROOT, "train", "[원천]무_0.정상"),
    "valid": os.path.join(DATA_ROOT, "valid", "[원천]무_0.정상"),
}
NORMAL_LBL_DIR = {
    "train": os.path.join(DATA_ROOT, "train", "[라벨]무_0.정상"),
    "valid": os.path.join(DATA_ROOT, "valid", "[라벨]무_0.정상"),
}


def by_disease_dir(split: str, disease_code: int, kind: str) -> str:
    """kind in {'images','labels'}."""
    return os.path.join(DATA_ROOT, "by_disease", split, f"disease_{disease_code}", kind)


# ----------------------------------------------------------------------------
# Robust loaders (shared by every Dataset)
# ----------------------------------------------------------------------------
def load_image(path: str) -> Image.Image:
    """EXIF-corrected RGB image. Real size is `im.size` after this."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def real_image_size(path: str) -> tuple[int, int]:
    """(width, height) read from the image file with EXIF orientation applied.

    NEVER trust JSON width/height (43 zero-dim labels). We must mirror the EXIF
    transpose so bbox coordinates (which are in the EXIF-corrected frame after
    exif_transpose during loading) align with the size used for normalization.
    """
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.size  # (w, h)


def _dir_index(image_dir: str, _cache: dict = {}) -> dict[str, str]:
    """Case-insensitive filename index for a directory (cached per dir)."""
    if image_dir not in _cache:
        try:
            _cache[image_dir] = {f.lower(): f for f in os.listdir(image_dir)}
        except FileNotFoundError:
            _cache[image_dir] = {}
    return _cache[image_dir]


def find_image(stem: str, image_dir: str) -> Optional[str]:
    """Resolve an image filename case-insensitively. `stem` is the image filename
    (e.g. 'V006_..._1.jpg'). Returns the on-disk filename or None."""
    return _dir_index(image_dir).get(stem.lower())


def parse_label(json_path: str, image_path: str) -> dict:
    """Parse a label JSON. Image size is read from the real file (not JSON)."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    a = data["annotations"]
    w, h = real_image_size(image_path)
    points = a.get("points") or []
    boxes = [
        [float(p["xtl"]), float(p["ytl"]), float(p["xbr"]), float(p["ybr"])]
        for p in points
    ]
    return dict(
        disease=int(a["disease"]),
        risk=int(a.get("risk", 0)),
        boxes=boxes,
        w=int(w),
        h=int(h),
    )


# ----------------------------------------------------------------------------
# Manifest sample record
# ----------------------------------------------------------------------------
@dataclass
class Sample:
    """한 샘플의 메타데이터(경로·클래스·disease 코드·risk·실제 크기 기준 bbox). manifest 행과 1:1."""
    split: str
    image_path: str
    label_path: str
    klass: str           # human-readable: 'normal' | 'disease_3' | 'disease_4'
    disease_code: int    # 0 / 3 / 4
    risk: int
    # bbox in real-image pixel coords (xyxy), clipped to image bounds.
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    w: int = 0
    h: int = 0
    skip_reason: str = ""  # non-empty => excluded


def _clip_box(box, w, h):
    """bbox(xyxy)를 이미지 경계 [0,w]/[0,h]로 클립하고 좌표 역전 시 정렬해 반환."""
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(x0, w))
    y0 = max(0.0, min(y0, h))
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _scan_normal(split: str, skipped: list) -> list[Sample]:
    """split의 정상 라벨/이미지를 스캔해 Sample 목록 생성(매칭 실패·zero-size는 skip 로그)."""
    img_dir = NORMAL_IMG_DIR[split]
    lbl_dir = NORMAL_LBL_DIR[split]
    out = []
    for lbl_name in sorted(os.listdir(lbl_dir)):
        if not lbl_name.lower().endswith(".json"):
            continue
        stem = lbl_name[:-5]  # strip '.json' -> '<img>.jpg'
        img_name = find_image(stem, img_dir)
        if img_name is None:
            skipped.append((os.path.join(lbl_dir, lbl_name), "image_not_found"))
            continue
        image_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, lbl_name)
        try:
            meta = parse_label(label_path, image_path)
        except Exception as e:  # noqa: BLE001
            skipped.append((label_path, f"parse_error:{type(e).__name__}"))
            continue
        if meta["w"] <= 0 or meta["h"] <= 0:
            skipped.append((label_path, "zero_real_size"))
            continue
        box = meta["boxes"][0] if meta["boxes"] else [0, 0, meta["w"], meta["h"]]
        box = _clip_box(box, meta["w"], meta["h"])
        out.append(Sample(
            split=split, image_path=image_path, label_path=label_path,
            klass="normal", disease_code=0, risk=meta["risk"],
            x0=box[0], y0=box[1], x1=box[2], y1=box[3], w=meta["w"], h=meta["h"],
        ))
    return out


def _scan_disease(split: str, disease_code: int, skipped: list) -> list[Sample]:
    """split·disease_code의 질병 라벨/이미지를 스캔(빈/퇴화 bbox는 detection 안전성 위해 skip)."""
    img_dir = by_disease_dir(split, disease_code, "images")
    lbl_dir = by_disease_dir(split, disease_code, "labels")
    out = []
    for lbl_name in sorted(os.listdir(lbl_dir)):
        if not lbl_name.lower().endswith(".json"):
            continue
        stem = lbl_name[:-5]
        img_name = find_image(stem, img_dir)
        if img_name is None:
            skipped.append((os.path.join(lbl_dir, lbl_name), "image_not_found"))
            continue
        image_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, lbl_name)
        try:
            meta = parse_label(label_path, image_path)
        except Exception as e:  # noqa: BLE001
            skipped.append((label_path, f"parse_error:{type(e).__name__}"))
            continue
        if meta["w"] <= 0 or meta["h"] <= 0:
            skipped.append((label_path, "zero_real_size"))
            continue
        if not meta["boxes"]:
            skipped.append((label_path, "no_bbox"))
            continue
        box = _clip_box(meta["boxes"][0], meta["w"], meta["h"])
        # degenerate (zero-area) box -> skip for detection-safety
        if box[2] <= box[0] or box[3] <= box[1]:
            skipped.append((label_path, "degenerate_bbox"))
            continue
        out.append(Sample(
            split=split, image_path=image_path, label_path=label_path,
            klass=f"disease_{disease_code}", disease_code=disease_code,
            risk=meta["risk"], x0=box[0], y0=box[1], x1=box[2], y1=box[3],
            w=meta["w"], h=meta["h"],
        ))
    return out


@dataclass
class Catalog:
    """All kept samples grouped by class, plus the skip log."""
    normal: dict[str, list[Sample]] = field(default_factory=dict)   # split -> samples
    d3: dict[str, list[Sample]] = field(default_factory=dict)
    d4: dict[str, list[Sample]] = field(default_factory=dict)
    skipped: list = field(default_factory=list)  # (path, reason)


_CATALOG_CACHE: Optional[Catalog] = None

# Frozen manifest produced by src/data/build_manifest.py. When present we
# reconstruct the catalog from it (real sizes/boxes already resolved) instead of
# re-opening ~13k images — turning a multi-minute scan into a sub-second CSV read.
MANIFEST_CLS = os.path.join(
    REPO_ROOT, "_workspace", "data", "manifest_classification.csv")


def _catalog_from_manifest(path: str) -> Optional[Catalog]:
    """동결된 manifest CSV에서 Catalog를 복원(이미지 재스캔 없이 sub-second). 누락/무효 시 None."""
    import csv
    try:
        fh = open(path, "r", encoding="utf-8", newline="")
    except FileNotFoundError:
        return None
    cat = Catalog()
    for split in ("train", "valid"):
        cat.normal[split], cat.d3[split], cat.d4[split] = [], [], []
    with fh:
        reader = csv.DictReader(fh)
        required = {"split", "image_path", "label_path", "klass",
                    "disease_code", "risk", "x0", "y0", "x1", "y1"}
        if not required.issubset(reader.fieldnames or []):
            return None
        for r in reader:
            split = r["split"]
            if split not in ("train", "valid"):
                continue
            s = Sample(
                split=split,
                image_path=os.path.join(REPO_ROOT, r["image_path"]),
                label_path=os.path.join(REPO_ROOT, r["label_path"]),
                klass=r["klass"], disease_code=int(r["disease_code"]),
                risk=int(r["risk"]),
                x0=float(r["x0"]), y0=float(r["y0"]),
                x1=float(r["x1"]), y1=float(r["y1"]),
            )
            if s.klass == "normal":
                cat.normal[split].append(s)
            elif s.klass == "disease_3":
                cat.d3[split].append(s)
            elif s.klass == "disease_4":
                cat.d4[split].append(s)
    # sanity: must have found samples in every bucket we expect
    if not (cat.normal["train"] and cat.d3["train"] and cat.d4["train"]):
        return None
    return cat


def build_catalog(force: bool = False, use_manifest: bool = True) -> Catalog:
    """Build (and cache) the sample catalog.

    Fast path: if the frozen classification manifest exists, reconstruct the
    catalog from it (sizes/boxes already resolved). Falls back to a full image
    scan if the manifest is missing/invalid or `use_manifest=False`.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and not force:
        return _CATALOG_CACHE

    cat = None
    if use_manifest:
        cat = _catalog_from_manifest(MANIFEST_CLS)

    if cat is None:
        cat = Catalog()
        for split in ("train", "valid"):
            cat.normal[split] = _scan_normal(split, cat.skipped)
            cat.d3[split] = _scan_disease(split, 3, cat.skipped)
            cat.d4[split] = _scan_disease(split, 4, cat.skipped)

    _CATALOG_CACHE = cat
    return cat
