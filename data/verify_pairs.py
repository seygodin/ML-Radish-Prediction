#!/usr/bin/env python3
"""Verify that every label JSON matches a source image (and report orphans)."""
import os

PAIRS = [
    ("train normal",  "train/[라벨]무_0.정상", "train/[원천]무_0.정상"),
    ("train disease", "train/[라벨]무_1.질병", "train/[원천]무_1.질병"),
    ("valid normal",  "valid/[라벨]무_0.정상", "valid/[원천]무_0.정상"),
    ("valid disease", "valid/[라벨]무_1.질병", "valid/[원천]무_1.질병"),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def listdir(d):
    """디렉토리 파일 목록(없으면 빈 리스트) — 라벨↔이미지 매칭 검증용."""
    p = os.path.join(HERE, d)
    return os.listdir(p) if os.path.isdir(p) else []


all_ok = True
for name, label_dir, src_dir in PAIRS:
    # label file "<image>.json"  ->  expected image filename "<image>"
    labels = [f for f in listdir(label_dir) if f.endswith(".json")]
    expected_imgs = {f[:-5] for f in labels}          # strip ".json"
    images = {f for f in listdir(src_dir) if not f.endswith(".json")}

    # case-insensitive map for images (.jpg vs .JPG)
    img_lower = {f.lower(): f for f in images}

    missing_img = sorted(s for s in expected_imgs
                         if s not in images and s.lower() not in img_lower)
    orphan_img = sorted(images - expected_imgs
                        - {img_lower.get(e.lower()) for e in expected_imgs})

    ok = not missing_img and not orphan_img
    all_ok &= ok
    print(f"\n=== {name} ===")
    print(f"  labels(json)   : {len(labels)}")
    print(f"  images         : {len(images)}")
    print(f"  json w/o image : {len(missing_img)}")
    print(f"  image w/o json : {len(orphan_img)}")
    print(f"  STATUS         : {'OK (all matched)' if ok else 'MISMATCH'}")
    for m in missing_img[:5]:
        print(f"    [no image] {m}")
    for o in orphan_img[:5]:
        print(f"    [no json ] {o}")

print("\n==============================")
print("RESULT:", "ALL MATCHED" if all_ok else "MISMATCHES FOUND")
