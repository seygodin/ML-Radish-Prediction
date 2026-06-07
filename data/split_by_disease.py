#!/usr/bin/env python3
"""Separate diseased data by disease class using relative symlinks.

Layout created:
  by_disease/<split>/disease_<code>/images/<img>   -> ../../../../<split>/[원천]무_1.질병/<img>
  by_disease/<split>/disease_<code>/labels/<json>  -> ../../../../<split>/[라벨]무_1.질병/<json>
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "by_disease")
SPLITS = ["train", "valid"]


def link(target, linkpath):
    """target을 src로 가리키는 상대 심링크 생성(기존 링크는 교체)."""
    if os.path.islink(linkpath) or os.path.exists(linkpath):
        os.remove(linkpath)
    rel = os.path.relpath(target, os.path.dirname(linkpath))
    os.symlink(rel, linkpath)


summary = Counter()
for split in SPLITS:
    label_dir = os.path.join(HERE, split, "[라벨]무_1.질병")
    src_dir = os.path.join(HERE, split, "[원천]무_1.질병")
    img_by_lower = {f.lower(): f for f in os.listdir(src_dir)}

    for jf in os.listdir(label_dir):
        if not jf.endswith(".json"):
            continue
        a = json.load(open(os.path.join(label_dir, jf)))["annotations"]
        code = a["disease"]
        img_name = jf[:-5]  # strip ".json"
        real_img = img_by_lower.get(img_name.lower())
        if real_img is None:
            print(f"  WARN no image for {jf}")
            continue

        base = os.path.join(OUT, split, f"disease_{code}")
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)
        link(os.path.join(src_dir, real_img), os.path.join(base, "images", real_img))
        link(os.path.join(label_dir, jf), os.path.join(base, "labels", jf))
        summary[(split, code)] += 1

print("=== created (split, disease_code): count ===")
for k in sorted(summary):
    print(f"  {k[0]:5s} disease_{k[1]}: {summary[k]}")
print("total links:", sum(summary.values()) * 2)
