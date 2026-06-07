#!/usr/bin/env python3
"""Draw annotation bboxes onto diseased images, mirroring the by_disease layout.

For each by_disease/<split>/disease_<code>/labels/<json>, open the matching image
and write an annotated copy to by_disease/<split>/disease_<code>/image_w_bbox/<img>.
"""
import json, os
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "by_disease")
SPLITS = ["train", "valid"]
CODES = [3, 4]


def load_font(size):
    """사용 가능한 트루타입 폰트를 size로 로드(없으면 기본 폰트)."""
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


summary = Counter()
errors = []
for split in SPLITS:
    for code in CODES:
        base = os.path.join(OUT, split, f"disease_{code}")
        label_dir = os.path.join(base, "labels")
        img_dir = os.path.join(base, "images")
        out_dir = os.path.join(base, "image_w_bbox")
        if not os.path.isdir(label_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        img_by_lower = {f.lower(): f for f in os.listdir(img_dir)}

        for jf in os.listdir(label_dir):
            if not jf.endswith(".json"):
                continue
            ann = json.load(open(os.path.join(label_dir, jf)))
            a = ann["annotations"]
            img_name = img_by_lower.get(jf[:-5].lower())
            if img_name is None:
                errors.append(f"{split}/disease_{code}: no image for {jf}")
                continue
            try:
                im = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            except Exception as e:  # corrupt/unreadable image
                errors.append(f"{split}/disease_{code}: cannot open {img_name}: {e}")
                continue

            draw = ImageDraw.Draw(im)
            lw = max(2, round(min(im.size) * 0.005))      # scale line to image size
            fsz = max(14, round(min(im.size) * 0.03))
            font = load_font(fsz)
            tag = f"d{a.get('disease')} risk{a.get('risk')}"
            for p in a.get("points", []):
                box = [p["xtl"], p["ytl"], p["xbr"], p["ybr"]]
                draw.rectangle(box, outline=(255, 0, 0), width=lw)
                tx, ty = p["xtl"], max(0, p["ytl"] - fsz - 4)
                l, t, r, b = draw.textbbox((tx, ty), tag, font=font)
                draw.rectangle([l - 2, t - 2, r + 2, b + 2], fill=(255, 0, 0))
                draw.text((tx, ty), tag, fill=(255, 255, 255), font=font)

            im.save(os.path.join(out_dir, img_name))
            summary[(split, code)] += 1

print("=== image_w_bbox created ===")
for k in sorted(summary):
    print(f"  {k[0]:5s} disease_{k[1]}: {summary[k]}")
print("total annotated:", sum(summary.values()))
if errors:
    print(f"\n{len(errors)} error(s):")
    for e in errors[:20]:
        print("  ", e)
