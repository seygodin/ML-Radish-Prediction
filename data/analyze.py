#!/usr/bin/env python3
"""Analyze the radish (무) disease dataset: build metadata table, stats, and figures.

Outputs (relative to project root):
  report/metadata.csv     one row per label json
  report/stats.json       summary statistics
  report/figures/*.png    visualizations (English labels, Agg backend)

Run:  ./.venv/bin/python data/analyze.py
"""
import os, json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

LABEL = {
    ("train", "normal"): "train/[라벨]무_0.정상",
    ("train", "disease"): "train/[라벨]무_1.질병",
    ("valid", "normal"): "valid/[라벨]무_0.정상",
    ("valid", "disease"): "valid/[라벨]무_1.질병",
}
IMGDIR = {
    ("train", "normal"): "train/[원천]무_0.정상",
    ("train", "disease"): "train/[원천]무_1.질병",
    ("valid", "normal"): "valid/[원천]무_0.정상",
    ("valid", "disease"): "valid/[원천]무_1.질병",
}


def img_index(split, klass):
    """case-insensitive {lower_name: real_name} for the source image dir."""
    d = os.path.join(DATA, IMGDIR[(split, klass)])
    return {f.lower(): f for f in os.listdir(d)}, d


# ---------------------------------------------------------------- build table
rows = []
zero_dim = 0
for (split, klass), rel in LABEL.items():
    p = os.path.join(DATA, rel)
    idx, imdir = img_index(split, klass)
    for f in sorted(os.listdir(p)):
        if not f.endswith(".json"):
            continue
        j = json.load(open(os.path.join(p, f)))
        de, an = j["description"], j["annotations"]
        w, h = de.get("width") or 0, de.get("height") or 0
        if not w or not h:  # 43 disease labels carry 0x0 dims -> read real image
            zero_dim += 1
            real = idx.get(f[:-5].lower())
            if real:
                with Image.open(os.path.join(imdir, real)) as im:
                    w, h = im.size
        pts = an.get("points") or []
        b = pts[0] if pts else None
        rec = dict(
            split=split, klass=klass, disease_code=an.get("disease"),
            risk=an.get("risk"), crop=an.get("crop"), area=an.get("area"),
            grow=an.get("grow"), region=an.get("region"), date=de.get("date"),
            img_w=w, img_h=h, aspect=(w / h if h else np.nan),
            n_boxes=len(pts), ext=os.path.splitext(de["image"])[1].lower(),
        )
        if b and w and h:
            bw, bh = b["xbr"] - b["xtl"], b["ybr"] - b["ytl"]
            rec.update(
                box_w=bw, box_h=bh, box_area=bw * bh,
                box_rel_area=(bw * bh) / (w * h),
                box_aspect=(bw / bh if bh else np.nan),
                cx_norm=(b["xtl"] + bw / 2) / w, cy_norm=(b["ytl"] + bh / 2) / h,
            )
        rows.append(rec)

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "report", "metadata.csv"), index=False)

dis = df[df.klass == "disease"]
nrm = df[df.klass == "normal"]


def vc(s):
    """Series의 값별 개수를 정렬된 dict로(JSON 직렬화용)."""
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


stats = {
    "total_samples": int(len(df)),
    "by_split_class": {f"{s}/{k}": int(n) for (s, k), n in df.groupby(["split", "klass"]).size().items()},
    "class_imbalance_normal_to_disease": {
        split: round(len(df[(df.split == split) & (df.klass == "normal")]) /
                     max(1, len(df[(df.split == split) & (df.klass == "disease")])), 1)
        for split in ["train", "valid"]
    },
    "disease_code_counts": vc(dis.disease_code),
    "disease_code_by_split": {f"{s}/{int(c)}": int(n)
                              for (s, c), n in dis.groupby(["split", "disease_code"]).size().items()},
    "risk_distribution_disease": vc(dis.risk),
    "risk_by_disease_code": {f"d{int(c)}/risk{int(r)}": int(n)
                             for (c, r), n in dis.groupby(["disease_code", "risk"]).size().items()},
    "image_format_counts": vc(df.ext),
    "labels_with_zero_dims_in_json": int(zero_dim),
    "img_dims_median": {k: [int(g.img_w.median()), int(g.img_h.median())]
                        for k, g in df.groupby("klass")},
    "box_rel_area_by_class": {
        k: {"median": round(float(g.box_rel_area.median()), 4),
            "mean": round(float(g.box_rel_area.mean()), 4),
            "min": round(float(g.box_rel_area.min()), 4),
            "max": round(float(g.box_rel_area.max()), 4)}
        for k, g in df.dropna(subset=["box_rel_area"]).groupby("klass")
    },
    "boxes_per_image_max": int(df.n_boxes.max()),
    "multi_box_images": int((df.n_boxes > 1).sum()),
    "date_range": [str(df.date.min().date()), str(df.date.max().date())],
    "grow_stage_disease": vc(dis.grow),
    "region_disease": {str(k): int(v) for k, v in dis.region.value_counts(dropna=False).items()},
}
json.dump(stats, open(os.path.join(ROOT, "report", "stats.json"), "w"),
          indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- figures
C = {"normal": "#4C72B0", "disease": "#C44E52", 3: "#DD8452", 4: "#55A868"}


def save(fig, name):
    """matplotlib figure를 tight_layout으로 PNG 저장 후 close."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=120)
    plt.close(fig)


# 01 class counts
fig, ax = plt.subplots(figsize=(7, 4.5))
splits = ["train", "valid"]
x = np.arange(len(splits)); wbar = 0.38
for i, kl in enumerate(["normal", "disease"]):
    vals = [int(len(df[(df.split == s) & (df.klass == kl)])) for s in splits]
    bars = ax.bar(x + (i - 0.5) * wbar, vals, wbar, label=kl, color=C[kl])
    ax.bar_label(bars, padding=2, fontsize=9)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("image count (log scale)"); ax.set_title("Class counts per split (normal vs disease)")
ax.legend(); save(fig, "01_class_counts.png")

# 02 disease type dist
fig, ax = plt.subplots(figsize=(7, 4.5))
codes = sorted(dis.disease_code.unique()); x = np.arange(len(codes)); wbar = 0.38
for i, s in enumerate(splits):
    vals = [int(len(dis[(dis.split == s) & (dis.disease_code == c)])) for c in codes]
    bars = ax.bar(x + (i - 0.5) * wbar, vals, wbar, label=s)
    ax.bar_label(bars, padding=2, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"disease_{int(c)}" for c in codes])
ax.set_ylabel("count"); ax.set_title("Disease-type distribution by split"); ax.legend()
save(fig, "02_disease_type_dist.png")

# 03 risk dist per disease code (stacked)
fig, ax = plt.subplots(figsize=(7, 4.5))
risks = sorted(dis.risk.unique()); bottom = np.zeros(len(codes))
for r in risks:
    vals = np.array([int(len(dis[(dis.disease_code == c) & (dis.risk == r)])) for c in codes])
    ax.bar([f"disease_{int(c)}" for c in codes], vals, bottom=bottom, label=f"risk {int(r)}")
    bottom += vals
ax.set_ylabel("count"); ax.set_title("Severity (risk) distribution per disease type"); ax.legend()
save(fig, "03_risk_dist.png")

# 04 image dims + formats
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
nsamp = nrm.sample(min(3000, len(nrm)), random_state=0)
ax.scatter(nsamp.img_w, nsamp.img_h, s=8, alpha=0.3, color=C["normal"], label="normal")
ax.scatter(dis.img_w, dis.img_h, s=8, alpha=0.5, color=C["disease"], label="disease")
ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
ax.set_title("Image dimensions by class"); ax.legend()
ax = axes[1]
fmt = df.ext.value_counts()
b = ax.bar(fmt.index, fmt.values, color="#8172B3"); ax.bar_label(b, fontsize=9)
ax.set_title("File-format counts"); ax.set_ylabel("count")
save(fig, "04_image_dims.png")

# 05 bbox rel area by disease code
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(0, 1, 31)
for c in codes:
    d = dis[(dis.disease_code == c)].box_rel_area.dropna()
    ax.hist(d, bins=bins, alpha=0.6, label=f"disease_{int(c)} (n={len(d)})", color=C[int(c)])
ax.set_xlabel("bbox area / image area"); ax.set_ylabel("count")
ax.set_title("Bounding-box relative area (disease)"); ax.legend()
save(fig, "05_bbox_rel_area.png")

# 06 bbox aspect
fig, ax = plt.subplots(figsize=(7, 4.5))
asp = dis.box_aspect.dropna()
ax.hist(asp, bins=40, color="#937860")
ax.axvline(1.0, color="k", ls="--", lw=1, label="square (1.0)")
ax.set_xlabel("bbox width / height"); ax.set_ylabel("count")
ax.set_title(f"Bounding-box aspect ratio (disease, median={asp.median():.2f})"); ax.legend()
save(fig, "06_bbox_aspect.png")

# 07 bbox center heatmap
fig, ax = plt.subplots(figsize=(5.5, 5))
hb = ax.hexbin(dis.cx_norm.dropna(), dis.cy_norm.dropna(), gridsize=20,
               cmap="magma", extent=[0, 1, 0, 1])
ax.set_xlim(0, 1); ax.set_ylim(1, 0)  # image coords: y down
ax.set_xlabel("normalized center x"); ax.set_ylabel("normalized center y")
ax.set_title("Lesion bbox center positions"); fig.colorbar(hb, ax=ax, label="count")
save(fig, "07_bbox_center_heatmap.png")

# 08 date timeline by class
fig, ax = plt.subplots(figsize=(8, 4.5))
mt = df.dropna(subset=["date"]).copy()
mt["month"] = mt.date.dt.to_period("M").dt.to_timestamp()
for kl in ["normal", "disease"]:
    s = mt[mt.klass == kl].groupby("month").size()
    ax.plot(s.index, s.values, marker="o", label=kl, color=C[kl])
ax.set_ylabel("captures"); ax.set_title("Captures over time (by month)")
ax.legend(); fig.autofmt_xdate()
save(fig, "08_date_timeline.png")


# montage helpers
def montage(items, name, title, rows_n=3, cols_n=4, draw_box=False):
    """이미지(선택적 bbox)들을 그리드로 모아 한 장의 샘플 몽타주로 저장."""
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 2.6, rows_n * 2.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (imgpath, boxes) in zip(axes.ravel(), items):
        try:
            im = Image.open(imgpath).convert("RGB")
        except Exception as e:
            print("WARN open", imgpath, e); continue
        if draw_box:
            dr = ImageDraw.Draw(im)
            lw = max(3, round(min(im.size) * 0.008))
            for bx in boxes:
                dr.rectangle([bx["xtl"], bx["ytl"], bx["xbr"], bx["ybr"]],
                             outline=(255, 0, 0), width=lw)
        ax.imshow(im)
    fig.suptitle(title, fontsize=13)
    save(fig, name)


def collect(split, klass, n, code=None):
    """split/클래스(코드)별 (이미지경로, 박스) 표본을 n개까지 결정적으로 수집."""
    p = os.path.join(DATA, LABEL[(split, klass)])
    idx, imdir = img_index(split, klass)
    out = []
    for f in sorted(os.listdir(p)):
        if not f.endswith(".json"):
            continue
        an = json.load(open(os.path.join(p, f)))["annotations"]
        if code is not None and an.get("disease") != code:
            continue
        real = idx.get(f[:-5].lower())
        if real:
            out.append((os.path.join(imdir, real), an.get("points") or []))
        if len(out) >= n:
            break
    return out


montage(collect("valid", "normal", 12), "09_samples_normal.png",
        "Normal radish samples (valid)")
montage(collect("train", "disease", 12, code=3), "10_samples_disease3.png",
        "disease_3 samples with bbox (train)", draw_box=True)
montage(collect("train", "disease", 12, code=4), "11_samples_disease4.png",
        "disease_4 samples with bbox (train)", draw_box=True)

figs = sorted(os.listdir(FIG))
print(f"rows: {len(df)} | zero-dim labels fixed: {zero_dim} | figures: {len(figs)}")
print("\n".join("  " + f for f in figs))
