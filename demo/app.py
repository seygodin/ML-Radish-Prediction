"""FastAPI demo for the radish (무) disease baseline experiments.

Serves a web UI that runs an uploaded image (or a valid-set image) through all
12 trained pipelines (9 classification + 3 detection) at once and compares
classification + detection results.

Run:
    ./.venv/bin/python -m uvicorn demo.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                         -> frontend (demo/static/index.html)
    GET  /api/pipelines            -> list of 12 pipelines
    GET  /api/valid-images         -> paginated valid-set index (?klass=&limit=&offset=)
    GET  /api/valid-images/{id}/raw-> raw image bytes
    POST /api/predict              -> classification + detection comparison
    POST /api/vqa                  -> voice/text VQA over selected image
"""
from __future__ import annotations

import csv
import io
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from src import inference

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MANIFEST = os.path.join(REPO_ROOT, "_workspace", "data", "manifest_classification.csv")

app = FastAPI(title="Radish Disease Baseline Demo")


# ---------------------------------------------------------------------------
# Valid-image index (stable id = position after sorting valid rows by image_path)
# ---------------------------------------------------------------------------
_VALID_INDEX: list[dict] = []


def _build_valid_index() -> list[dict]:
    """manifest의 valid 행을 정렬해 안정적 id(0..N-1)와 메타 인덱스 구성."""
    rows = []
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] != "valid":
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["image_path"])
    index = []
    for i, r in enumerate(rows):
        klass = r["klass"]
        gt_box = None
        if klass != "normal":
            try:
                gt_box = [float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])]
            except (ValueError, KeyError):
                gt_box = None
        index.append({
            "id": i,
            "image_path": os.path.join(REPO_ROOT, r["image_path"]),
            "filename": os.path.basename(r["image_path"]),
            "true_klass": klass,
            "disease_code": r.get("disease_code"),
            "risk": r.get("risk"),
            "gt_box_xyxy": gt_box,
        })
    return index


@app.on_event("startup")
def _startup():
    """서버 시작 시 추론 레지스트리(파이프라인)와 valid 인덱스를 1회 로드."""
    global _VALID_INDEX
    _VALID_INDEX = _build_valid_index()
    reg = inference.load_registry()
    print(f"[demo] loaded {len(reg)} pipelines, {len(_VALID_INDEX)} valid images, "
          f"device={inference.DEVICE}")


# ---------------------------------------------------------------------------
# API: pipelines
# ---------------------------------------------------------------------------
@app.get("/api/pipelines")
def api_pipelines():
    """GET /api/pipelines — 로드된 파이프라인 목록(arch/task/지표) 반환."""
    return inference.list_pipelines()


# ---------------------------------------------------------------------------
# API: valid images (filter + pagination)
# ---------------------------------------------------------------------------
@app.get("/api/valid-images")
def api_valid_images(klass: Optional[str] = None, limit: int = 50, offset: int = 0):
    """GET /api/valid-images — klass 필터·페이지네이션된 valid 이미지 목록."""
    items = _VALID_INDEX
    if klass:
        items = [it for it in items if it["true_klass"] == klass]
    total = len(items)
    page = items[offset: offset + limit]
    return {
        "total": total,
        "items": [
            {
                "id": it["id"],
                "filename": it["filename"],
                "true_klass": it["true_klass"],
                "disease_code": it["disease_code"],
                "risk": it["risk"],
            }
            for it in page
        ],
    }


@app.get("/api/valid-images/{image_id}/raw")
def api_valid_image_raw(image_id: int):
    """GET /api/valid-images/{id}/raw — 해당 valid 이미지 파일 응답."""
    if image_id < 0 or image_id >= len(_VALID_INDEX):
        raise HTTPException(status_code=404, detail="valid image id out of range")
    path = _VALID_INDEX[image_id]["image_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image file not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Shared image resolution (upload file OR valid-set id)
# ---------------------------------------------------------------------------
def _resolve_image(raw: Optional[bytes], valid_image_id: Optional[int]):
    """Return (pil_rgb, source, valid_id, ground_truth) from an upload OR a
    valid-set id. Mirrors the EXIF/orientation handling used at training time.
    """
    if raw is not None:
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot read image: {e}")
        return pil, "upload", None, None
    if valid_image_id is not None:
        if valid_image_id < 0 or valid_image_id >= len(_VALID_INDEX):
            raise HTTPException(status_code=404, detail="valid image id out of range")
        it = _VALID_INDEX[valid_image_id]
        pil = Image.open(it["image_path"]).convert("RGB")
        # match training (load_image w/ exif_transpose)
        from PIL import ImageOps
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        ground_truth = {
            "true_klass": it["true_klass"],
            "gt_box_xyxy": it["gt_box_xyxy"],
        }
        return pil, "valid", valid_image_id, ground_truth
    raise HTTPException(status_code=400, detail="provide either file or valid_image_id")


# ---------------------------------------------------------------------------
# API: predict
# ---------------------------------------------------------------------------
@app.post("/api/predict")
async def api_predict(
    file: Optional[UploadFile] = File(None),
    valid_image_id: Optional[int] = Form(None),
    pipelines: str = Form("all"),
):
    """POST /api/predict — 업로드/valid 이미지에 선택 파이프라인들로 분류+detection 추론."""
    raw = await file.read() if file is not None else None
    pil, source, valid_id, ground_truth = _resolve_image(raw, valid_image_id)
    W, H = pil.size
    pipeline_ids = "all" if pipelines in ("all", "", None) else pipelines
    result = inference.predict_image(pil, pipeline_ids)

    return JSONResponse({
        "input": {
            "source": source,
            "width": W,
            "height": H,
            "valid_id": valid_id,
            "ground_truth": ground_truth,
        },
        "classification": result["classification"],
        "detection": result["detection"],
    })


# ---------------------------------------------------------------------------
# API: vqa (voice/text question -> answer over the selected image)
# ---------------------------------------------------------------------------
@app.post("/api/vqa")
async def api_vqa(
    file: Optional[UploadFile] = File(None),
    valid_image_id: Optional[int] = Form(None),
    audio: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
):
    """POST /api/vqa — 이미지 + (오디오 STT 또는 텍스트) 질문으로 SmolVLM VQA 답변."""
    raw = await file.read() if file is not None else None
    pil, source, _valid_id, _gt = _resolve_image(raw, valid_image_id)

    audio_bytes = await audio.read() if audio is not None else None
    q_text = (question or "").strip() or None
    if not audio_bytes and not q_text:
        raise HTTPException(
            status_code=400,
            detail="provide audio or question (at least one)",
        )

    try:
        from src import vqa as vqa_mod
    except Exception as e:  # import-time failure (deps)
        raise HTTPException(status_code=500, detail=f"VQA module unavailable: {e}")

    try:
        result = vqa_mod.answer(pil, question_text=q_text, wav_bytes=audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VQA failed: {e}")

    return JSONResponse({
        "transcript": result["transcript"],
        "question": result["question"],
        "answer": result["answer"],
        "image_source": source,
    })


# ---------------------------------------------------------------------------
# API: report-pdf (classification + detection + VQA -> formatted PDF)
# ---------------------------------------------------------------------------
_DET_COLORS = ["#ff5d5d", "#ffb13d", "#1aa564", "#9d6dff", "#1f9bd1",
               "#e6194B", "#3cb44b"]
_KO_FONT = "HYSMyeongJo-Medium"  # reportlab 내장 한글 CID 폰트


def _draw_overlay(pil: "Image.Image", detection: list, gt_box) -> "Image.Image":
    """검출 박스(파이프라인별 색)와 GT 박스를 원본 좌표로 이미지에 그려 반환."""
    im = pil.copy()
    dr = ImageDraw.Draw(im)
    lw = max(2, round(min(im.size) * 0.004))
    if gt_box:
        dr.rectangle(gt_box, outline="#1aa564", width=lw + 1)
        dr.text((gt_box[0] + 4, max(0, gt_box[1] - 16)), "GT", fill="#1aa564")
    for i, d in enumerate(detection):
        col = _DET_COLORS[i % len(_DET_COLORS)]
        box = d.get("box_xyxy")
        if not box:
            continue
        dr.rectangle(box, outline=col, width=lw)
        dr.text((box[0] + 4, box[1] + 4),
                f'{d.get("arch","")} {d.get("objectness",0):.2f}', fill=col)
    return im


def _build_report_pdf(pil: "Image.Image", results: dict) -> bytes:
    """예측 결과(classification/detection/vqa)를 보기 좋은 A4 PDF로 구성해 bytes 반환."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors as rl
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLImage)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    try:
        pdfmetrics.registerFont(UnicodeCIDFont(_KO_FONT))
    except Exception:
        pass  # 이미 등록됨

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1ko", parent=styles["Title"], fontName=_KO_FONT, fontSize=18)
    h2 = ParagraphStyle("h2ko", parent=styles["Heading2"], fontName=_KO_FONT, fontSize=12,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("bodyko", parent=styles["Normal"], fontName=_KO_FONT, fontSize=9,
                          leading=13)
    muted = ParagraphStyle("mutedko", parent=body, textColor=rl.grey, fontSize=8)

    inp = results.get("input") or {}
    gt = inp.get("ground_truth") or None
    detection = results.get("detection") or []
    classification = results.get("classification") or []
    vqa = results.get("vqa") or None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="무 질병 진단 결과 리포트",
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    avail_w = doc.width
    story = []

    story.append(Paragraph("무(Radish) 질병 진단 결과 리포트", h1))
    src = inp.get("source", "?")
    meta = f'입력: {src} · 크기 {inp.get("width","?")}×{inp.get("height","?")}'
    if gt:
        meta += f' · Ground Truth 클래스: <b>{gt.get("true_klass")}</b>'
        if gt.get("gt_box_xyxy"):
            meta += f' · GT box [{", ".join(f"{v:.0f}" for v in gt["gt_box_xyxy"])}]'
    story.append(Paragraph(meta, muted))
    story.append(Spacer(1, 6))

    # --- 이미지 (검출 오버레이) ---
    gt_box = gt.get("gt_box_xyxy") if gt else None
    overlay = _draw_overlay(pil, detection, gt_box)
    # 인쇄 해상도면 충분하므로 임베드 전 다운스케일(PDF 용량 절감)
    long_side = max(overlay.size)
    if long_side > 1400:
        s = 1400 / long_side
        overlay = overlay.resize((round(overlay.width * s), round(overlay.height * s)),
                                 Image.LANCZOS)
    img_buf = io.BytesIO()
    overlay.save(img_buf, format="JPEG", quality=88)
    img_buf.seek(0)
    iw, ih = overlay.size
    disp_w = min(avail_w, 150 * mm)
    disp_h = disp_w * ih / iw
    max_h = 110 * mm
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h * iw / ih
    story.append(Paragraph("입력 이미지 + 검출 박스 (빨강·주황…=파이프라인, 초록=GT)", h2))
    story.append(RLImage(img_buf, width=disp_w, height=disp_h))
    story.append(Spacer(1, 4))

    # --- 분류 ---
    story.append(Paragraph("분류 (Classification)", h2))
    if classification:
        order = ["normal_vs_d3", "normal_vs_d4", "normal_d3_d4"]
        groups = {}
        for c in classification:
            groups.setdefault(c.get("setting", "?"), []).append(c)
        for setting in [s for s in order if s in groups] + [s for s in groups if s not in order]:
            story.append(Paragraph(f"<b>[{setting}]</b>", body))
            data = [["백본 (arch)", "예측", "클래스별 확률"]]
            for c in groups[setting]:
                probs = "  ".join(f'{n}={p*100:.1f}%'
                                  for n, p in zip(c.get("class_names", []), c.get("probs", [])))
                data.append([c.get("arch", ""), c.get("pred_class", ""), probs])
            tbl = Table(data, colWidths=[avail_w * 0.26, avail_w * 0.22, avail_w * 0.52])
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), _KO_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), rl.HexColor("#1f3b5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl.white),
                ("GRID", (0, 0), (-1, -1), 0.4, rl.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl.white, rl.HexColor("#f3f6fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("선택된 분류 파이프라인 없음.", muted))

    # --- 검출 ---
    story.append(Paragraph("검출 (Detection — 이미지 단위 질병 유무 + 박스)", h2))
    if detection:
        data = [["백본 (arch)", "objectness", "판정", "box (xyxy)"]]
        for d in detection:
            box = d.get("box_xyxy") or []
            data.append([d.get("arch", ""), f'{d.get("objectness",0):.3f}',
                         "질병" if d.get("is_disease") else "정상",
                         "[" + ", ".join(f"{v:.0f}" for v in box) + "]"])
        tbl = Table(data, colWidths=[avail_w * 0.26, avail_w * 0.18, avail_w * 0.14, avail_w * 0.42])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), _KO_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), rl.HexColor("#1f3b5c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl.white),
            ("GRID", (0, 0), (-1, -1), 0.4, rl.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl.white, rl.HexColor("#f3f6fa")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("선택된 검출 파이프라인 없음.", muted))

    # --- VQA ---
    story.append(Paragraph("VQA (질의응답 — SmolVLM + whisper)", h2))
    if vqa and (vqa.get("answer") or vqa.get("question")):
        if vqa.get("transcript"):
            story.append(Paragraph(f'<b>인식된 질문(STT)</b>: {vqa["transcript"]}', body))
        story.append(Paragraph(f'<b>질문</b>: {vqa.get("question") or "(없음)"}', body))
        story.append(Paragraph(f'<b>답변</b>: {vqa.get("answer") or "(없음)"}', body))
    else:
        story.append(Paragraph("VQA를 실행하지 않았습니다.", muted))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "baseline=from-scratch, Ours=DINOv3 pretrained(frozen). 범용 VQA(SmolVLM)는 보조 설명용이며 "
        "질병 판정은 분류/검출 파이프라인이 담당합니다. 자세한 지표는 report/PAPER.md 참조.", muted))

    doc.build(story)
    return buf.getvalue()


@app.post("/api/report-pdf")
async def api_report_pdf(
    file: Optional[UploadFile] = File(None),
    valid_image_id: Optional[int] = Form(None),
    results: str = Form(...),
):
    """POST /api/report-pdf — 프론트의 예측 결과(JSON)+이미지로 결과 리포트 PDF 생성·반환."""
    import json
    raw = await file.read() if file is not None else None
    pil, source, valid_id, ground_truth = _resolve_image(raw, valid_image_id)
    try:
        parsed = json.loads(results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid results JSON: {e}")
    # input 메타는 신뢰 가능한 서버측 값으로 보정(GT는 valid일 때만 존재)
    parsed.setdefault("input", {})
    parsed["input"].update({"source": source, "width": pil.size[0], "height": pil.size[1],
                            "valid_id": valid_id, "ground_truth": ground_truth})
    try:
        pdf = _build_report_pdf(pil, parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 생성 실패: {e}")
    fname = f"radish_report_{source}_{valid_id if valid_id is not None else 'upload'}.pdf"
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    """GET / — 데모 프론트엔드(index.html) 서빙."""
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
