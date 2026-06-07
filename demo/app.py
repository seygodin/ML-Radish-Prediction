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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

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
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    """GET / — 데모 프론트엔드(index.html) 서빙."""
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
