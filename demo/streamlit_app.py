"""Streamlit 데모 — 무(radish) 질병 분류·검출 다중 파이프라인 비교 + VQA.

FastAPI 데모(`demo/app.py`)와 동일한 추론 로직(`src.inference`, `src.vqa`)을 재사용한다.
- 이미지를 업로드하거나 valid 셋에서 골라, 선택한 여러 파이프라인(백본·세팅)으로
  동시에 분류 + 검출 결과를 비교한다(검출 박스는 이미지에 오버레이).
- VQA: 마이크 녹음(st.audio_input)·오디오 파일 업로드·텍스트 중 하나로 질문하면
  whisper-base가 STT, SmolVLM이 선택 이미지에 답한다.

실행:  ./.venv/bin/streamlit run demo/streamlit_app.py
모델 체크포인트(experiments/*/checkpoints/best.pt)가 있어야 한다(학습 먼저).
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# repo 루트를 import 경로에 추가(스크립트 위치와 무관하게 동작)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inference  # noqa: E402  (predict_image / load_registry / list_pipelines)

MANIFEST = ROOT / "_workspace" / "data" / "manifest_classification.csv"
# 검출 파이프라인 오버레이 박스 색(파이프라인별 구분)
_BOX_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]


# ---------------------------------------------------------------------------
# 캐시: 모델 레지스트리 / valid 인덱스 (재실행마다 재로딩 방지)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="모델 로딩 중... (최초 1회, 수십 초)")
def get_pipelines() -> list[dict]:
    """전 파이프라인을 1회 로드하고 공개 메타 목록을 반환."""
    inference.load_registry()
    return inference.list_pipelines()


@st.cache_data(show_spinner=False)
def get_valid_index() -> list[dict]:
    """manifest의 valid 행을 정렬해 안정적 id와 메타(klass·disease_code·risk·GT box) 구성."""
    if not MANIFEST.exists():
        return []
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "valid"]
    rows.sort(key=lambda r: r["image_path"])
    out = []
    for i, r in enumerate(rows):
        gt_box = None
        if r["klass"] != "normal":  # 질병이면 GT bbox(실제 크기 좌표) 보유
            try:
                gt_box = [float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])]
            except (ValueError, KeyError):
                gt_box = None
        out.append({
            "id": i,
            "image_path": str(ROOT / r["image_path"]),
            "klass": r["klass"],
            "disease_code": r.get("disease_code"),
            "risk": r.get("risk"),
            "gt_box": gt_box,
        })
    return out


def load_image(path_or_bytes) -> Image.Image:
    """경로/바이트에서 EXIF 보정된 RGB 이미지를 연다(학습 전처리와 동일)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        im = Image.open(path_or_bytes)
    return ImageOps.exif_transpose(im).convert("RGB")


def draw_detection(pil: Image.Image, detections: list[dict], gt_box=None) -> Image.Image:
    """검출 박스(파이프라인별 색)와 GT 박스(흰 점선 느낌)를 이미지에 그려 반환."""
    im = pil.copy()
    dr = ImageDraw.Draw(im)
    lw = max(2, round(min(im.size) * 0.004))
    if gt_box is not None:  # 정답 박스(노랑)
        dr.rectangle(gt_box, outline="#ffe119", width=lw + 1)
        dr.text((gt_box[0] + 3, max(0, gt_box[1] - 14)), "GT", fill="#ffe119")
    for j, d in enumerate(detections):
        color = _BOX_COLORS[j % len(_BOX_COLORS)]
        box = d["box_xyxy"]
        dr.rectangle(box, outline=color, width=lw)
        tag = f'{d["arch"]} {d["objectness"]:.2f}'
        dr.text((box[0] + 3, box[1] + 3), tag, fill=color)
    return im


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="무 질병 진단 데모", layout="wide")
st.title("🥬 무(Radish) 질병 진단 — 다중 파이프라인 비교 + VQA")
st.caption("DINOv3 frozen Ours · from-scratch baseline 백본들을 한 이미지에서 동시 비교. "
           "검출 박스 오버레이 + 음성/텍스트 VQA. (불균형 데이터: 우열은 PR-AUC/F1 기준 — report 참조)")

pipelines = get_pipelines()
valid_index = get_valid_index()
cls_pipes = [p for p in pipelines if p["task"] == "classification"]
det_pipes = [p for p in pipelines if p["task"] == "detection"]

# ---- 사이드바: 이미지 선택 + 파이프라인 선택 ----
with st.sidebar:
    st.header("① 이미지")
    source = st.radio("입력 방식", ["valid 셋에서 선택", "이미지 업로드"], index=0)
    pil = None
    gt_box = None
    gt_label = None

    if source == "이미지 업로드":
        up = st.file_uploader("이미지 파일", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            pil = load_image(up.getvalue())
    else:
        if not valid_index:
            st.warning("manifest가 없습니다. `python -m src.data.build_manifest` 먼저 실행하세요.")
        else:
            klasses = ["(전체)"] + sorted({v["klass"] for v in valid_index})
            kf = st.selectbox("klass 필터", klasses, index=0)
            pool = [v for v in valid_index if kf == "(전체)" or v["klass"] == kf]
            labels = [f'#{v["id"]} · {v["klass"]}' for v in pool]
            sel = st.selectbox(f"valid 이미지 ({len(pool)}개)", range(len(pool)),
                               format_func=lambda i: labels[i]) if pool else None
            if sel is not None:
                v = pool[sel]
                pil = load_image(v["image_path"])
                gt_box = v["gt_box"]
                gt_label = v["klass"]

    st.header("② 파이프라인")
    cls_names = [p["id"] for p in cls_pipes]
    det_names = [p["id"] for p in det_pipes]
    # 기본 선택: Ours(dinov3) 분류 + detection을 우선 노출
    default_cls = [n for n in cls_names if "dinov3" in n] or cls_names
    sel_cls = st.multiselect("분류 파이프라인", cls_names, default=default_cls)
    sel_det = st.multiselect("검출 파이프라인", det_names, default=det_names)
    run_btn = st.button("▶ 추론 실행", type="primary", use_container_width=True,
                        disabled=pil is None)

# ---- 메인: 결과 ----
if pil is None:
    st.info("좌측에서 이미지를 업로드하거나 valid 셋에서 고르세요.")
    st.stop()

col_img, col_res = st.columns([1, 1])

if run_btn:
    chosen = sel_cls + sel_det
    if not chosen:
        st.warning("파이프라인을 1개 이상 선택하세요.")
    else:
        with st.spinner("추론 중..."):
            res = inference.predict_image(pil, chosen)
        st.session_state["res"] = res  # VQA 등에서 재사용

res = st.session_state.get("res")

with col_img:
    st.subheader("입력 이미지" + (f" · GT: {gt_label}" if gt_label else ""))
    if res and res.get("detection"):
        st.image(draw_detection(pil, res["detection"], gt_box),
                 caption="검출 박스 오버레이 (색=파이프라인, 노랑=GT)", use_container_width=True)
    else:
        st.image(pil, use_container_width=True)

with col_res:
    if not res:
        st.info("‘추론 실행’을 누르세요.")
    else:
        # 분류 결과: 세팅별 그룹 표
        st.subheader("분류 결과")
        if gt_label:
            st.markdown(f"**정답(GT): `{gt_label}`**")
        if res["classification"]:
            import pandas as pd
            rows = []
            for c in res["classification"]:
                probs = ", ".join(f"{n}={p:.3f}" for n, p in zip(c["class_names"], c["probs"]))
                rows.append({"arch": c["arch"], "setting": c["setting"],
                             "예측": c["pred_class"], "확률": probs})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("선택된 분류 파이프라인 없음")

        # 검출 결과
        st.subheader("검출 결과 (이미지 단위 질병 유무 + 박스)")
        if res["detection"]:
            import pandas as pd
            drows = [{"arch": d["arch"], "objectness": round(d["objectness"], 3),
                      "질병?": "🦠 질병" if d["is_disease"] else "✅ 정상",
                      "box(xyxy)": [round(v) for v in d["box_xyxy"]]} for d in res["detection"]]
            st.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)
        else:
            st.caption("선택된 검출 파이프라인 없음")

# ---- VQA 패널 ----
st.divider()
st.subheader("💬 VQA — 음성/텍스트 질의응답 (whisper STT + SmolVLM)")
st.caption("현재 선택된 이미지에 대해 질문합니다. 마이크 녹음 / 오디오 파일 / 텍스트 중 하나.")

vq1, vq2 = st.columns(2)
with vq1:
    mic = st.audio_input("🎙️ 마이크로 질문 녹음")
    audio_file = st.file_uploader("또는 오디오 파일 (wav·mp3·ogg·flac)",
                                  type=["wav", "mp3", "ogg", "flac"])
with vq2:
    text_q = st.text_input("또는 텍스트 질문", placeholder="Is this radish leaf diseased?")
    ask = st.button("❓ Ask VQA", use_container_width=True)

if ask:
    wav_bytes = None
    if mic is not None:
        wav_bytes = mic.getvalue()          # st.audio_input → WAV
    elif audio_file is not None:
        wav_bytes = audio_file.getvalue()   # 업로드 오디오(soundfile 디코딩)
    if wav_bytes is None and not text_q.strip():
        st.warning("마이크 녹음, 오디오 파일, 텍스트 중 하나로 질문하세요.")
    else:
        with st.spinner("VQA 추론 중... (모델 최초 로드 시 수십 초)"):
            from src import vqa as vqa_mod
            out = vqa_mod.answer(pil, question_text=text_q.strip() or None, wav_bytes=wav_bytes)
        if out.get("transcript"):
            st.markdown(f"**인식된 질문(STT)**: _{out['transcript']}_")
        st.markdown(f"**질문**: {out.get('question') or '(없음)'}")
        st.success(f"**답변**: {out.get('answer')}")

st.divider()
st.caption("⚠️ SmolVLM·whisper는 범용 모델이라 무 질병 특화가 아닙니다(보조 설명용). "
           "질병 판정은 분류/검출 파이프라인(위)이 담당합니다. baseline=from-scratch, Ours=DINOv3 pretrained.")
