"""Inference registry + predict for the radish baseline demo.

Loads all 12 trained pipelines (9 classification + 3 detection) ONCE at startup
into a registry (eval mode, cuda if available, fp32), then exposes
`predict_image(pil, pipeline_ids)` returning classification + detection results
in original-image coordinates.

Pipeline discovery: scans `experiments/<name>/` for `config.snapshot`
(JSON: spec.model / spec.data / spec.eval) and `checkpoints/best.pt`
(dict with key "model" = state_dict), plus `metrics.json` (final metrics).

Used by demo/app.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from src.models import build_classifier, build_detector
from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    classification_eval_transform,
)
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class names per classification setting (from data.setting)
SETTING_CLASS_NAMES = {
    "normal_vs_d3": ["normal", "disease_3"],
    "normal_vs_d4": ["normal", "disease_4"],
    "normal_d3_d4": ["normal", "disease_3", "disease_4"],
}


@dataclass
class Pipeline:
    """데모용 로드된 한 파이프라인(모델+메타). public()으로 JSON 직렬화."""
    id: str
    arch: str
    task: str  # "classification" | "detection"
    img_size: int
    num_classes: Optional[int] = None
    setting: Optional[str] = None
    class_names: Optional[list] = None
    primary_metric: dict = field(default_factory=dict)  # {name, value}
    model: torch.nn.Module = None  # loaded lazily into registry

    def public(self) -> dict:
        """Pipeline의 외부 노출용 메타(id/arch/task/지표 등) dict."""
        return {
            "id": self.id,
            "arch": self.arch,
            "task": self.task,
            "setting": self.setting,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "img_size": self.img_size,
            "primary_metric": self.primary_metric,
        }


_REGISTRY: dict[str, Pipeline] = {}


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------
def _load_one(name: str) -> Pipeline:
    """experiments/<name>의 config.snapshot+best.pt로 모델을 만들어 Pipeline 구성."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    with open(os.path.join(exp_dir, "config.snapshot")) as f:
        cfg = json.load(f)
    spec = cfg["spec"]
    task = spec["task"]
    model_cfg = spec["model"]
    data_cfg = spec.get("data", {})
    arch = model_cfg["arch"]
    img_size = int(model_cfg["img_size"])

    # primary metric (name from spec.eval.primary, value from metrics.json final)
    primary_name = spec.get("eval", {}).get("primary")
    primary_val = None
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        final = metrics.get("final", {})
        if primary_name and primary_name in final:
            primary_val = final[primary_name]
    primary_metric = {"name": primary_name, "value": primary_val}

    ckpt = os.path.join(exp_dir, "checkpoints", "best.pt")
    sd = torch.load(ckpt, map_location=DEVICE)["model"]

    if task == "classification":
        setting = data_cfg.get("setting")
        class_names = SETTING_CLASS_NAMES.get(setting)
        num_classes = int(model_cfg.get("num_classes") or len(class_names))
        model = build_classifier(arch, num_classes=num_classes, img_size=img_size)
        model.load_state_dict(sd)
        model.eval().to(DEVICE)
        return Pipeline(
            id=name, arch=arch, task=task, img_size=img_size,
            num_classes=num_classes, setting=setting, class_names=class_names,
            primary_metric=primary_metric, model=model,
        )
    else:  # detection
        with_obj = bool(model_cfg.get("with_objectness", True))
        model = build_detector(arch, img_size=img_size, with_objectness=with_obj)
        model.load_state_dict(sd)
        model.eval().to(DEVICE)
        return Pipeline(
            id=name, arch=arch, task=task, img_size=img_size,
            primary_metric=primary_metric, model=model,
        )


# Pure-ablation runs are kept on disk for the report but excluded from the demo
# pipeline list to avoid clutter (they are gamma/aug variants of dinov3_base_focal).
_EXCLUDE_TOKENS = ("augonly", "focalonly", "focalg1", "focalg3")


def load_registry() -> dict[str, Pipeline]:
    """Load all experiment pipelines once. Idempotent."""
    if _REGISTRY:
        return _REGISTRY
    names = sorted(
        d for d in os.listdir(EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))
        and os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "config.snapshot"))
        and os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "checkpoints", "best.pt"))
        and not any(tok in d for tok in _EXCLUDE_TOKENS)
    )
    for name in names:
        _REGISTRY[name] = _load_one(name)
    return _REGISTRY


def list_pipelines() -> list[dict]:
    """로드된 전 파이프라인의 공개 메타 목록 반환(/api/pipelines)."""
    reg = load_registry()
    return [reg[k].public() for k in sorted(reg)]


def _resolve_ids(pipeline_ids) -> list[str]:
    """요청의 pipelines 인자('all' 또는 id 리스트)를 실제 파이프라인 id로 해석."""
    reg = load_registry()
    if pipeline_ids in (None, "all", ["all"]):
        return sorted(reg)
    if isinstance(pipeline_ids, str):
        pipeline_ids = [p.strip() for p in pipeline_ids.split(",") if p.strip()]
    return [p for p in pipeline_ids if p in reg]


# ---------------------------------------------------------------------------
# Detection preprocessing: square resize (img_size) + ImageNet normalize.
# pred box [0,1] is inverse-transformed back to ORIGINAL image coords.
# ---------------------------------------------------------------------------
def _detection_input(pil: Image.Image, img_size: int) -> torch.Tensor:
    """PIL을 detection 입력(img_size 정사각 resize+ImageNet 정규화) 텐서로 변환."""
    img = pil.resize((img_size, img_size), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.unsqueeze(0)


@torch.no_grad()
def predict_image(pil: Image.Image, pipeline_ids="all") -> dict:
    """Run selected pipelines on a single PIL image.

    Returns {"classification": [...], "detection": [...]}.
    Detection boxes are returned in ORIGINAL image coordinates (xyxy).
    """
    reg = load_registry()
    ids = _resolve_ids(pipeline_ids)
    W, H = pil.size

    classification = []
    detection = []

    for pid in ids:
        p = reg[pid]
        if p.task == "classification":
            x = classification_eval_transform(p.img_size)(pil).unsqueeze(0).to(DEVICE)
            logits = p.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred_index = int(max(range(len(probs)), key=lambda i: probs[i]))
            classification.append({
                "pipeline_id": pid,
                "arch": p.arch,
                "setting": p.setting,
                "class_names": p.class_names,
                "probs": [round(v, 6) for v in probs],
                "pred_class": p.class_names[pred_index],
                "pred_index": pred_index,
            })
        else:  # detection
            x = _detection_input(pil, p.img_size).to(DEVICE)
            out = p.model(x)
            if isinstance(out, tuple):
                boxes, obj_logit = out
                score = float(torch.sigmoid(obj_logit)[0].cpu())
            else:
                boxes = out
                score = 1.0
            b = boxes[0].cpu().tolist()  # [0,1] xyxy in square frame
            # inverse-transform [0,1] -> original image coords
            x0 = b[0] * W
            y0 = b[1] * H
            x1 = b[2] * W
            y1 = b[3] * H
            detection.append({
                "pipeline_id": pid,
                "arch": p.arch,
                "box_xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "objectness": round(score, 6),
                "is_disease": bool(score > 0.5),
            })

    return {"classification": classification, "detection": detection}
