"""XAI 실험 — DINOv3(Ours) 분류기가 이미지를 제대로 보고 있는지 시각화.

Ours+ = `dinov3_base_focal_normal_d3_d4` (DINOv3-B/16 frozen + 2-layer head, 3-class).
예시 이미지 3장(normal / disease_3 / disease_4)에 대해 세 가지 설명기법을 적용:
  - Grad-CAM       : 마지막 ViT 블록(blocks[-1].norm1) 토큰 saliency를 32x32 패치
                     그리드로 reshape → "모델이 어디를 보는가".
  - LIME           : 슈퍼픽셀 단위 perturbation으로 예측 클래스에 대한 국소 선형 기여.
  - SHAP (Partition): 블러 마스킹 기반 Shapley 값(이미지 영역 기여), 예측 클래스.

frozen 백본은 forward_features가 no_grad로 감싸져 있어 Grad-CAM용 gradient가
흐르지 못한다. 따라서 backbone.forward_features→forward_head→classifier를 직접
호출하는 grad-friendly 래퍼(GradModel)로 backbone activation에 gradient를 흘린다.
(가중치는 그대로, 추론 전용 — 학습 없음.)

산출물:
  report/figures/exp_xai_dinov3.png       (행=예시, 열=원본/Grad-CAM/LIME/SHAP)
  _workspace/eval/xai_dinov3.json         (예측·기법별 상태 메타)
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inference  # noqa: E402
from src.data.transforms import classification_eval_transform  # noqa: E402

MANIFEST = ROOT / "_workspace" / "data" / "manifest_classification.csv"
FIG_OUT = ROOT / "report" / "figures" / "exp_xai_dinov3.png"
JSON_OUT = ROOT / "_workspace" / "eval" / "xai_dinov3.json"
CLS_PIPE = "dinov3_base_focal_normal_d3_d4"  # Ours+ focal+aug, 3-class
DEVICE = inference.DEVICE
CLASSES = ["normal", "disease_3", "disease_4"]


# --------------------------------------------------------------------------
# grad-friendly 래퍼: frozen no_grad 우회, backbone activation에 grad 허용
# --------------------------------------------------------------------------
class GradModel(nn.Module):
    """forward_features(no_grad) 우회 — backbone 토큰에 gradient가 흐르게 한다."""

    def __init__(self, dinov3_model):
        super().__init__()
        self.bb = dinov3_model.backbone
        self.head = dinov3_model.classifier
        self.n_prefix = getattr(self.bb, "num_prefix_tokens", 1)

    def forward(self, x):
        tokens = self.bb.forward_features(x)                 # (B, N, C) grad 허용
        pooled = self.bb.forward_head(tokens, pre_logits=True)  # (B, C)
        return self.head(pooled)


def reshape_transform(tokens):
    """ViT 토큰 (B, N, C) → (B, C, H, W). 앞쪽 prefix(CLS+register) 토큰 제거."""
    n_prefix = reshape_transform.n_prefix
    patch = tokens[:, n_prefix:, :]
    n = patch.shape[1]
    s = int(round(n ** 0.5))
    out = patch.reshape(patch.size(0), s, s, patch.size(2))
    return out.permute(0, 3, 1, 2)


def pick_examples():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "valid"]
    out = []
    for k in CLASSES:
        sub = sorted([r for r in rows if r["klass"] == k], key=lambda r: r["image_path"])
        r = sub[len(sub) // 2]
        out.append({"klass": k, "path": str(ROOT / r["image_path"])})
    return out


def load_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def main():
    reg = inference.load_registry()
    model = reg[CLS_PIPE].model.eval().to(DEVICE)
    img_size = reg[CLS_PIPE].img_size
    tfm = classification_eval_transform(img_size)
    examples = pick_examples()

    # 예측 클래스 계산(공통)
    def predict_pil(pil):
        x = tfm(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
        return probs

    # LIME/SHAP용 batch predict: numpy [N,H,W,3] (0-255 or 0-1) -> probs [N,3]
    def predict_np(arr):
        arr = np.asarray(arr)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
        xs = []
        for im in arr:
            xs.append(tfm(Image.fromarray(im)).to(DEVICE))
        x = torch.stack(xs)
        with torch.no_grad():
            out = []
            for i in range(0, x.size(0), 32):
                logits = model(x[i:i + 32])
                if isinstance(logits, tuple):
                    logits = logits[0]
                out.append(torch.softmax(logits, 1).cpu().numpy())
        return np.concatenate(out, 0)

    # ---- Grad-CAM 준비 ----
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    grad_model = GradModel(model).to(DEVICE).eval()
    reshape_transform.n_prefix = grad_model.n_prefix
    target_layer = grad_model.bb.blocks[-1].norm1
    cam = GradCAM(model=grad_model, target_layers=[target_layer],
                  reshape_transform=reshape_transform)

    # ---- LIME / SHAP ----
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import shap

    lime_expl = lime_image.LimeImageExplainer()

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.4 * n))
    records = []

    for row, ex in enumerate(examples):
        pil = load_image(ex["path"])
        probs = predict_pil(pil)
        pred = int(probs.argmax())
        rec = {"klass": ex["klass"], "pred": CLASSES[pred],
               "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
               "methods": {}}

        # 표시/perturbation용 정규화 이미지 (모델 입력 프레임과 동일 정사각 리사이즈)
        disp = pil.resize((img_size, img_size))
        disp_np = np.asarray(disp).astype(np.float32) / 255.0  # [H,W,3] 0-1

        # (0) 원본
        ax = axes[row, 0]
        ax.imshow(disp_np); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f'{ex["klass"]}', fontsize=12, weight="bold")
        ax.set_title(f'input — pred: {CLASSES[pred]} ({probs[pred]:.2f})', fontsize=10)

        # (1) Grad-CAM
        try:
            # frozen 백본은 input에 requires_grad를 켜야 backbone activation에 gradient가 흐른다
            x = tfm(pil).unsqueeze(0).to(DEVICE).requires_grad_(True)
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
            cam_img = show_cam_on_image(disp_np, grayscale, use_rgb=True)
            axes[row, 1].imshow(cam_img)
            rec["methods"]["gradcam"] = "ok"
        except Exception as e:
            axes[row, 1].text(0.5, 0.5, f"Grad-CAM\nN/A\n{type(e).__name__}", ha="center", va="center")
            rec["methods"]["gradcam"] = f"fail:{type(e).__name__}:{e}"
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        axes[row, 1].set_title("Grad-CAM (last ViT block)", fontsize=10)

        # (2) LIME (224 해상도에서 perturbation)
        try:
            lime_in = (np.asarray(pil.resize((224, 224)))).astype(np.double)
            exp = lime_expl.explain_instance(
                lime_in, predict_np, labels=(pred,), hide_color=0,
                num_samples=1000, top_labels=None, random_seed=42)
            temp, mask = exp.get_image_and_mask(
                pred, positive_only=True, num_features=8, hide_rest=False)
            axes[row, 2].imshow(mark_boundaries(temp / 255.0, mask))
            rec["methods"]["lime"] = "ok"
        except Exception as e:
            axes[row, 2].text(0.5, 0.5, f"LIME\nN/A\n{type(e).__name__}", ha="center", va="center")
            rec["methods"]["lime"] = f"fail:{type(e).__name__}:{e}"
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        axes[row, 2].set_title("LIME (superpixels, +contrib)", fontsize=10)

        # (3) SHAP (Partition/blur masker, 160 해상도)
        try:
            sz = 160
            shap_in = np.asarray(pil.resize((sz, sz))).astype(np.float32) / 255.0
            masker = shap.maskers.Image("blur(32,32)", shap_in.shape)
            explainer = shap.Explainer(predict_np, masker, output_names=CLASSES)
            sv = explainer(shap_in[None], max_evals=400, batch_size=64,
                           outputs=shap.Explanation.argsort.flip[:1])
            vals = sv.values[0]              # (H,W,3,1)
            heat = vals[..., 0].sum(-1) if vals.ndim == 4 else vals.sum(-1)
            lim = np.abs(heat).max() + 1e-8
            axes[row, 3].imshow(shap_in)
            axes[row, 3].imshow(heat, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.6)
            rec["methods"]["shap"] = "ok"
        except Exception as e:
            axes[row, 3].text(0.5, 0.5, f"SHAP\nN/A\n{type(e).__name__}", ha="center", va="center")
            rec["methods"]["shap"] = f"fail:{type(e).__name__}:{e}"
        axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])
        axes[row, 3].set_title("SHAP (partition, pred class)", fontsize=10)

        records.append(rec)
        print(f'[{ex["klass"]}] pred={CLASSES[pred]} probs={rec["probs"]} methods={rec["methods"]}')

    fig.suptitle("XAI on Ours (DINOv3-B frozen, 3-class focal+aug): Grad-CAM / LIME / SHAP",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    JSON_OUT.write_text(json.dumps(
        {"pipeline": CLS_PIPE, "classes": CLASSES, "records": records,
         "examples": [e["path"] for e in examples]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("saved:", FIG_OUT)
    print("saved:", JSON_OUT)


if __name__ == "__main__":
    main()
