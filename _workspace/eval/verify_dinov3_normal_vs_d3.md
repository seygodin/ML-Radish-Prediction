# verify_dinov3_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: **dinov3 (DINOv3 ViT-S/16 frozen + 2-layer head)**, Ours
- img_size = **256** (from config.snapshot), num_classes = 2, best epoch = 25
- params: total = **21.686M** (frozen backbone), trainable = **99.07k** (head only)

## 1) Boundary cross-validation (predictions vs manifest, ORIGINAL dist)

- predictions valid N = **1379**, manifest valid N = **1379** -> MATCH
- predictions label dist = {0: 1303, 1: 76}
- manifest valid label dist = {0: 1303, 1: 76}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_3']

## 2) Independent metric recomputation vs reported (ORIGINAL dist)

Recomputed with sklearn from predictions/valid.npz; cross-checked vs metrics.json final.

| metric | reported | recomputed (sklearn) | match |
|---|---|---|---|
| pr_auc | 0.9964 | 0.9964 | ok |
| f1_macro | 0.9237 | 0.9237 | ok |
| recall_macro | 0.9842 | 0.9842 | ok |
| precision_macro | 0.8784 | 0.8784 | ok |
| auroc | 0.9997 | 0.9997 | ok |
| accuracy | 0.9819 | 0.9819 | ok |

- ORIGINAL 7-metric: acc=0.9819, train_loss=0.0024, val_loss=0.0406, recall_macro=0.9842, precision_macro=0.8784, f1_macro=0.9237, AUROC=0.9997; PR-AUC=0.9964
- confusion (orig, recomputed): [[1279, 24], [1, 75]]

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=256)

- balanced valid counts = {'normal': 76, 'disease_3': 76} (N=152, seed=42)
- BALANCED 7-metric: acc=0.9803, train_loss=0.0024 (training-time), val_loss=0.0426 (recomputed on balanced), recall_macro=0.9803, precision_macro=0.9803, f1_macro=0.9803, AUROC=0.9995; PR-AUC=0.9995
- confusion (balanced): [[74, 2], [1, 75]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_3] = 75/76 = 0.987 (95% CI 0.929-0.998)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours 0.9964 vs baseline best 0.9647 (resnet50) -> abs +0.0317, rel +3.3%
- F1-macro: Ours 0.9237 vs baseline best 0.8604 (densenet121) -> abs +0.0634, rel +7.4%

### BALANCED
- PR-AUC: Ours 0.9995 vs baseline best 0.9997 (resnet50) -> abs -0.0002, rel -0.0%
- F1-macro: Ours 0.9803 vs baseline best 0.9803 (densenet121) -> abs +0.0000, rel +0.0%
