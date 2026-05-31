# verify_dinov3_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: **dinov3 (DINOv3 ViT-S/16 frozen + 2-layer head)**, Ours
- img_size = **256** (from config.snapshot), num_classes = 2, best epoch = 15
- params: total = **21.686M** (frozen backbone), trainable = **99.07k** (head only)

## 1) Boundary cross-validation (predictions vs manifest, ORIGINAL dist)

- predictions valid N = **1327**, manifest valid N = **1327** -> MATCH
- predictions label dist = {0: 1303, 1: 24}
- manifest valid label dist = {0: 1303, 1: 24}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_4']

## 2) Independent metric recomputation vs reported (ORIGINAL dist)

Recomputed with sklearn from predictions/valid.npz; cross-checked vs metrics.json final.

| metric | reported | recomputed (sklearn) | match |
|---|---|---|---|
| pr_auc | 0.9968 | 0.9968 | ok |
| f1_macro | 0.7378 | 0.7378 | ok |
| recall_macro | 0.9812 | 0.9812 | ok |
| precision_macro | 0.6644 | 0.6644 | ok |
| auroc | 0.9999 | 0.9999 | ok |
| accuracy | 0.9631 | 0.9631 | ok |

- ORIGINAL 7-metric: acc=0.9631, train_loss=0.0087, val_loss=0.0808, recall_macro=0.9812, precision_macro=0.6644, f1_macro=0.7378, AUROC=0.9999; PR-AUC=0.9968
- confusion (orig, recomputed): [[1254, 49], [0, 24]]

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=256)

- balanced valid counts = {'normal': 24, 'disease_4': 24} (N=48, seed=42)
- BALANCED 7-metric: acc=1.0000, train_loss=0.0087 (training-time), val_loss=0.0361 (recomputed on balanced), recall_macro=1.0000, precision_macro=1.0000, f1_macro=1.0000, AUROC=1.0000; PR-AUC=1.0000
- confusion (balanced): [[24, 0], [0, 24]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_4] = 24/24 = 1.000 (95% CI 0.862-1.000)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours 0.9968 vs baseline best 0.8668 (resnet50) -> abs +0.1300, rel +15.0%
- F1-macro: Ours 0.7378 vs baseline best 0.6785 (efficientnetv2) -> abs +0.0593, rel +8.7%

### BALANCED
- PR-AUC: Ours 1.0000 vs baseline best 0.9983 (densenet121) -> abs +0.0017, rel +0.2%
- F1-macro: Ours 1.0000 vs baseline best 0.9792 (efficientnetv2) -> abs +0.0208, rel +2.1%
