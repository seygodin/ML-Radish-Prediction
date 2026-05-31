# verify_dinov3_base_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: **dinov3_base (Ours+ = DINOv3 ViT-B/16 frozen @512 + 2-layer head, hidden=512)**
- img_size = **512** (from config.snapshot), num_classes = 2, best epoch = 7
- params: total = **86.036M** (frozen ViT-B/16 backbone), trainable = **394.75k** (head only)

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
| pr_auc | 1.0000 | 1.0000 | ok |
| f1_macro | 0.8659 | 0.8659 | ok |
| recall_macro | 0.9935 | 0.9935 | ok |
| precision_macro | 0.7927 | 0.7927 | ok |
| auroc | 1.0000 | 1.0000 | ok |
| accuracy | 0.9872 | 0.9872 | ok |

- ORIGINAL 7-metric: acc=0.9872, train_loss=0.0023, val_loss=0.0280, recall_macro=0.9935, precision_macro=0.7927, f1_macro=0.8659, AUROC=1.0000; PR-AUC=1.0000
- confusion (orig, recomputed): [[1286, 17], [0, 24]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.987 | 1.000 |
| disease_4 | 24 | 1.000 | 0.585 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 24, 'disease_4': 24} (N=48, seed=42)
- BALANCED 7-metric: acc=1.0000, train_loss=0.0023 (training-time), val_loss=0.0037 (recomputed on balanced), recall_macro=1.0000, precision_macro=1.0000, f1_macro=1.0000, AUROC=1.0000; PR-AUC=1.0000
- confusion (balanced): [[24, 0], [0, 24]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_4] = 24/24 = 1.000 (95% CI 0.862-1.000)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours+ 1.0000 vs baseline best 0.8668 (resnet50) -> abs +0.1332, rel +15.4%
- F1-macro: Ours+ 0.8659 vs baseline best 0.6785 (efficientnetv2) -> abs +0.1874, rel +27.6%

### BALANCED
- PR-AUC: Ours+ 1.0000 vs baseline best 0.9983 (densenet121) -> abs +0.0017, rel +0.2%
- F1-macro: Ours+ 1.0000 vs baseline best 0.9792 (efficientnetv2) -> abs +0.0208, rel +2.1%

## 6) small@256 -> base@512 improvement

### ORIGINAL dist
- PR-AUC: small 0.9968 -> base 1.0000 (Δ +0.0032)
- F1-macro: small 0.7378 -> base 0.8659 (Δ +0.1281)

### BALANCED
- PR-AUC: small 1.0000 -> base 1.0000 (Δ +0.0000)
- F1-macro: small 1.0000 -> base 1.0000 (Δ +0.0000)
