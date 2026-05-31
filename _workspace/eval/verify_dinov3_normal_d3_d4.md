# verify_dinov3_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: **dinov3 (DINOv3 ViT-S/16 frozen + 2-layer head)**, Ours
- img_size = **256** (from config.snapshot), num_classes = 3, best epoch = 13
- params: total = **21.6863M** (frozen backbone), trainable = **99.33k** (head only)

## 1) Boundary cross-validation (predictions vs manifest, ORIGINAL dist)

- predictions valid N = **1403**, manifest valid N = **1403** -> MATCH
- predictions label dist = {0: 1303, 1: 76, 2: 24}
- manifest valid label dist = {0: 1303, 1: 76, 2: 24}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_3', 'disease_4']

## 2) Independent metric recomputation vs reported (ORIGINAL dist)

Recomputed with sklearn from predictions/valid.npz; cross-checked vs metrics.json final.

| metric | reported | recomputed (sklearn) | match |
|---|---|---|---|
| pr_auc | 0.6892 | 0.6892 | ok |
| f1_macro | 0.6977 | 0.6977 | ok |
| recall_macro | 0.7534 | 0.7534 | ok |
| precision_macro | 0.6738 | 0.6738 | ok |
| auroc | 0.9913 | 0.9913 | ok |
| accuracy | 0.9608 | 0.9608 | ok |

- ORIGINAL 7-metric: acc=0.9608, train_loss=0.3175, val_loss=0.0921, recall_macro=0.7534, precision_macro=0.6738, f1_macro=0.6977, AUROC=0.9913; PR-AUC=0.6892
- confusion (orig, recomputed): [[1279, 7, 17], [1, 56, 19], [0, 11, 13]]

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=256)

- balanced valid counts = {'normal': 24, 'disease_3': 24, 'disease_4': 24} (N=72, seed=42)
- BALANCED 7-metric: acc=0.7778, train_loss=0.3175 (training-time), val_loss=0.3745 (recomputed on balanced), recall_macro=0.7778, precision_macro=0.7852, f1_macro=0.7743, AUROC=0.9361; PR-AUC=0.8098
- confusion (balanced): [[24, 0, 0], [0, 19, 5], [0, 11, 13]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_3] = 56/76 = 0.737 (95% CI 0.628-0.823)
- recall[disease_4] = 13/24 = 0.542 (95% CI 0.351-0.721)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours 0.6892 vs baseline best 0.5703 (densenet121) -> abs +0.1188, rel +20.8%
- F1-macro: Ours 0.6977 vs baseline best 0.7089 (densenet121) -> abs -0.0112, rel -1.6%

### BALANCED
- PR-AUC: Ours 0.8098 vs baseline best 0.8596 (densenet121) -> abs -0.0498, rel -5.8%
- F1-macro: Ours 0.7743 vs baseline best 0.8503 (densenet121) -> abs -0.0761, rel -8.9%
