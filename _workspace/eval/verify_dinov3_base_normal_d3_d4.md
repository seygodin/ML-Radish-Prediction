# verify_dinov3_base_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: **dinov3_base (Ours+ = DINOv3 ViT-B/16 frozen @512 + 2-layer head, hidden=512)**
- img_size = **512** (from config.snapshot), num_classes = 3, best epoch = 29
- params: total = **86.0365M** (frozen ViT-B/16 backbone), trainable = **395.27k** (head only)

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
| pr_auc | 0.7452 | 0.7452 | ok |
| f1_macro | 0.7502 | 0.7502 | ok |
| recall_macro | 0.8399 | 0.8399 | ok |
| precision_macro | 0.7546 | 0.7546 | ok |
| auroc | 0.9927 | 0.9927 | ok |
| accuracy | 0.9672 | 0.9672 | ok |

- ORIGINAL 7-metric: acc=0.9672, train_loss=0.1625, val_loss=0.0831, recall_macro=0.8399, precision_macro=0.7546, f1_macro=0.7502, AUROC=0.9927; PR-AUC=0.7452
- confusion (orig, recomputed): [[1286, 1, 16], [1, 50, 25], [0, 3, 21]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.987 | 0.999 |
| disease_3 | 76 | 0.658 | 0.926 |
| disease_4 | 24 | 0.875 | 0.339 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 24, 'disease_3': 24, 'disease_4': 24} (N=72, seed=42)
- BALANCED 7-metric: acc=0.8750, train_loss=0.1625 (training-time), val_loss=0.3196 (recomputed on balanced), recall_macro=0.8750, precision_macro=0.8783, f1_macro=0.8745, AUROC=0.9618; PR-AUC=0.8846
- confusion (balanced): [[24, 0, 0], [0, 18, 6], [0, 3, 21]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_3] = 50/76 = 0.658 (95% CI 0.546-0.755)
- recall[disease_4] = 21/24 = 0.875 (95% CI 0.690-0.957)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours+ 0.7452 vs baseline best 0.5703 (densenet121) -> abs +0.1749, rel +30.7%
- F1-macro: Ours+ 0.7502 vs baseline best 0.7089 (densenet121) -> abs +0.0413, rel +5.8%

### BALANCED
- PR-AUC: Ours+ 0.8846 vs baseline best 0.8596 (densenet121) -> abs +0.0250, rel +2.9%
- F1-macro: Ours+ 0.8745 vs baseline best 0.8503 (densenet121) -> abs +0.0242, rel +2.8%

## 6) small@256 -> base@512 improvement

### ORIGINAL dist
- PR-AUC: small 0.6892 -> base 0.7452 (Δ +0.0561)
- F1-macro: small 0.6977 -> base 0.7502 (Δ +0.0525)

### BALANCED
- PR-AUC: small 0.8098 -> base 0.8846 (Δ +0.0748)
- F1-macro: small 0.7743 -> base 0.8745 (Δ +0.1003)
