# verify_dinov3_base_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: **dinov3_base (Ours+ = DINOv3 ViT-B/16 frozen @512 + 2-layer head, hidden=512)**
- img_size = **512** (from config.snapshot), num_classes = 2, best epoch = 25
- params: total = **86.036M** (frozen ViT-B/16 backbone), trainable = **394.75k** (head only)

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
| pr_auc | 1.0000 | 1.0000 | ok |
| f1_macro | 0.9831 | 0.9831 | ok |
| recall_macro | 0.9981 | 0.9981 | ok |
| precision_macro | 0.9691 | 0.9691 | ok |
| auroc | 1.0000 | 1.0000 | ok |
| accuracy | 0.9964 | 0.9964 | ok |

- ORIGINAL 7-metric: acc=0.9964, train_loss=0.0002, val_loss=0.0103, recall_macro=0.9981, precision_macro=0.9691, f1_macro=0.9831, AUROC=1.0000; PR-AUC=1.0000
- confusion (orig, recomputed): [[1298, 5], [0, 76]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.996 | 1.000 |
| disease_3 | 76 | 1.000 | 0.938 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 76, 'disease_3': 76} (N=152, seed=42)
- BALANCED 7-metric: acc=0.9934, train_loss=0.0002 (training-time), val_loss=0.0171 (recomputed on balanced), recall_macro=0.9934, precision_macro=0.9935, f1_macro=0.9934, AUROC=1.0000; PR-AUC=1.0000
- confusion (balanced): [[75, 1], [0, 76]]

## 4) Small-sample note (Wilson 95% CI on disease recall, ORIGINAL dist)

- recall[disease_3] = 76/76 = 1.000 (95% CI 0.952-1.000)

- NOTE: disease_4 valid N=24 (orig) is small -> recall CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: Ours+ 1.0000 vs baseline best 0.9647 (resnet50) -> abs +0.0353, rel +3.7%
- F1-macro: Ours+ 0.9831 vs baseline best 0.8604 (densenet121) -> abs +0.1228, rel +14.3%

### BALANCED
- PR-AUC: Ours+ 1.0000 vs baseline best 0.9997 (resnet50) -> abs +0.0003, rel +0.0%
- F1-macro: Ours+ 0.9934 vs baseline best 0.9803 (densenet121) -> abs +0.0132, rel +1.3%

## 6) small@256 -> base@512 improvement

### ORIGINAL dist
- PR-AUC: small 0.9964 -> base 1.0000 (Δ +0.0036)
- F1-macro: small 0.9237 -> base 0.9831 (Δ +0.0594)

### BALANCED
- PR-AUC: small 0.9995 -> base 1.0000 (Δ +0.0005)
- F1-macro: small 0.9803 -> base 0.9934 (Δ +0.0132)
