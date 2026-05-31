# verify_dinov3_base_focal_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: **dinov3_base focal+aug (Ours+ improved = DINOv3 ViT-B/16 frozen @512 + 2-layer head hidden=512 + strong aug + focal loss gamma=2, class_weights=from_meta)**
- img_size = **512** (from config.snapshot), num_classes = 2, best epoch = 20, aug = **strong**, loss = **focal(gamma=2.0, class_weights=[1.0, 1.0])**
- params: total = **86.036M** (frozen ViT-B/16 backbone), trainable = **394.75k** (head only; identical to base-CE)
- NOTE: `from_meta` class_weights resolved to all-ones (train loader is balanced-downsampled) -> focal's **gamma hard-example focusing** is the active mechanism, not alpha re-weighting. **aug+focal changed together** vs base-CE -> the delta below is the JOINT effect (ablation needed to split).

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
| pr_auc | 0.9998 | 0.9998 | ok |
| f1_macro | 0.9897 | 0.9897 | ok |
| recall_macro | 0.9988 | 0.9988 | ok |
| precision_macro | 0.9810 | 0.9810 | ok |
| auroc | 1.0000 | 1.0000 | ok |
| accuracy | 0.9978 | 0.9978 | ok |

- ORIGINAL 7-metric: acc=0.9978, train_loss=0.0022 (focal), val_loss=0.0020 (focal), recall_macro=0.9988, precision_macro=0.9810, f1_macro=0.9897, AUROC=1.0000; PR-AUC=0.9998
- confusion (orig, recomputed): [[1300, 3], [0, 76]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.998 | 1.000 |
| disease_3 | 76 | 1.000 | 0.962 |

### CE (base, default aug) -> focal+aug per-class change (ORIGINAL dist)

| class | support | recall CE | recall focal | precision CE | precision focal |
|---|---|---|---|---|---|
| disease_3 | 76 | 1.000 | 1.000 | 0.938 | 0.962 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 76, 'disease_3': 76} (N=152, seed=42)
- BALANCED 7-metric: acc=0.9934, val_loss=0.0209 (plain CE on balanced), recall_macro=0.9934, precision_macro=0.9935, f1_macro=0.9934, AUROC=1.0000; PR-AUC=1.0000
- confusion (balanced): [[75, 1], [0, 76]]

## 4) Small-sample note (Wilson 95% CI on disease recall/precision, ORIGINAL dist)

- recall[disease_3] = 76/76 = 1.000 (95% CI 0.952-1.000); precision[disease_3] = 76/79 = 0.962 (95% CI 0.894-0.987)

- NOTE: disease_4 valid N=24 (orig) is small -> CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: focal+aug 0.9998 vs baseline best 0.9647 (resnet50) -> abs +0.0351, rel +3.6%
- F1-macro: focal+aug 0.9897 vs baseline best 0.8604 (densenet121) -> abs +0.1294, rel +15.0%

### BALANCED
- PR-AUC: focal+aug 1.0000 vs baseline best 0.9997 (resnet50) -> abs +0.0003, rel +0.0%
- F1-macro: focal+aug 0.9934 vs baseline best 0.9803 (densenet121) -> abs +0.0132, rel +1.3%

## 6) base-CE (default aug) -> base-focal+aug improvement (JOINT aug+focal effect)

### ORIGINAL dist
- PR-AUC: CE 1.0000 -> focal+aug 0.9998 (Δ -0.0002)
- F1-macro: CE 0.9831 -> focal+aug 0.9897 (Δ +0.0066)

### BALANCED
- PR-AUC: CE 1.0000 -> focal+aug 1.0000 (Δ +0.0000)
- F1-macro: CE 0.9934 -> focal+aug 0.9934 (Δ +0.0000)
