# verify_dinov3_base_focal_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: **dinov3_base focal+aug (Ours+ improved = DINOv3 ViT-B/16 frozen @512 + 2-layer head hidden=512 + strong aug + focal loss gamma=2, class_weights=from_meta)**
- img_size = **512** (from config.snapshot), num_classes = 2, best epoch = 16, aug = **strong**, loss = **focal(gamma=2.0, class_weights=[1.0, 1.0])**
- params: total = **86.036M** (frozen ViT-B/16 backbone), trainable = **394.75k** (head only; identical to base-CE)
- NOTE: `from_meta` class_weights resolved to all-ones (train loader is balanced-downsampled) -> focal's **gamma hard-example focusing** is the active mechanism, not alpha re-weighting. **aug+focal changed together** vs base-CE -> the delta below is the JOINT effect (ablation needed to split).

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
| f1_macro | 0.9193 | 0.9193 | ok |
| recall_macro | 0.9965 | 0.9965 | ok |
| precision_macro | 0.8636 | 0.8636 | ok |
| auroc | 1.0000 | 1.0000 | ok |
| accuracy | 0.9932 | 0.9932 | ok |

- ORIGINAL 7-metric: acc=0.9932, train_loss=0.0035 (focal), val_loss=0.0037 (focal), recall_macro=0.9965, precision_macro=0.8636, f1_macro=0.9193, AUROC=1.0000; PR-AUC=1.0000
- confusion (orig, recomputed): [[1294, 9], [0, 24]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.993 | 1.000 |
| disease_4 | 24 | 1.000 | 0.727 |

### CE (base, default aug) -> focal+aug per-class change (ORIGINAL dist)

| class | support | recall CE | recall focal | precision CE | precision focal |
|---|---|---|---|---|---|
| disease_4 | 24 | 1.000 | 1.000 | 0.585 | 0.727 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 24, 'disease_4': 24} (N=48, seed=42)
- BALANCED 7-metric: acc=1.0000, val_loss=0.0163 (plain CE on balanced), recall_macro=1.0000, precision_macro=1.0000, f1_macro=1.0000, AUROC=1.0000; PR-AUC=1.0000
- confusion (balanced): [[24, 0], [0, 24]]

## 4) Small-sample note (Wilson 95% CI on disease recall/precision, ORIGINAL dist)

- recall[disease_4] = 24/24 = 1.000 (95% CI 0.862-1.000); precision[disease_4] = 24/33 = 0.727 (95% CI 0.558-0.849)

- NOTE: disease_4 valid N=24 (orig) is small -> CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: focal+aug 1.0000 vs baseline best 0.8668 (resnet50) -> abs +0.1332, rel +15.4%
- F1-macro: focal+aug 0.9193 vs baseline best 0.6785 (efficientnetv2) -> abs +0.2408, rel +35.5%

### BALANCED
- PR-AUC: focal+aug 1.0000 vs baseline best 0.9983 (densenet121) -> abs +0.0017, rel +0.2%
- F1-macro: focal+aug 1.0000 vs baseline best 0.9792 (efficientnetv2) -> abs +0.0208, rel +2.1%

## 6) base-CE (default aug) -> base-focal+aug improvement (JOINT aug+focal effect)

### ORIGINAL dist
- PR-AUC: CE 1.0000 -> focal+aug 1.0000 (Δ -0.0000)
- F1-macro: CE 0.8659 -> focal+aug 0.9193 (Δ +0.0534)

### BALANCED
- PR-AUC: CE 1.0000 -> focal+aug 1.0000 (Δ +0.0000)
- F1-macro: CE 1.0000 -> focal+aug 1.0000 (Δ +0.0000)
