# verify_dinov3_base_focal_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: **dinov3_base focal+aug (Ours+ improved = DINOv3 ViT-B/16 frozen @512 + 2-layer head hidden=512 + strong aug + focal loss gamma=2, class_weights=from_meta)**
- img_size = **512** (from config.snapshot), num_classes = 3, best epoch = 26, aug = **strong**, loss = **focal(gamma=2.0, class_weights=[1.0, 1.0, 1.0])**
- params: total = **86.0365M** (frozen ViT-B/16 backbone), trainable = **395.27k** (head only; identical to base-CE)
- NOTE: `from_meta` class_weights resolved to all-ones (train loader is balanced-downsampled) -> focal's **gamma hard-example focusing** is the active mechanism, not alpha re-weighting. **aug+focal changed together** vs base-CE -> the delta below is the JOINT effect (ablation needed to split).

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
| pr_auc | 0.7647 | 0.7647 | ok |
| f1_macro | 0.7743 | 0.7743 | ok |
| recall_macro | 0.8503 | 0.8503 | ok |
| precision_macro | 0.7692 | 0.7692 | ok |
| auroc | 0.9951 | 0.9951 | ok |
| accuracy | 0.9729 | 0.9729 | ok |

- ORIGINAL 7-metric: acc=0.9729, train_loss=0.0720 (focal), val_loss=0.0194 (focal), recall_macro=0.8503, precision_macro=0.7692, f1_macro=0.7743, AUROC=0.9951; PR-AUC=0.7647
- confusion (orig, recomputed): [[1292, 2, 9], [1, 52, 23], [0, 3, 21]]

### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.992 | 0.999 |
| disease_3 | 76 | 0.684 | 0.912 |
| disease_4 | 24 | 0.875 | 0.396 |

### CE (base, default aug) -> focal+aug per-class change (ORIGINAL dist)

| class | support | recall CE | recall focal | precision CE | precision focal |
|---|---|---|---|---|---|
| disease_3 | 76 | 0.658 | 0.684 | 0.926 | 0.912 |
| disease_4 | 24 | 0.875 | 0.875 | 0.339 | 0.396 |

## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)

- balanced valid counts = {'normal': 24, 'disease_3': 24, 'disease_4': 24} (N=72, seed=42)
- BALANCED 7-metric: acc=0.8750, val_loss=0.3499 (plain CE on balanced), recall_macro=0.8750, precision_macro=0.8783, f1_macro=0.8745, AUROC=0.9520; PR-AUC=0.8563
- confusion (balanced): [[24, 0, 0], [0, 18, 6], [0, 3, 21]]

## 4) Small-sample note (Wilson 95% CI on disease recall/precision, ORIGINAL dist)

- recall[disease_3] = 52/76 = 0.684 (95% CI 0.573-0.778); precision[disease_3] = 52/57 = 0.912 (95% CI 0.811-0.962)
- recall[disease_4] = 21/24 = 0.875 (95% CI 0.690-0.957); precision[disease_4] = 21/53 = 0.396 (95% CI 0.276-0.531)

- NOTE: disease_4 valid N=24 (orig) is small -> CI wide; balanced 3-class N=72 -> per-class N=24.

## 5) Uplift vs baseline best

### ORIGINAL dist
- PR-AUC: focal+aug 0.7647 vs baseline best 0.5703 (densenet121) -> abs +0.1944, rel +34.1%
- F1-macro: focal+aug 0.7743 vs baseline best 0.7089 (densenet121) -> abs +0.0654, rel +9.2%

### BALANCED
- PR-AUC: focal+aug 0.8563 vs baseline best 0.8596 (densenet121) -> abs -0.0033, rel -0.4%
- F1-macro: focal+aug 0.8745 vs baseline best 0.8503 (densenet121) -> abs +0.0242, rel +2.8%

## 6) base-CE (default aug) -> base-focal+aug improvement (JOINT aug+focal effect)

### ORIGINAL dist
- PR-AUC: CE 0.7452 -> focal+aug 0.7647 (Δ +0.0195)
- F1-macro: CE 0.7502 -> focal+aug 0.7743 (Δ +0.0240)

### BALANCED
- PR-AUC: CE 0.8846 -> focal+aug 0.8563 (Δ -0.0283)
- F1-macro: CE 0.8745 -> focal+aug 0.8745 (Δ +0.0000)
