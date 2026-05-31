# verify_dinov3_base_focalg1_normal_d3_d4.md

- task: classification (normal_d3_d4) ablation run — **gamma=1 (strong aug)**
- arch = **dinov3_base** (DINOv3 ViT-B/16 frozen @512, 2-layer head hidden512), trainable = **395267** (head only, identical to base/focal reference)
- loss = **focal**(gamma=1.0, class_weights=[1.0, 1.0, 1.0], ls=0.0), aug = **strong**, best epoch = 26
- NOTE: class_weights `from_meta` -> all-ones (balanced-downsampled train loader); focal acts via **gamma** (hard-example focusing), not alpha. Single seed=42; d4 valid N=24 -> wide CI.

## 1) Boundary cross-validation (predictions vs manifest, original dist)

- predictions valid N = **1403**, manifest valid N = **1403** -> MATCH
- predictions label dist = {0: 1303, 1: 76, 2: 24}
- manifest valid label dist = {0: 1303, 1: 76, 2: 24}
- distribution match: **PASS**  (no leakage / mislabel / split violation)
- class_names from npz: ['normal', 'disease_3', 'disease_4']

## 2) Independent recomputation (sklearn) vs reported (metrics.json.final)

| metric | reported | recomputed (sklearn) | match (|Δ|≤0.01) |
|---|---|---|---|
| pr_auc | 0.7579 | 0.7579 | ok |
| f1_macro | 0.7823 | 0.7823 | ok |
| recall_macro | 0.8549 | 0.8549 | ok |
| precision_macro | 0.7749 | 0.7749 | ok |
| auroc | 0.9947 | 0.9947 | ok |
| accuracy | 0.9743 | 0.9743 | ok |

- 7-metric (recomputed): acc=0.9743, recall_macro=0.8549, precision_macro=0.7749, f1_macro=0.7823, AUROC=0.9947; PR-AUC=0.7579
- confusion (recomputed): [[1293, 2, 8], [1, 53, 22], [0, 3, 21]]
- sklearn crosscheck: PR-AUC[d3]=0.9212, PR-AUC[d4]=0.5946 (disease OvR macro = mean)

## 3) Per-class recall/precision (original dist)

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.992 | 0.999 |
| disease_3 | 76 | 0.697 | 0.914 |
| disease_4 | 24 | 0.875 | 0.412 |

## 4) Small-sample note (Wilson 95% CI on d4, N=24)

- d4 recall = 21/24 = 0.875 (95% CI 0.690-0.957)
- d4 precision = 21/51 = 0.412 (95% CI 0.288-0.548)
- d4 valid N=24 (orig) is tiny -> CIs wide; single-seed deltas <0.02 are within noise.

## 5) Delta vs reference base (CE / default aug)

- pr_auc: base 0.7452 -> this 0.7579 (Δ +0.0127)
- f1_macro: base 0.7502 -> this 0.7823 (Δ +0.0321)
- accuracy: base 0.9672 -> this 0.9743 (Δ +0.0071)
- auroc: base 0.9927 -> this 0.9947 (Δ +0.0020)
- d4 precision: base 0.339 -> this 0.412 (Δ +0.073); d4 recall: base 0.875 -> this 0.875 (Δ +0.000)
