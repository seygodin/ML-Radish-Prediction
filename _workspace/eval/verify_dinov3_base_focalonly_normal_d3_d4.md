# verify_dinov3_base_focalonly_normal_d3_d4.md

- task: classification (normal_d3_d4) ablation run — **focal-only**
- arch = **dinov3_base** (DINOv3 ViT-B/16 frozen @512, 2-layer head hidden512), trainable = **395267** (head only, identical to base/focal reference)
- loss = **focal**(gamma=2.0, class_weights=[1.0, 1.0, 1.0], ls=0.0), aug = **default**, best epoch = 28
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
| pr_auc | 0.7364 | 0.7364 | ok |
| f1_macro | 0.7350 | 0.7350 | ok |
| recall_macro | 0.8403 | 0.8403 | ok |
| precision_macro | 0.7186 | 0.7186 | ok |
| auroc | 0.9930 | 0.9930 | ok |
| accuracy | 0.9601 | 0.9601 | ok |

- 7-metric (recomputed): acc=0.9601, recall_macro=0.8403, precision_macro=0.7186, f1_macro=0.7350, AUROC=0.9930; PR-AUC=0.7364
- confusion (recomputed): [[1273, 5, 25], [0, 54, 22], [0, 4, 20]]
- sklearn crosscheck: PR-AUC[d3]=0.9038, PR-AUC[d4]=0.5690 (disease OvR macro = mean)

## 3) Per-class recall/precision (original dist)

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.977 | 1.000 |
| disease_3 | 76 | 0.711 | 0.857 |
| disease_4 | 24 | 0.833 | 0.299 |

## 4) Small-sample note (Wilson 95% CI on d4, N=24)

- d4 recall = 20/24 = 0.833 (95% CI 0.641-0.933)
- d4 precision = 20/67 = 0.299 (95% CI 0.202-0.417)
- d4 valid N=24 (orig) is tiny -> CIs wide; single-seed deltas <0.02 are within noise.

## 5) Delta vs reference base (CE / default aug)

- pr_auc: base 0.7452 -> this 0.7364 (Δ -0.0088)
- f1_macro: base 0.7502 -> this 0.7350 (Δ -0.0153)
- accuracy: base 0.9672 -> this 0.9601 (Δ -0.0071)
- auroc: base 0.9927 -> this 0.9930 (Δ +0.0003)
- d4 precision: base 0.339 -> this 0.299 (Δ -0.040); d4 recall: base 0.875 -> this 0.833 (Δ -0.042)
