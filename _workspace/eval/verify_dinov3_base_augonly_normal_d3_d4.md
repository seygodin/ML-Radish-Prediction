# verify_dinov3_base_augonly_normal_d3_d4.md

- task: classification (normal_d3_d4) ablation run — **aug-only**
- arch = **dinov3_base** (DINOv3 ViT-B/16 frozen @512, 2-layer head hidden512), trainable = **395267** (head only, identical to base/focal reference)
- loss = **cross_entropy**(gamma=None, class_weights=[1.0, 1.0, 1.0], ls=0.0), aug = **strong**, best epoch = 26
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
| pr_auc | 0.7558 | 0.7558 | ok |
| f1_macro | 0.7679 | 0.7679 | ok |
| recall_macro | 0.8420 | 0.8420 | ok |
| precision_macro | 0.7713 | 0.7713 | ok |
| auroc | 0.9946 | 0.9946 | ok |
| accuracy | 0.9729 | 0.9729 | ok |

- 7-metric (recomputed): acc=0.9729, recall_macro=0.8420, precision_macro=0.7713, f1_macro=0.7679, AUROC=0.9946; PR-AUC=0.7558
- confusion (recomputed): [[1294, 1, 8], [1, 50, 25], [0, 3, 21]]
- sklearn crosscheck: PR-AUC[d3]=0.9226, PR-AUC[d4]=0.5890 (disease OvR macro = mean)

## 3) Per-class recall/precision (original dist)

| class | support | recall | precision |
|---|---|---|---|
| normal | 1303 | 0.993 | 0.999 |
| disease_3 | 76 | 0.658 | 0.926 |
| disease_4 | 24 | 0.875 | 0.389 |

## 4) Small-sample note (Wilson 95% CI on d4, N=24)

- d4 recall = 21/24 = 0.875 (95% CI 0.690-0.957)
- d4 precision = 21/54 = 0.389 (95% CI 0.270-0.522)
- d4 valid N=24 (orig) is tiny -> CIs wide; single-seed deltas <0.02 are within noise.

## 5) Delta vs reference base (CE / default aug)

- pr_auc: base 0.7452 -> this 0.7558 (Δ +0.0105)
- f1_macro: base 0.7502 -> this 0.7679 (Δ +0.0177)
- accuracy: base 0.9672 -> this 0.9729 (Δ +0.0057)
- auroc: base 0.9927 -> this 0.9946 (Δ +0.0019)
- d4 precision: base 0.339 -> this 0.389 (Δ +0.050); d4 recall: base 0.875 -> this 0.875 (Δ +0.000)
