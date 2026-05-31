# verify_efficientnetv2_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: efficientnetv2
- status (reported): ok
- primary metric: pr_auc

## 1) Boundary cross-validation (predictions vs manifest)

- predictions valid N = **1403**, manifest valid N (setting) = **1403** -> MATCH
- predictions label dist = {0: 1303, 1: 76, 2: 24}
- manifest valid label dist = {0: 1303, 1: 76, 2: 24}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_3', 'disease_4']
- expected (data_card): normal=1303 + disease (d3=76 / d4=24) non-downsampled valid

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| pr_auc (primary) | 0.5090 | 0.5090 | ok |
| macro_f1 | 0.5913 | 0.5913 | ok |
| recall_disease | 0.9500 | 0.9500 | ok |
| precision_disease | 0.5588 | 0.5588 | ok |
| accuracy | 0.9202 | 0.9202 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.6760 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.5454 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.5913 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9774 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.5805 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.2309 | recomputed-only |
| accuracy | 0.9202 | 0.9202 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 13.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1228, 62, 13], [4, 54, 18], [1, 14, 9]]
```
reported confusion: [[1228, 62, 13], [4, 54, 18], [1, 14, 9]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 54/76 = 0.711 (95% CI 0.600-0.800)
- recall[disease_4] = 9/24 = 0.375 (95% CI 0.212-0.573)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.691
- disease_4: PR-AUC=0.327
- macro disease PR-AUC (primary): 0.509