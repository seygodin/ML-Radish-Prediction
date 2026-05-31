# verify_nextvit_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: nextvit
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
| pr_auc (primary) | 0.4872 | 0.4872 | ok |
| macro_f1 | 0.6245 | 0.6245 | ok |
| recall_disease | 0.7200 | 0.7200 | ok |
| precision_disease | 0.8372 | 0.8372 | ok |
| accuracy | 0.9515 | 0.9515 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.6170 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.6499 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.6245 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9694 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.4295 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.1312 | recomputed-only |
| accuracy | 0.9515 | 0.9515 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 26.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1289, 8, 6], [21, 37, 18], [7, 8, 9]]
```
reported confusion: [[1289, 8, 6], [21, 37, 18], [7, 8, 9]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 37/76 = 0.487 (95% CI 0.378-0.597)
- recall[disease_4] = 9/24 = 0.375 (95% CI 0.212-0.573)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.745
- disease_4: PR-AUC=0.229
- macro disease PR-AUC (primary): 0.487