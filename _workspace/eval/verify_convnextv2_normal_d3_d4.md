# verify_convnextv2_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: convnextv2
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
| pr_auc (primary) | 0.3428 | 0.3428 | ok |
| macro_f1 | 0.5227 | 0.5227 | ok |
| recall_disease | 0.9000 | 0.9000 | ok |
| precision_disease | 0.4036 | 0.4036 | ok |
| accuracy | 0.8738 | 0.8738 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.6400 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.4899 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.5227 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9425 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.5876 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.3363 | recomputed-only |
| accuracy | 0.8738 | 0.8738 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 19.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1170, 64, 69], [7, 46, 23], [3, 11, 10]]
```
reported confusion: [[1170, 64, 69], [7, 46, 23], [3, 11, 10]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 46/76 = 0.605 (95% CI 0.493-0.708)
- recall[disease_4] = 10/24 = 0.417 (95% CI 0.245-0.612)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.536
- disease_4: PR-AUC=0.150
- macro disease PR-AUC (primary): 0.343