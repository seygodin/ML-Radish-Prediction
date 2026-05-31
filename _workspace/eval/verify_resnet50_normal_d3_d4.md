# verify_resnet50_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: resnet50
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
| pr_auc (primary) | 0.5302 | 0.5302 | ok |
| macro_f1 | 0.6847 | 0.6847 | ok |
| recall_disease | 0.9500 | 0.9500 | ok |
| precision_disease | 0.6884 | 0.6884 | ok |
| accuracy | 0.9494 | 0.9494 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.7617 | 0.7617 | ok |
| precision (macro) | 0.6403 | 0.6403 | ok |
| f1-score (macro) | 0.6847 | 0.6847 | ok |
| AUROC | 0.9828 | 0.9828 | ok |
| train_loss (best ep) | 0.3146 | 0.3146 | ok |
| val_loss (best ep) | 0.1461 | 0.1461 | ok |
| accuracy | 0.9494 | 0.9494 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 41.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1260, 23, 20], [3, 59, 14], [2, 9, 13]]
```
reported confusion: [[1260, 23, 20], [3, 59, 14], [2, 9, 13]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 59/76 = 0.776 (95% CI 0.671-0.855)
- recall[disease_4] = 13/24 = 0.542 (95% CI 0.351-0.721)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.751
- disease_4: PR-AUC=0.309
- macro disease PR-AUC (primary): 0.530