# verify_densenet121_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: densenet121
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
| pr_auc (primary) | 0.5703 | 0.5703 | ok |
| macro_f1 | 0.7089 | 0.7089 | ok |
| recall_disease | 0.9800 | 0.9800 | ok |
| precision_disease | 0.6012 | 0.6012 | ok |
| accuracy | 0.9401 | 0.9401 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.8620 | 0.8620 | ok |
| precision (macro) | 0.6419 | 0.6419 | ok |
| f1-score (macro) | 0.7089 | 0.7089 | ok |
| AUROC | 0.9844 | 0.9844 | ok |
| train_loss (best ep) | 0.2163 | 0.2163 | ok |
| val_loss (best ep) | 0.1943 | 0.1943 | ok |
| accuracy | 0.9401 | 0.9401 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 42.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1238, 35, 30], [2, 61, 13], [0, 4, 20]]
```
reported confusion: [[1238, 35, 30], [2, 61, 13], [0, 4, 20]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 61/76 = 0.803 (95% CI 0.700-0.877)
- recall[disease_4] = 20/24 = 0.833 (95% CI 0.641-0.933)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.728
- disease_4: PR-AUC=0.413
- macro disease PR-AUC (primary): 0.570