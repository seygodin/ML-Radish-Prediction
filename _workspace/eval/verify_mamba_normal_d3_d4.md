# verify_mamba_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: mamba
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
| pr_auc (primary) | 0.3585 | 0.3585 | ok |
| macro_f1 | 0.5237 | 0.5237 | ok |
| recall_disease | 0.9200 | 0.9200 | ok |
| precision_disease | 0.4381 | 0.4381 | ok |
| accuracy | 0.8753 | 0.8753 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.6343 | 0.6343 | ok |
| precision (macro) | 0.5688 | 0.5688 | ok |
| f1-score (macro) | 0.5237 | 0.5237 | ok |
| AUROC | 0.9472 | 0.9472 | ok |
| train_loss (best ep) | 0.7361 | 0.7361 | ok |
| val_loss (best ep) | 0.3395 | 0.3395 | ok |
| accuracy | 0.8753 | 0.8753 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 4.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1185, 10, 108], [6, 28, 42], [2, 7, 15]]
```
reported confusion: [[1185, 10, 108], [6, 28, 42], [2, 7, 15]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 28/76 = 0.368 (95% CI 0.269-0.481)
- recall[disease_4] = 15/24 = 0.625 (95% CI 0.427-0.788)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.581
- disease_4: PR-AUC=0.136
- macro disease PR-AUC (primary): 0.359