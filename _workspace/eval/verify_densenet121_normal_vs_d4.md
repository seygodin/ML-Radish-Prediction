# verify_densenet121_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: densenet121
- status (reported): ok
- primary metric: pr_auc

## 1) Boundary cross-validation (predictions vs manifest)

- predictions valid N = **1327**, manifest valid N (setting) = **1327** -> MATCH
- predictions label dist = {0: 1303, 1: 24}
- manifest valid label dist = {0: 1303, 1: 24}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_4']
- expected (data_card): normal=1303 + disease (d3=76 / d4=24) non-downsampled valid

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| pr_auc (primary) | 0.8572 | 0.8572 | ok |
| macro_f1 | 0.6652 | 0.6652 | ok |
| recall_disease | 1.0000 | 1.0000 | ok |
| precision_disease | 0.2222 | 0.2222 | ok |
| accuracy | 0.9367 | 0.9367 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9678 | 0.9678 | ok |
| precision (macro) | 0.6111 | 0.6111 | ok |
| f1-score (macro) | 0.6652 | 0.6652 | ok |
| AUROC | 0.9948 | 0.9948 | ok |
| train_loss (best ep) | 0.0523 | 0.0523 | ok |
| val_loss (best ep) | 0.1529 | 0.1529 | ok |
| accuracy | 0.9367 | 0.9367 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 23.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1219, 84], [0, 24]]
```
reported confusion: [[1219, 84], [0, 24]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **24**

- recall[disease_4] = 24/24 = 1.000 (95% CI 0.862-1.000)


