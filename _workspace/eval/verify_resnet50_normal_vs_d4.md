# verify_resnet50_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: resnet50
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
| pr_auc (primary) | 0.8668 | 0.8668 | ok |
| macro_f1 | 0.6684 | 0.6684 | ok |
| recall_disease | 0.9583 | 0.9583 | ok |
| precision_disease | 0.2277 | 0.2277 | ok |
| accuracy | 0.9405 | 0.9405 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9492 | 0.9492 | ok |
| precision (macro) | 0.6135 | 0.6135 | ok |
| f1-score (macro) | 0.6684 | 0.6684 | ok |
| AUROC | 0.9909 | 0.9909 | ok |
| train_loss (best ep) | 0.0575 | 0.0575 | ok |
| val_loss (best ep) | 0.1466 | 0.1466 | ok |
| accuracy | 0.9405 | 0.9405 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 39.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1225, 78], [1, 23]]
```
reported confusion: [[1225, 78], [1, 23]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **24**

- recall[disease_4] = 23/24 = 0.958 (95% CI 0.798-0.993)


