# verify_resnet50_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: resnet50
- status (reported): ok
- primary metric: pr_auc

## 1) Boundary cross-validation (predictions vs manifest)

- predictions valid N = **1379**, manifest valid N (setting) = **1379** -> MATCH
- predictions label dist = {0: 1303, 1: 76}
- manifest valid label dist = {0: 1303, 1: 76}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_3']
- expected (data_card): normal=1303 + disease (d3=76 / d4=24) non-downsampled valid

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| pr_auc (primary) | 0.9647 | 0.9647 | ok |
| macro_f1 | 0.8486 | 0.8486 | ok |
| recall_disease | 1.0000 | 1.0000 | ok |
| precision_disease | 0.5630 | 0.5630 | ok |
| accuracy | 0.9572 | 0.9572 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9774 | 0.9774 | ok |
| precision (macro) | 0.7815 | 0.7815 | ok |
| f1-score (macro) | 0.8486 | 0.8486 | ok |
| AUROC | 0.9979 | 0.9979 | ok |
| train_loss (best ep) | 0.0245 | 0.0245 | ok |
| val_loss (best ep) | 0.1140 | 0.1140 | ok |
| accuracy | 0.9572 | 0.9572 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 39.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1244, 59], [0, 76]]
```
reported confusion: [[1244, 59], [0, 76]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 76/76 = 1.000 (95% CI 0.952-1.000)


