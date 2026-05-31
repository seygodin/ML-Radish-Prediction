# verify_efficientnetv2_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: efficientnetv2
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
| pr_auc (primary) | 0.9493 | 0.9493 | ok |
| macro_f1 | 0.7992 | 0.7992 | ok |
| recall_disease | 0.9868 | 0.9868 | ok |
| precision_disease | 0.4658 | 0.4658 | ok |
| accuracy | 0.9369 | 0.9369 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.9604 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.7325 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.7992 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9954 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.0173 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.2011 | recomputed-only |
| accuracy | 0.9369 | 0.9369 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 59.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1217, 86], [1, 75]]
```
reported confusion: [[1217, 86], [1, 75]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 75/76 = 0.987 (95% CI 0.929-0.998)


