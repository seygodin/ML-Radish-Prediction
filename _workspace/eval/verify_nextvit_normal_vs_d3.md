# verify_nextvit_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: nextvit
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
| pr_auc (primary) | 0.9293 | 0.9293 | ok |
| macro_f1 | 0.8007 | 0.8007 | ok |
| recall_disease | 0.9868 | 0.9868 | ok |
| precision_disease | 0.4688 | 0.4688 | ok |
| accuracy | 0.9376 | 0.9376 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.9608 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.7340 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.8007 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9945 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.0235 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.1541 | recomputed-only |
| accuracy | 0.9376 | 0.9376 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 47.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1218, 85], [1, 75]]
```
reported confusion: [[1218, 85], [1, 75]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 75/76 = 0.987 (95% CI 0.929-0.998)


