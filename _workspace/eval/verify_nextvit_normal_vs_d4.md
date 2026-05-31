# verify_nextvit_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: nextvit
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
| pr_auc (primary) | 0.8222 | 0.8222 | ok |
| macro_f1 | 0.6313 | 0.6313 | ok |
| recall_disease | 0.9583 | 0.9583 | ok |
| precision_disease | 0.1811 | 0.1811 | ok |
| accuracy | 0.9209 | 0.9209 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.9393 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.5901 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.6313 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9899 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.1144 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 0.2022 | recomputed-only |
| accuracy | 0.9209 | 0.9209 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 32.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1199, 104], [1, 23]]
```
reported confusion: [[1199, 104], [1, 23]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **24**

- recall[disease_4] = 23/24 = 0.958 (95% CI 0.798-0.993)


