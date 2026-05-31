# verify_convnextv2_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: convnextv2
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
| pr_auc (primary) | 0.7874 | 0.7873 | ok |
| macro_f1 | 0.4364 | 0.4364 | ok |
| recall_disease | 1.0000 | 1.0000 | ok |
| precision_disease | 0.1072 | 0.1072 | ok |
| accuracy | 0.5410 | 0.5410 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | n/a (absent in metrics.json) | 0.7571 | recomputed-only |
| precision (macro) | n/a (absent in metrics.json) | 0.5536 | recomputed-only |
| f1-score (macro) | n/a (absent in metrics.json) | 0.4364 | recomputed-only |
| AUROC | n/a (absent in metrics.json) | 0.9626 | recomputed-only |
| train_loss (best ep) | n/a (absent in metrics.json) | 0.2370 | recomputed-only |
| val_loss (best ep) | n/a (absent in metrics.json) | 1.1785 | recomputed-only |
| accuracy | 0.5410 | 0.5410 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 5.
- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[670, 633], [0, 76]]
```
reported confusion: [[670, 633], [0, 76]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 76/76 = 1.000 (95% CI 0.952-1.000)


