# verify_mamba_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: mamba
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
| pr_auc (primary) | 0.8776 | 0.8776 | ok |
| macro_f1 | 0.7292 | 0.7292 | ok |
| recall_disease | 0.9868 | 0.9868 | ok |
| precision_disease | 0.3488 | 0.3488 | ok |
| accuracy | 0.8978 | 0.8978 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9397 | 0.9397 | ok |
| precision (macro) | 0.6740 | 0.6740 | ok |
| f1-score (macro) | 0.7292 | 0.7292 | ok |
| AUROC | 0.9887 | 0.9887 | ok |
| train_loss (best ep) | 0.0447 | 0.0447 | ok |
| val_loss (best ep) | 0.2914 | 0.2914 | ok |
| accuracy | 0.8978 | 0.8978 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 39.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1163, 140], [1, 75]]
```
reported confusion: [[1163, 140], [1, 75]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 75/76 = 0.987 (95% CI 0.929-0.998)


