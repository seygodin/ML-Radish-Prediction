# verify_nextvit20_normal_vs_d3.md

- task: classification (normal_vs_d3), backbone: nextvit20
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
| pr_auc (primary) | 0.9307 | 0.9307 | ok |
| macro_f1 | 0.9099 | 0.9099 | ok |
| recall_disease | 0.8684 | 0.8684 | ok |
| precision_disease | 0.7952 | 0.7952 | ok |
| accuracy | 0.9804 | 0.9804 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9277 | 0.9277 | ok |
| precision (macro) | 0.8937 | 0.8937 | ok |
| f1-score (macro) | 0.9099 | 0.9099 | ok |
| AUROC | 0.9941 | 0.9941 | ok |
| train_loss (best ep) | 0.0737 | 0.0737 | ok |
| val_loss (best ep) | 0.0455 | 0.0455 | ok |
| accuracy | 0.9804 | 0.9804 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 22.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1286, 17], [10, 66]]
```
reported confusion: [[1286, 17], [10, 66]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **76**

- recall[disease_3] = 66/76 = 0.868 (95% CI 0.774-0.927)


