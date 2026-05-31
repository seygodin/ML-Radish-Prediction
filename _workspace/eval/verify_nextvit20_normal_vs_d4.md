# verify_nextvit20_normal_vs_d4.md

- task: classification (normal_vs_d4), backbone: nextvit20
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
| pr_auc (primary) | 0.8694 | 0.8694 | ok |
| macro_f1 | 0.8981 | 0.8981 | ok |
| recall_disease | 0.8333 | 0.8333 | ok |
| precision_disease | 0.7692 | 0.7692 | ok |
| accuracy | 0.9925 | 0.9925 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.9144 | 0.9144 | ok |
| precision (macro) | 0.8831 | 0.8831 | ok |
| f1-score (macro) | 0.8981 | 0.8981 | ok |
| AUROC | 0.9923 | 0.9923 | ok |
| train_loss (best ep) | 0.1019 | 0.1019 | ok |
| val_loss (best ep) | 0.0262 | 0.0262 | ok |
| accuracy | 0.9925 | 0.9925 | ok |

- AUROC convention: ROC-AUC on P(disease) (2-class). best epoch = 35.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1297, 6], [4, 20]]
```
reported confusion: [[1297, 6], [4, 20]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **24**

- recall[disease_4] = 20/24 = 0.833 (95% CI 0.641-0.933)


