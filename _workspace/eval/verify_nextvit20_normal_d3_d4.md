# verify_nextvit20_normal_d3_d4.md

- task: classification (normal_d3_d4), backbone: nextvit20
- status (reported): ok
- primary metric: pr_auc

## 1) Boundary cross-validation (predictions vs manifest)

- predictions valid N = **1403**, manifest valid N (setting) = **1403** -> MATCH
- predictions label dist = {0: 1303, 1: 76, 2: 24}
- manifest valid label dist = {0: 1303, 1: 76, 2: 24}
- distribution match: **PASS**
- class_names from npz: ['normal', 'disease_3', 'disease_4']
- expected (data_card): normal=1303 + disease (d3=76 / d4=24) non-downsampled valid

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| pr_auc (primary) | 0.5084 | 0.5084 | ok |
| macro_f1 | 0.6189 | 0.6189 | ok |
| recall_disease | 0.8700 | 0.8700 | ok |
| precision_disease | 0.6905 | 0.6905 | ok |
| accuracy | 0.9394 | 0.9394 | ok |

### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch

| metric | reported | recomputed | match |
|---|---|---|---|
| recall (macro) | 0.6742 | 0.6742 | ok |
| precision (macro) | 0.5942 | 0.5942 | ok |
| f1-score (macro) | 0.6189 | 0.6189 | ok |
| AUROC | 0.9629 | 0.9629 | ok |
| train_loss (best ep) | 0.5573 | 0.5573 | ok |
| val_loss (best ep) | 0.1548 | 0.1548 | ok |
| accuracy | 0.9394 | 0.9394 | ok |

- AUROC convention: macro one-vs-rest over 3 classes. best epoch = 12.

Confusion matrix (rows=true, cols=pred), recomputed:

```
[[1264, 26, 13], [9, 42, 25], [4, 8, 12]]
```
reported confusion: [[1264, 26, 13], [9, 42, 25], [4, 8, 12]]

## 3) Small-sample note (Wilson 95% CI)

- valid disease N = **100**

- recall[disease_3] = 42/76 = 0.553 (95% CI 0.441-0.659)
- recall[disease_4] = 12/24 = 0.500 (95% CI 0.314-0.686)

## Per-disease PR-AUC (OvR)

- disease_3: PR-AUC=0.713
- disease_4: PR-AUC=0.304
- macro disease PR-AUC (primary): 0.508