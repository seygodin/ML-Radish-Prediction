# verify_efficientnetv2_detection_singlebox.md

- task: detection single-box, backbone: efficientnetv2
- status (reported): ok
- primary: det_pr_auc (det_pr_auc)

## 1) Boundary cross-validation (predictions vs manifest)

- predictions total valid images N = **1403** (include_normal=True)
- positives (disease, is_positive=True, non-empty GT) = **100**; negatives (normal, empty GT) = **1303**
- manifest detection valid (disease only) = **100** (d3=76 + d4=24); classification manifest valid normal = **1303** -> total **1403** matches predictions N
- reported meta valid_counts = {'disease_3': 76, 'disease_4': 24, 'normal': 1303}; train_counts = {'disease_3': 470, 'disease_4': 227, 'normal': 697}
- N match (1403): **PASS**; pos/neg (100/1303): **PASS**; reported n_positive=100, n_negative=1303 match recomputed 100/1303

### Objectness collapse RESOLVED (normal vs disease score separation)
- disease objectness: median=0.9987, p25=0.9949, min=0.6005
- normal  objectness: median=0.0021, p75=0.0034, p95=0.0235, max=0.9216
- normals scored >=0.5: **7/1303** (0.5%)
- Separation is clean: disease median ~1.00 vs normal median ~0.002. The earlier collapse (objectness ~1.0 for ALL images incl. normals) is FIXED — objectness now functions as a real image-level disease score.

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| det_pr_auc (primary) | 0.9990 | 0.9990 | ok |
| det_roc_auc | 0.9999 | 0.9999 | ok |
| presence_recall@0.5 | 1.0000 | 1.0000 | ok |
| fp_rate@0.5 | 0.0054 | 0.0054 | ok |
| iou_at_0.5_presence | 0.6400 | 0.6400 | ok |
| iou median (positives) | 0.5764 | 0.5764 | ok |
| iou mean (positives) | 0.5736 | 0.5736 | ok |
| map_at_0.5 | 0.4890 | 0.4940 | ok |

## 3) Image-level disease detection (primary interpretation)

- det_pr_auc = **0.9990**, det_roc_auc = **0.9999** (n_pos=100, n_neg=1303)
- presence_recall@0.5 (disease recall) = **1.000** = 100/100 (95% CI 0.963-1.000)
- fp_rate@0.5 (normal false alarm) = **0.0054** = 7/1303 (95% CI 0.0026-0.0110)

## 4) Localization quality (positives only, coarse-box caveat)

- IoU(pred,GT) on positive images (n=100): median=0.576, mean=0.574, p25=0.429, p75=0.756, min=0.111, max=0.994
- frac positives localized @IoU>=0.5 = 0.640
- mAP@0.5 (objectness-ranked, normal preds = FP) = **0.494** — interpret SEPARATELY from det_pr_auc: mAP couples ranking with coarse-box localization, so it is bounded by the IoU>=0.5 hit rate, not by disease/normal separability.
- NOTE: GT boxes are coarse (median ~50% of frame, center-biased). IoU median ~0.6 reflects crop-region localization, NOT fine lesion pinpointing (REPORT sec.6).

## 5) Config notes

- meta: {"train_counts": {"disease_3": 470, "disease_4": 227, "normal": 697}, "valid_counts": {"disease_3": 76, "disease_4": 24, "normal": 1303}, "include_normal": true}
