# verify_nextvit20_detection_singlebox.md

- task: detection single-box, backbone: nextvit20
- status (reported): ok
- primary: det_pr_auc (det_pr_auc)

## 1) Boundary cross-validation (predictions vs manifest)

- predictions total valid images N = **1403** (include_normal=True)
- positives (disease, is_positive=True, non-empty GT) = **100**; negatives (normal, empty GT) = **1303**
- manifest detection valid (disease only) = **100** (d3=76 + d4=24); classification manifest valid normal = **1303** -> total **1403** matches predictions N
- reported meta valid_counts = {'disease_3': 76, 'disease_4': 24, 'normal': 1303}; train_counts = {'disease_3': 470, 'disease_4': 227, 'normal': 697}
- N match (1403): **PASS**; pos/neg (100/1303): **PASS**; reported n_positive=100, n_negative=1303 match recomputed 100/1303

### Objectness collapse RESOLVED (normal vs disease score separation)
- disease objectness: median=0.9928, p25=0.9768, min=0.4130
- normal  objectness: median=0.0043, p75=0.0072, p95=0.1095, max=0.9314
- normals scored >=0.5: **16/1303** (1.2%)
- Separation is clean: disease median ~0.99 vs normal median ~0.004. The earlier collapse (objectness ~1.0 for ALL images incl. normals) is FIXED — objectness now functions as a real image-level disease score.

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| det_pr_auc (primary) | 0.9934 | 0.9934 | ok |
| det_roc_auc | 0.9995 | 0.9995 | ok |
| presence_recall@0.5 | 0.9900 | 0.9900 | ok |
| fp_rate@0.5 | 0.0123 | 0.0123 | ok |
| iou_at_0.5_presence | 0.5600 | 0.5600 | ok |
| iou median (positives) | 0.5695 | 0.5695 | ok |
| iou mean (positives) | 0.5596 | 0.5596 | ok |
| map_at_0.5 | 0.4215 | 0.4271 | ok |

## 3) Image-level disease detection (primary interpretation)

- det_pr_auc = **0.9934**, det_roc_auc = **0.9995** (n_pos=100, n_neg=1303)
- presence_recall@0.5 (disease recall) = **0.990** = 99/100 (95% CI 0.946-0.998)
- fp_rate@0.5 (normal false alarm) = **0.0123** = 16/1303 (95% CI 0.0076-0.0199)

## 4) Localization quality (positives only, coarse-box caveat)

- IoU(pred,GT) on positive images (n=100): median=0.569, mean=0.560, p25=0.414, p75=0.727, min=0.063, max=0.994
- frac positives localized @IoU>=0.5 = 0.560
- mAP@0.5 (objectness-ranked, normal preds = FP) = **0.427** — interpret SEPARATELY from det_pr_auc: mAP couples ranking with coarse-box localization, so it is bounded by the IoU>=0.5 hit rate, not by disease/normal separability.
- NOTE: GT boxes are coarse (median ~50% of frame, center-biased). IoU median ~0.6 reflects crop-region localization, NOT fine lesion pinpointing (REPORT sec.6).

## 5) Config notes

- meta: {"train_counts": {"disease_3": 470, "disease_4": 227, "normal": 697}, "valid_counts": {"disease_3": 76, "disease_4": 24, "normal": 1303}, "include_normal": true}
