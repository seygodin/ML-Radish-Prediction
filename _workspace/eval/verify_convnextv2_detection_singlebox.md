# verify_convnextv2_detection_singlebox.md

- task: detection single-box, backbone: convnextv2
- status (reported): ok
- primary: det_pr_auc (det_pr_auc)

## 1) Boundary cross-validation (predictions vs manifest)

- predictions total valid images N = **1403** (include_normal=True)
- positives (disease, is_positive=True, non-empty GT) = **100**; negatives (normal, empty GT) = **1303**
- manifest detection valid (disease only) = **100** (d3=76 + d4=24); classification manifest valid normal = **1303** -> total **1403** matches predictions N
- reported meta valid_counts = {'disease_3': 76, 'disease_4': 24, 'normal': 1303}; train_counts = {'disease_3': 470, 'disease_4': 227, 'normal': 697}
- N match (1403): **PASS**; pos/neg (100/1303): **PASS**; reported n_positive=100, n_negative=1303 match recomputed 100/1303

### Objectness collapse RESOLVED (normal vs disease score separation)
- disease objectness: median=0.9989, p25=0.9953, min=0.2855
- normal  objectness: median=0.0000, p75=0.0001, p95=0.0026, max=0.9600
- normals scored >=0.5: **4/1303** (0.3%)
- Separation is clean: disease median ~1.00 vs normal median ~0.000. The earlier collapse (objectness ~1.0 for ALL images incl. normals) is FIXED — objectness now functions as a real image-level disease score.

## 2) Independent metric recomputation vs reported

| metric | reported | recomputed | match |
|---|---|---|---|
| det_pr_auc (primary) | 0.9977 | 0.9977 | ok |
| det_roc_auc | 0.9998 | 0.9998 | ok |
| presence_recall@0.5 | 0.9900 | 0.9900 | ok |
| fp_rate@0.5 | 0.0031 | 0.0031 | ok |
| iou_at_0.5_presence | 0.7000 | 0.7000 | ok |
| iou median (positives) | 0.6670 | 0.6670 | ok |
| iou mean (positives) | 0.6199 | 0.6199 | ok |
| map_at_0.5 | 0.6025 | 0.5991 | ok |

## 3) Image-level disease detection (primary interpretation)

- det_pr_auc = **0.9977**, det_roc_auc = **0.9998** (n_pos=100, n_neg=1303)
- presence_recall@0.5 (disease recall) = **0.990** = 99/100 (95% CI 0.946-0.998)
- fp_rate@0.5 (normal false alarm) = **0.0031** = 4/1303 (95% CI 0.0012-0.0079)

## 4) Localization quality (positives only, coarse-box caveat)

- IoU(pred,GT) on positive images (n=100): median=0.667, mean=0.620, p25=0.445, p75=0.781, min=0.054, max=0.990
- frac positives localized @IoU>=0.5 = 0.700
- mAP@0.5 (objectness-ranked, normal preds = FP) = **0.599** — interpret SEPARATELY from det_pr_auc: mAP couples ranking with coarse-box localization, so it is bounded by the IoU>=0.5 hit rate, not by disease/normal separability.
- NOTE: GT boxes are coarse (median ~50% of frame, center-biased). IoU median ~0.6 reflects crop-region localization, NOT fine lesion pinpointing (REPORT sec.6).

## 5) Config notes

- meta: {"train_counts": {"disease_3": 470, "disease_4": 227, "normal": 697}, "valid_counts": {"disease_3": 76, "disease_4": 24, "normal": 1303}, "include_normal": true}
