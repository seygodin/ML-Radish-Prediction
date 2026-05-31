# verify_dinov3_base_detection_singlebox.md

- task: detection single-box (Ours), backbone: **dinov3_base** = DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3`), **frozen** + single-box+objectness head
- model: total **85.84M** params / trainable (head only) **199.7K** (≈0.2M). Backbone fully frozen (requires_grad=False).
- img_size: 512. primary: det_pr_auc. status (reported): ok. best epoch (reported) = **0**.
- recompute script: `_workspace/eval/run_ours_detection_eval.py`; dump: `_workspace/eval/ours_detection.json`.
- ⚠️ **공정성**: Ours는 DINOv3 자기지도 pretrained 백본을 frozen하고 head(0.2M)만 학습 → §3/§3B 상단 6 백본(전부 from-scratch, 전 파라미터 trainable)과 **동일 조건 비교가 아니다**.

## 1) Boundary cross-validation (predictions ↔ manifest)

- predictions total valid images N = **1403** (include_normal=True) → **PASS** (==1403).
- positives (disease, is_positive=True, non-empty GT) = **100**; negatives (normal, empty GT) = **1303** → **PASS** (100/1303).
- reported meta valid_counts = `{disease_3: 76, disease_4: 24, normal: 1303}` (= 100 disease + 1303 normal = 1403) — matches manifest detection valid (d3 76 + d4 24) + classification valid normal 1303.
- reported `final.n_positive`=100, `final.n_negative`=1303 = recomputed 100/1303 → **PASS**.
- balanced loader (`balance_valid=True`, seed=42) valid_counts = `{disease_3: 76, disease_4: 24, normal: 100}` → N=200 (pos 100 / neg 100, prevalence 0.5), positives untouched (same 100 disease) → **PASS** (matches §3B baseline procedure).

누수·오집계·split 위반 징후 **없음**.

## 2) Independent metric recomputation vs reported (orig distribution, §3)

| metric | reported (metrics.json final) | recomputed (predictions) | match |
|---|---|---|---|
| det_pr_auc (primary) | 1.0000 | 1.0000 | ok |
| det_roc_auc | 1.0000 | 1.0000 | ok |
| presence_recall@0.5 | 1.0000 | 1.0000 (100/100) | ok |
| fp_rate@0.5 | 0.0007675 (1/1303) | 0.0007675 (1/1303) | ok |
| iou median (positives) | 0.5649 | 0.5649 | ok |
| iou mean (positives) | 0.5469 | 0.5469 | ok |
| iou_at_0.5_presence | 0.6300 | 0.6300 | ok |
| map_at_0.5 | 0.4856 | 0.4906 | ok (AP 보간 방식 차 ≤0.05, baseline과 동일 성질) |

**sklearn 교차검증 (orig)**: det_pr_auc 재계산 1.0 = sklearn `average_precision_score` 1.0 (1e-9 이내); det_roc_auc 재계산 1.0 = sklearn `roc_auc_score` 1.0 (1e-9 이내) → **PASS**.

## 3) Balanced valid recompute (best.pt reload, §3B 절차)

best.pt를 forward 전용 로드(**가중치 변경 없음**), `build_detection_loaders(img_size=512, batch_size=16, num_workers=8, seed=42, include_normal=True, balance_valid=True)`, fp32 forward.

| metric | balanced (pos100/neg100) |
|---|---|
| det_pr_auc | 1.0000 |
| det_roc_auc | 1.0000 |
| presence_recall@0.5 | 1.000 (100/100) |
| fp_rate@0.5 | 0.000 (0/100) |
| map_at_0.5 | 0.4906 |
| iou median (positives) | 0.5649 |
| iou mean (positives) | 0.5469 |

**sklearn 교차검증 (balanced)**: PR-AUC 재계산 1.0 = sklearn 1.0; ROC-AUC 재계산 1.0 = sklearn 1.0 (둘 다 1e-9 이내) → **PASS**.

- presence_recall@0.5와 IoU(median/mean)는 §3(orig)과 **완전히 동일** — 양성(질병 100장)에만 의존하고 정상 다운샘플은 양성 집합을 건드리지 않으므로 불변(baseline 6종과 동일 성질, §3B 검증 통과).
- fp_rate@0.5: orig 1/1303(0.08%) → balanced 0/100. orig에서 0.5를 넘던 정상 1장이 seed=42 다운샘플 100장에 포함되지 않음. 균형 표본 fp_rate는 표본 추출 운에 민감 → 정상 오경보의 신뢰 수치는 **orig 1303 기준(1/1303 ≈ 0.08%)**.
- mAP@0.5는 orig·balanced 동일(0.4906): 단일 박스가 objectness로 랭크되며 정상 예측(전부 FP)이 양성보다 낮게 랭크돼 AP 곡선 끝에만 기여 → 음성 수 변화에 거의 불변. 국소화 hit-rate(IoU≥0.5)에 상한이 묶임.

## 4) Objectness 분리 (정상 vs 질병)

| 분포 | 질병 objectness median | 정상 objectness median | 정상 ≥0.5 |
|---|---|---|---|
| orig (100 / 1303) | **0.9552** | 0.0166 | 1/1303 (0.08%) |
| balanced (100 / 100) | **0.9552** | 0.0173 | 0/100 |

objectness가 정상/질병을 **깨끗이 분리**(질병 median 0.955 vs 정상 median ~0.017). collapse 없음. det_pr_auc=1.0은 정상 최고 점수보다 모든 양성 점수가 높음을 의미.

## 5) ⚠️ Best-epoch IoU 과소평가 주의 (중요)

- 이 run은 **det_pr_auc가 ep0부터 1.0으로 포화** → early-stop(monitor=val_det_pr_auc, mode=max)이 **best=ep0**을 선택. 따라서 저장된 best.pt/predictions의 **국소화 IoU가 전 epoch 중 최저**다.
- ep0 IoU median = **0.5649** (저장 predictions 값). 전 epoch IoU median 범위 0.5649(ep0, 최저) → 0.6429(ep29, 최고). **후반(ep≥28) IoU median ≈ 0.635, 최대 0.6429**(train.log/metrics.json per_epoch 기준).
- 즉 §3/§3B 표의 Ours IoU median 0.565는 **국소화 능력을 과소평가**한 값이며, det_pr_auc 외 기준으로 골랐다면 IoU median ≈0.64까지 보고 가능. 검출 지표(det_pr_auc 1.0 / presence 1.0 / fp ~0)는 ep0부터 포화라 epoch 선택의 영향 없음.
- 가중치는 변경하지 않았으므로(읽기 전용) 표에는 저장된 best.pt(ep0) 기준 수치를 싣고, 이 한계를 각주로 명시한다.

## 6) Config notes

- meta: `{train_counts: {disease_3:470, disease_4:227, normal:697}, valid_counts: {disease_3:76, disease_4:24, normal:1303}, include_normal:true}`
- AMP disabled (fp32 train/eval, spec optim.amp=false). loss=giou_l1_obj (giou 2.0 / l1 5.0 / obj 1.0, box_loss_on=positive_only).
