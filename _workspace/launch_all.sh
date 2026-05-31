#!/bin/bash
cd /home/seungeon/Workspace/side/radish
PY=./.venv/bin/python
run() {  # $1=gpu  $2=specname
  CUDA_VISIBLE_DEVICES=$1 $PY -m src.train --spec _workspace/specs/$2.yaml --device cuda:0 \
    > _workspace/launch_$2.log 2>&1 &
  echo "launched $2 on GPU$1 (pid $!)"
}
# detection (long) -> dedicated GPUs
run 1 exp_convnextv2_detection_singlebox
run 2 exp_efficientnetv2_detection_singlebox
run 3 exp_nextvit_detection_singlebox
# classification (short) -> spread
run 4 exp_convnextv2_normal_d3_d4
run 5 exp_convnextv2_normal_vs_d3
run 6 exp_convnextv2_normal_vs_d4
run 7 exp_efficientnetv2_normal_d3_d4
run 0 exp_efficientnetv2_normal_vs_d3
run 4 exp_efficientnetv2_normal_vs_d4
run 5 exp_nextvit_normal_d3_d4
run 6 exp_nextvit_normal_vs_d3
run 7 exp_nextvit_normal_vs_d4
wait
echo "ALL_12_RUNS_DONE"
