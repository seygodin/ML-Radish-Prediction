#!/bin/bash
cd /home/seungeon/Workspace/side/radish
PY=./.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY -m src.train --spec _workspace/specs/exp_$2.yaml --device cuda:0 > _workspace/launch_exp_$2.log 2>&1 & echo "launched $2 on GPU$1 (pid $!)"; }
# slow mamba (4) on distinct GPUs
run 0 mamba_normal_vs_d3
run 1 mamba_normal_vs_d4
run 2 mamba_normal_d3_d4
run 3 mamba_detection_singlebox
# detection (heavy img512) of other 3 backbones on distinct GPUs
run 4 densenet121_detection_singlebox
run 5 resnet50_detection_singlebox
run 6 nextvit20_detection_singlebox
# classification 9 spread
run 7 densenet121_normal_vs_d3
run 0 densenet121_normal_vs_d4
run 1 densenet121_normal_d3_d4
run 2 resnet50_normal_vs_d3
run 3 resnet50_normal_vs_d4
run 4 resnet50_normal_d3_d4
run 5 nextvit20_normal_vs_d3
run 6 nextvit20_normal_vs_d4
run 7 nextvit20_normal_d3_d4
wait
echo "ALL_NEW16_DONE"
