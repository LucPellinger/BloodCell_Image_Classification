#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate image_classification
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python3 main_scripts/main_resnet50.py
