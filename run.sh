#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate image_classification
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python3 src/main_app.py
