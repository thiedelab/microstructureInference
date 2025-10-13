#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

correlation_Threshold_for_Bragg_disk_detection=12000

echo "step_01_predict_orientations_from_exp_BraggDisks.py"
echo ""
python step_01_predict_orientations_from_exp_BraggDisks.py --correlationThresholdTemplateMatch "$correlation_Threshold_for_Bragg_disk_detection"


